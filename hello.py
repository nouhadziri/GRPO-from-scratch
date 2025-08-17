import os
from pathlib import Path

# Set the environment variables for HuggingFace
# This is done to ensure that the cache directory for HuggingFace is set to a specific location,
# preventing the storage from being overwhelmed with model files and other data.
SCRATCH = Path.home() / "scratch"
os.environ["HF_HOME"] = str(SCRATCH / "hf_home")



import gc
import re
import time
from typing import Any, Dict, List, Tuple, Union

import deepspeed
import numpy as np
import torch
from datasets import load_dataset
from deepspeed import DeepSpeedEngine
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams


import wandb
# from utils import (
# compute_token_log_probs,
# dump_episodes,
# evaluate_on_test_set,
# find_free_port,
# find_last_checkpoint,
# prepare_model_inputs,
# load_model_into_vllm
# )

# Needed to stop DeepSpeed from complaining
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = str(find_free_port())
# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"


# Let's define the hyperparameters for the training. These are mostly taken from Mini-R1 implementation.

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-3B"
MODEL_CHAT_NAME = MODEL_NAME + "-Instruct"

# Dataset configuration
DATASET_NAME = "Jiayi-Pan/Countdown-Tasks-3to4"

# Total number of training iterations
NUM_ITERATIONS = 1000
# Number of episodes to collect per iteration for training
EPISODES_PER_ITERATION = 64
# Number of responses to generate for each input prompt (i.e. group size in GRPO)
GENERATIONS_PER_SAMPLE = 4
# Controls how much the policy can deviate from the reference model
KL_COEFFICIENT = 0.001

# Training hyperparameters
# Batch size for each GPU device during training
PER_DEVICE_BATCH_SIZE = 4
# Learning rate for model updates
LEARNING_RATE = 1e-6


# Sampling parameters
# Maximum number of tokens to generate in each response
MAX_RESPONSE_TOKENS = 1024
# Controls randomness in generation (higher = more random)
TEMPERATURE = 1.0
# Nucleus sampling parameter (1.0 = disabled)
TOP_P = 1.0
# Top-k sampling parameter (-1 = disabled)
TOP_K = -1  # no top k

# DeepSpeed configuration
# DeepSpeed config for the policy model
deepspeed_config = {
"bf16": {"enabled": True},
"zero_optimization": {"stage": 2, "overlap_comm": False},
"train_batch_size": EPISODES_PER_ITERATION,
"train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
"gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
"gradient_clipping": 1.0,
"optimizer": {
    "type": "AdamW",
    "params": {
        "lr": LEARNING_RATE,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "torch_adam": True,
    },
},
}
# DeepSpeed config for the reference model
ref_deepspeed_config = {
"bf16": {"enabled": True},
# Note that we don't train the reference model
# These are just for compatibility with DeepSpeed.
"train_batch_size": EPISODES_PER_ITERATION,
"train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
"gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
}

RUN_NAME = "r1-zero"
EXP_DIR = SCRATCH / "deepseek_r1z_hackathon" / RUN_NAME
EXP_DIR.mkdir(parents=True, exist_ok=True)
print(f"Logs and Checkpoints will be saved to: {EXP_DIR}")

SYSTEM_MESSAGE = (
"You are a helpful assistant. You first think about the reasoning process in the mind "
"and then provide the user with the answer."
)
PROMPT_TEMPLATE = (
"Using the numbers {numbers}, create an equation that equals {target}. "
"You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
"Show your work in <think> </think> tags. And return the final equation and answer in "
"<answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."
)

# Load and process dataset
def preprocess_example(example: Dict[str, Any]):
    numbers: List[int] = example["nums"]
    target: int = example["target"]

    prefix = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": PROMPT_TEMPLATE.format(numbers=numbers, target=target)},
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    input_ids = tokenizer.apply_chat_template(
        prefix, tokenize=True, continue_final_message=True
    )
    prompt = tokenizer.decode(
        input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return {"prompt": prompt, "input_ids": input_ids}

# Note that the base model and "instruct" model have different eos token. 
# Here we make sure to use the correct one.
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHAT_NAME)
EOS_TOKEN_ID = AutoTokenizer.from_pretrained(MODEL_NAME).eos_token_id
EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)

dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.map(preprocess_example, num_proc=6)

print(" #### Example: ####", dataset[0])

# Split dataset
train_test_split = dataset.train_test_split(test_size=500, seed=42)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

print(len(train_dataset))
print(len(test_dataset))

print("Target: ", train_dataset[0]["target"])
print("Available Numbers: ", train_dataset[0]["nums"])

print(train_dataset[0]["prompt"])

def format_reward_func(completion: str) -> float:
    """
    Format: <think>...</think>\n</answer>...</answer>

    Also checks that the content within <answer>...</answer> conforms to a
    specified pattern (only digits, + - * / ( ) . and whitespace).

    Args:
        completion (str): Generated output

    Returns:
        float: Reward score
    """
    # Define the allowed pattern (only numbers, +, -, *, /, (, ), ., and whitespace)
    allowed_pattern = r"^[\d+\-*/().\s]+$"

    try:
        # add synthetic <think> as its already part of the prompt and prefilled 
        # for the assistant to more easily match the regex
        completion = "<think>" + completion

        # Strip EOS token if present
        if completion.endswith(EOS_TOKEN):
            completion = completion[:-len(EOS_TOKEN)]

        # Check if the format is correct
        # Pattern means:
        # 1) <think>...contents not including other <think> tags...</think>
        # 2) \n
        # 3) <answer>...anything...</answer>
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)

        if match is None or len(match.groups()) != 2:
            # Format is incorrect
            return 0.0
        else:
            # Extract the content inside <answer>...</answer>
            answer_content = match.group(2).strip()

            # Check if answer content matches the allowed pattern
            if not re.match(allowed_pattern, answer_content):
                # If it doesn't match, reward is 0.5
                return 0.5
            else:
                # If both format and pattern are correct, reward is 1
                return 1.0
    except Exception:
        # Any error leads to 0 reward
        return 0.0


def equation_reward_func(completion: str, nums: List[int], target: int) -> float:
    """
    Evaluates completion based on mathematical correctness of the answer

    Args:
        completion (str): Generated output
        target (str): Expected answer
        nums (list): Available numbers to use in the equation

    Returns:
        float: Reward score
    """
    try:
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0
    except Exception:
        # If evaluation fails, reward is 0
        return 0.0
    

def compute_reward(completion: str, sample: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    nums = sample["nums"]
    target = sample["target"]

    format_reward = format_reward_func(completion)
    equation_reward = equation_reward_func(
        completion=completion, nums=nums, target=target
    )

    reward = format_reward + equation_reward

    metrics = {
        "format_reward": format_reward,
        "equation_reward": equation_reward,
    }   

    return reward, metrics


print(format_reward_func("<think>I think the answer is </think>\n<answer>1+2</answer>"))
print(format_reward_func("I think the answer is </think>\n<answer>1+2</answer>"))
print(format_reward_func("<think>I think the<think>and even more</think> answer is </think>\n<answer>1+2</answer>")
)
print(equation_reward_func("I think the answer is </think>\n<answer>1+2+2</answer>", [1,2], 3)
)