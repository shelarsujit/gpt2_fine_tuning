import os
import logging
import sys
import torch
import transformers
import loralib as lora
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Hyperparameters
training_config = {
    "bf16": True,
    "do_eval": False,
    "learning_rate": 5.0e-06,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "max_steps": -1,
    "output_dir": "./checkpoint_dir",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 4,
    "per_device_train_batch_size": 4,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2,
}

peft_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
    "modules_to_save": None,
}

# Model and tokenizer loading
checkpoint_path = "microsoft/Phi-3-mini-4k-instruct"
model_kwargs = {
    "use_cache": False,
    "trust_remote_code": True,
    "attn_implementation": "flash_attention_2",
    "torch_dtype": torch.bfloat16,
    "device_map": None
}
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

dataset = load_dataset('json', data_files={'train': 'data/train.json', 'test': 'data/test.json'})
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Adjust labels
def adjust_labels(examples):
    examples['labels'] = examples['input_ids'].copy()
    return examples

tokenized_dataset = tokenized_dataset.map(adjust_labels, batched=True)

# Training arguments
train_conf = TrainingArguments(**training_config)
peft_conf = LoraConfig(**peft_config)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    peft_config=peft_conf
)

# Fine-tune the model
trainer.train()
