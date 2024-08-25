import os
from random import randrange
from dotenv import load_dotenv

import torch
from datasets import load_dataset, load_metric
from huggingface_hub import login
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    pipeline
)
from trl import SFTTrainer

import wandb
import numpy as np

# Install necessary packages
os.system("pip install -qqq --upgrade bitsandbytes transformers peft accelerate datasets trl flash_attn")
os.system("pip install huggingface_hub")
os.system("pip install python-dotenv")
os.system("pip install wandb -qqq")
os.system("pip install absl-py nltk rouge_score")

# Model and dataset parameters
model_id = "microsoft/Phi-3-mini-4k-instruct"
dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
dataset_split = "train"
new_model = "phi3-mini-4k-qlora-python-code-20k"
hf_model_repo = "sujit/" + new_model
device_map = {"": 0}

# BitsAndBytes parameters
use_4bit = True
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_quant_type = "nf4"
use_double_quant = True

# LoRA parameters
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05
target_modules = ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]

# Set seed for reproducibility
set_seed(1234)

# Load the environment variables and login to Hugging Face Hub
load_dotenv()
login(token=os.getenv("HF_HUB_TOKEN"))

# Load dataset from the hub
dataset = load_dataset(dataset_name, split=dataset_split)
print(f"dataset size: {len(dataset)}")
print(dataset[randrange(len(dataset))])

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right'

# Dataset preparation functions
def create_message_column(row):
    messages = []
    user = {"content": f"{row['instruction']}\n Input: {row['input']}", "role": "user"}
    messages.append(user)
    assistant = {"content": f"{row['output']}", "role": "assistant"}
    messages.append(assistant)
    return {"messages": messages}

def format_dataset_chatml(row):
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}

# Prepare the dataset
dataset_chatml = dataset.map(create_message_column)
dataset_chatml = dataset_chatml.map(format_dataset_chatml)
dataset_chatml = dataset_chatml.train_test_split(test_size=0.05, seed=1234)

# Determine compute dtype and attention implementation
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    attn_implementation = 'flash_attention_2'
else:
    compute_dtype = torch.float16
    attn_implementation = 'sdpa'

print(attn_implementation)
print(compute_dtype)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, add_eos_token=True, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'left'

# Set the quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_double_quant,
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=compute_dtype, trust_remote_code=True, quantization_config=bnb_config, device_map=device_map,
    attn_implementation=attn_implementation
)
model = prepare_model_for_kbit_training(model)

# LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules
)

# Define the training arguments
args = TrainingArguments(
    output_dir="./phi-3-mini-QLoRA",
    evaluation_strategy="steps",
    do_eval=True,
    optim="paged_adamw_8bit",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=4,
    log_level="debug",
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=1e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    eval_steps=100,
    num_train_epochs=3,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    report_to="wandb",
    seed=42,
)

# Initialize WandB
wandb.login()
os.environ["WANDB_PROJECT"] = "Phi3-mini-ft-python-code"
project_name = "Phi3-mini-ft-python-code"
wandb.init(project=project_name, name="phi-3-mini-qft-py-3e")

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_chatml['train'],
    eval_dataset=dataset_chatml['test'],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=args,
)

# Train the model
trainer.train()

# Save the model locally
trainer.save_model()

# Push the adapter to the Hub
hf_adapter_repo = "sujit/phi3-mini-adapter-ql-py-code"
trainer.push_to_hub(hf_adapter_repo)

# Clean up VRAM
del model
del trainer
import gc
gc.collect()
torch.cuda.empty_cache()
gc.collect()

# Load the model from the hub
model_name, hf_adapter_repo, compute_dtype = hf_model_repo, "sujit/phi-3-mini-QLoRA", torch.bfloat16
peft_model_id, tr_model_id = hf_adapter_repo, model_name

model = AutoModelForCausalLM.from_pretrained(tr_model_id, trust_remote_code=True, torch_dtype=compute_dtype)
model = PeftModel.from_pretrained(model, peft_model_id)
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

# Save the merged model to the Hub
model.push_to_hub(hf_model_repo)
tokenizer.push_to_hub(hf_model_repo)

# Test the model
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def test_inference(prompt):
    prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95, max_time=180)
    return outputs[0]['generated_text'][len(prompt):].strip()

# Calculate Rouge metrics
rouge_metric = load_metric("rouge", trust_remote_code=True)

def calculate_rogue(row):
    response = test_inference(row['messages'][0]['content'])
    result = rouge_metric.compute(predictions=[response], references=[row['output']], use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result['response'] = response
    return result

# Evaluate the model
metricas = dataset_chatml['test'].select(range(0, 500)).map(calculate_rogue, batched=False)
print("Rouge 1 Mean: ", np.mean(metricas['rouge1']))
print("Rouge 2 Mean: ", np.mean(metricas['rouge2']))
print("Rouge L Mean: ", np.mean(metricas['rougeL']))
print("Rouge Lsum Mean: ", np.mean(metricas['rougeLsum']))

# Generate outputs for a batch of prompts
num_samples = 500
prompts = [pipe.tokenizer.apply_chat_template([{"role": "user", "content": dataset_chatml['test'][i]['messages'][0]['content']}], tokenize=False, add_generation_prompt=True) for i in range(num_samples)]
outputs = pipe(prompts, batch_size=4, max_new_tokens=256, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95, max_time=180)
preds = [outputs[i][0]['generated_text'].split("\n")[1].strip() for i in range(len(outputs))]
references = [dataset_chatml['test'][i]['output'] for i in range(len(outputs))]
rouge_metric.add_batch(predictions=preds, references=references)
result = rouge_metric.compute(use_stemmer=True)

print("Rouge 1 Mean: ", np.mean(result['rouge1']))
print("Rouge 2 Mean: ", np.mean(result['rouge2']))