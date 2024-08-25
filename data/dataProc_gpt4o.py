from datasets import load_dataset
import json

# Load the WikiText-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Prepare the JSONL file
def prepare_jsonl(data, file_name):
    with open(file_name, 'w') as f:
        for item in data:
            text = item["text"].strip()
            if text:  # Check if the text is not empty
                json_obj = {
                    "prompt": "",
                    "completion": text
                }
                f.write(json.dumps(json_obj) + "\n")

# Use the training split for fine-tuning
prepare_jsonl(dataset['train'], 'training_data.jsonl')
