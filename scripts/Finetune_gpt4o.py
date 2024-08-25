import openai
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Function to upload the dataset
def upload_dataset(file_path):
    response = openai.File.create(file=open(file_path), purpose='fine-tune')
    return response['id']

# Function to create fine-tuning job
def create_fine_tune(training_file_id, model="gpt-4-0613"):
    response = openai.FineTune.create(
        training_file=training_file_id,
        model=model
    )
    return response['id']

# Function to monitor fine-tuning job
def monitor_fine_tune(fine_tune_id):
    while True:
        status = openai.FineTune.retrieve(id=fine_tune_id)
        status = status['status']
        if status in ['succeeded', 'failed']:
            print(f"Fine-tuning job {status}")
            break
        print(f"Fine-tuning job status: {status}")
        time.sleep(60)

# Function to generate text using the fine-tuned model
def generate_text(fine_tuned_model_id, prompt):
    completion = openai.Completion.create(
        model=fine_tuned_model_id,
        prompt=prompt,
        max_tokens=50
    )
    return completion.choices[0].text

# Upload dataset
training_file_id = upload_dataset("training_data.jsonl")
print(f"Uploaded dataset with file ID: {training_file_id}")

# Create fine-tuning job
fine_tune_id = create_fine_tune(training_file_id)
print(f"Created fine-tuning job with ID: {fine_tune_id}")

# Monitor fine-tuning job
monitor_fine_tune(fine_tune_id)

# Get fine-tuned model ID
fine_tuned_model_id = openai.FineTune.retrieve(id=fine_tune_id)['fine_tuned_model']
print(f"Fine-tuned model ID: {fine_tuned_model_id}")

# Generate text using the fine-tuned model
prompt = "Translate the following English text to French: 'Good night!'"
generated_text = generate_text(fine_tuned_model_id, prompt)
print(f"Generated text: {generated_text}")
