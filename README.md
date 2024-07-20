# GPT-2 Fine-Tuning 

This project fine-tunes a small GPT-2 model using the Hugging Face Transformers library.

## Summary Table

| Model       | Parameters | GPU Memory | GPUs           | Batch Size | Sequence Length | Disk Space | Training Time      |
|-------------|-------------|------------|----------------|------------|-----------------|------------|--------------------|
| GPT-2       | 124M        | 8-16GB     | 1 V100 or A100 | 2-4        | 1024 tokens     | 10GB       | Several hours-days |
| DistilGPT-2 | 82M         | 4-8GB      | 1 T4 or RTX 2080| 4-8        | 1024 tokens     | 5GB        | Several hours      |
| Pythia      | 160M        | 8-16GB     | 1 V100 or A100 | 2-4        | 1024 tokens     | 10GB       | Several hours-days |
| Phi-3       | 1.4B        | 24-32GB    | 1-2 A100       | 1-2        | 2048 tokens     | 20GB       | 1-3 days           |
| Phi-3 Mini  | 350M        | 16-24GB    | 1 V100 or A100 | 2-4        | 2048 tokens     | 10GB       | Several hours-1 day|

### Notes

- **Memory Requirements**: These estimates include space for model parameters, optimizer states, and activations.
- **Training Time**: Highly dependent on dataset size, complexity, and specific computational setup.
- **Batch Size and Sequence Length**: Adjusting these can significantly impact memory requirements and training time.


## Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/shelarsujit/gpt2_fine_tuning.git
   cd gpt2-fine_tuning

2. Install the dependencies:
    ```sh
    pip install -r requirements.txt

3. Add your training and testing data in the data directory.
    ```sh
    python scripts/data_processing.py

4. Fine-Tuning
    To fine-tune the specific model, run the following command:
    ```sh
    python scripts/fine_tune_distilgpt2.py
    python scripts/fine_tune_gpt2.py
    python scripts/fine_tune_phi3.py
    python scripts/fine_tune_pythia.py
    python scripts/finetune_phi3_mini.py

