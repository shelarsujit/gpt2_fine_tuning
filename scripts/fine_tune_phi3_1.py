import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from utils import tokenize_function

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    dataset = load_dataset("the_pile", split="train")

    dataset = dataset.shuffle(seed=42).select(range(10000))

    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a small language model")
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m', help='Pretrained model name')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save the fine-tuned model')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    args = parser.parse_args()

    main(args)
