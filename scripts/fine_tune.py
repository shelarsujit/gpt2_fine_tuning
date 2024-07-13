from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def main():
    dataset = load_dataset('csv', data_files={'train': 'data/train.csv', 'test': 'data/test.csv'})
    print(dataset['train'][0])

    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer.pad_token is not None

    model = GPT2LMHeadModel.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    def adjust_labels(examples):
        examples['labels'] = examples['input_ids'].copy()
        return examples

    tokenized_dataset = tokenized_dataset.map(adjust_labels, batched=True)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test']
    )

    trainer.train()

if __name__ == "__main__":
    main()
