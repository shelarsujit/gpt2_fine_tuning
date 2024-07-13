def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
