import pandas as pd

# Load the raw CSV data
train_df = pd.read_csv('data/train.csv', header=None, names=['text'])
test_df = pd.read_csv('data/test.csv', header=None, names=['text'])

def clean_and_combine(df):
    combined_texts = []
    current_text = ""
    for index, row in df.iterrows():
        text = row['text']
        if pd.notna(text):
            current_text += " " + text.strip()
        else:
            if current_text:
                combined_texts.append(current_text.strip())
                current_text = ""
    if current_text:
        combined_texts.append(current_text.strip())
    return pd.DataFrame(combined_texts, columns=['text'])

clean_train_df = clean_and_combine(train_df)
clean_test_df = clean_and_combine(test_df)

clean_train_df.to_csv('data/train.csv', index=False)
clean_test_df.to_csv('data/test.csv', index=False)

print("Cleaned train and test CSV files have been created.")
