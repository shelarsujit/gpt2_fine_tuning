from datasets import load_dataset
import pandas as pd

# Load the wikitext-2 dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

# Convert the dataset to pandas DataFrame
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Save the DataFrames to CSV files
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print("Train and test CSV files have been created.")
