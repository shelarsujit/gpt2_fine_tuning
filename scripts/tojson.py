import pandas as pd
train_data = pd.read_csv('/data/train.csv')
test_data = pd.read_csv('/data/test.csv')

train_data.to_json('data/train.json', orient='records', lines=True)

# Save the testing data to test.json
test_data.to_json('data/test.json', orient='records', lines=True)