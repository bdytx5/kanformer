import os
import numpy as np
from datasets import load_dataset
import tiktoken

# Load the dataset from Hugging Face
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split='train')

# Combine the relevant columns into a single text column
dataset = dataset.map(lambda x: {'text': x['instruction'] + ' ' + x['input'] + ' ' + x['output']})

# Get the text data as a list of strings
data = dataset['text']

# Split the data into train and validation sets
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(' '.join(train_data))
val_ids = enc.encode_ordinary(' '.join(val_data))

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))