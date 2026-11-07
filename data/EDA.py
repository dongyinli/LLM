import pandas as pd
from collections import Counter

from streamlit import title

from config import Config

conf = Config()

train_data = pd.read_csv(conf.train_datapath, sep='\t',names=['content', 'label'])
test_data = pd.read_csv(conf.test_datapath, sep='\t',names=['content', 'label'])
val_data = pd.read_csv(conf.val_datapath, sep='\t',names=['content', 'label'])

print(train_data.head())
print(test_data.head())
print(val_data.head())

print(f'train data size:{len(train_data)}')
print(f'test data size:{len(test_data)}')
print(f'val data size{len(val_data)}')

count_label = Counter(train_data['label'])
print(count_label)

train_data["text_length"] = train_data['content'].str.len()
print(train_data.head())

print(f'max text length:{train_data["text_length"].max()}')
print(f'min text length:{train_data["text_length"].min()}')
print(f'avg text length:{train_data["text_length"].mean():.2f}')
print(f'std text length:{train_data["text_length"].std():.2f}')
