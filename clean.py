import pandas as pd
from sklearn.model_selection import train_test_split

# Loading the Dataset
data = pd.read_csv('data/data.csv')

# Performing stratified train-test split + random_state for reproducibility
train_data, test_data = train_test_split(data, test_size=0.3, stratify=data['category'], random_state=20)

# saving the split into separate csv files
train_data.to_csv('./data/\\train.csv', index=False) 
test_data.to_csv('./data/\\test.csv', index=False)