import csv

# 文件路径
txt_file_path = '/data/liaowen/SGcCA/dataset/bindingdb.txt'
csv_file_path = '/data/liaowen/SGcCA/dataset/bindingdb.csv'

# 分隔符（根据你的TXT文件的格式进行调整）
delimiter = ' '  # 可以是空格 ' '，或者其他分隔符

# 定义表头
headers = ['SMILES', 'Protein', 'Y']

# 读取TXT文件
with open(txt_file_path, 'r') as txt_file:
    lines = txt_file.readlines()

# 写入CSV文件
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # 写入表头
    csv_writer.writerow(headers)

    for line in lines:
        # 去掉行尾的换行符并根据分隔符进行分割
        row = line.strip().split(delimiter)
        # 检查行是否有三列数据
        if len(row) == 3:
            # 写入CSV文件
            csv_writer.writerow(row)

print("TXT文件已成功转换为带表头的CSV文件。")




import pandas as pd
from copy import copy
import os

dataFolder = f'./dataset/'
datasets_path = os.path.join(dataFolder, "bindingdb.txt")
datasets = pd.read_csv(datasets_path, header=None, sep=' ')
datasets.columns = ['SMILES', 'Protein', 'Y']
# Load your dataset
data = copy(datasets)

# Define the target drug SMILES string
target_drug_smiles = 'CN[C@@H]1C[C@H]2O[C@@](C)([C@@H]1OC)n1c3ccccc3c3c4c(c5c6ccccc6n2c5c31)C(=O)NC4'

# Split the dataset into training and testing sets
dataset_test = data[data['SMILES'] == target_drug_smiles]
dataset_train = data[data['SMILES'] != target_drug_smiles]

# Further split the testing set to create validation and test sets
test_size = len(dataset_test)
train_set = dataset_train
val_set = dataset_test[: int(test_size * 1 / 3)]
test_set = dataset_test[int(test_size * 1 / 3):]

# Output the shapes of the resulting datasets
print(f'Train set size: {len(train_set)}')
print(f'Validation set size: {len(val_set)}')
print(f'Test set size: {len(test_set)}')

# Save the datasets to new CSV files if needed
train_set.to_csv('train_set.csv', index=False)
val_set.to_csv('val_set.csv', index=False)
test_set.to_csv('test_set.csv', index=False)