import pandas as pd
import numpy as np
import random
import collections
import csv

data = pd.read_csv('./data/automobile-customer.csv')
data.head()
columns = data.columns
values = data.values
# random.shuffle(values)

num_example = len(values)
num_train = int(num_example * 80 / 100)

train = open('train.csv', 'w')
train_w = csv.writer(train)
test = open('test.csv', 'w')
test_w = csv.writer(test)
print('num_example', num_example)
print('num_train', num_train)
print('num_test', num_example - num_train)

train_w.writerow(columns)
test_w.writerow(columns)
for i in range(0, num_example):
    row = values[i]
    if i < num_train:
        train_w.writerow(row)
    else:
        test_w.writerow(row)
