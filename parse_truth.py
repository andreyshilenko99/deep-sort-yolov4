from os.path import join
import os
import pandas as pd

data_labels = pd.DataFrame(columns=['filename', 'in', 'out'])
data_labels['in'] = 2

filenames = os.listdir('data_files/')
with open('labels.txt', 'w+') as file:
    for filename in filenames:

        data_labels['filename'] = filenames
        file.write(filename)
        file.write('\n')

data_labels.to_csv('labels.csv')

res_pd = pd.DataFrame()
with open ('data_files/labels_counted.csv', 'r') as file:
    lines = file.readlines()

class CountTruth:
    def __init__(self, inside, outside):
        self.inside = inside
        self.outside = outside

TruthArr = CountTruth(0, 0)
for line in lines:
    if line[1]=="filename":
        continue
    else:
        TruthArr.inside = line[2]
        TruthArr.outside = line[3]


