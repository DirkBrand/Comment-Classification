'''
Created on 30 Jul 2015

@author: Dirk
'''
from collections import Counter

from FeatureExtraction.mainExtractor import read_slashdot_comments,\
    read_toy_comments

from config import comment_data_path
import matplotlib.pyplot as plt
import numpy as np


articleList, commentList, commentCount = read_slashdot_comments(comment_data_path + 'slashdotDataSet.txt', skip=False)
#articleList, commentList, parList, commentCount = read_toy_comments(comment_data_path + 'trainTestDataSet.txt', comment_data_path + 'toyComments.csv')
labels_anon = dict()
labels_anon[-1],labels_anon[0],labels_anon[1],labels_anon[2],labels_anon[3],labels_anon[4],labels_anon[5] = 0,0,0,0,0,0,0
labels_not_anon = dict()
labels_not_anon[-1],labels_not_anon[0],labels_not_anon[1],labels_not_anon[2],labels_not_anon[3],labels_not_anon[4],labels_not_anon[5] = 0,0,0,0,0,0,0


i = 0
for art in commentList.items():        
    for comm in art[1]:
        labels_anon[int(comm.score)] += 1
        i += 1
        if i % 1000 == 0:
            print i
            
print "Class distribution:", labels

keys = labels.keys()
keys.sort()
values = []
values.append(labels[-1])
values.append(labels[0])
values.append(labels[1])
values.append(labels[2])
values.append(labels[3])
values.append(labels[4])
values.append(labels[5])


pos = np.arange(len(keys))
width = 0.25     # gives histogram aspect to the bar diagram
ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(keys)

plt.bar(pos, values, width, color='b')
plt.xlim([min(pos) - 0.5, max(pos) + 0.5])
plt.xlabel("Rating")
plt.ylabel("Number of Comments")
plt.savefig(r'Images\slashdotdistribution.png', bbox_inches='tight')

'''
plt.hist(, bins=dist.keys())
plt.show()'''