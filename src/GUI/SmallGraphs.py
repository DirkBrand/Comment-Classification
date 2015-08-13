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


#articleList, commentList, commentCount = read_slashdot_comments(comment_data_path + 'slashdotDataSet.txt')
articleList, commentList, parList, commentCount = read_toy_comments(comment_data_path + 'trainTestDataSet.txt', comment_data_path + 'toyComments.csv')
labels = []
i = 0
for art in commentList.items():        
    for comm in art[1]:
        labels.append(comm.rating)
        i += 1
        if i % 1000 == 0:
            print i
            
dist = Counter(labels)
print "Class distribution:", dist

keys = dist.keys()
keys.sort()

pos = np.arange(len(keys))
width = 0.25     # gives histogram aspect to the bar diagram
ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(keys)

plt.bar(pos, dist.values(), width, color='b')
plt.xlim([min(pos) - 0.5, max(pos) + 0.5])
plt.xlabel("Rating")
plt.ylabel("Number of Comments")
plt.savefig(r'Images\toydistribution.png', bbox_inches='tight')

'''
plt.hist(, bins=dist.keys())
plt.show()'''