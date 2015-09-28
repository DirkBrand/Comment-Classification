'''
Created on 29 Jul 2015

@author: Dirk
'''
from FeatureExtraction.mainExtractor import read_slashdot_comments,\
    read_comments

from config import comment_data_path


if __name__ == '__main__':
    articleList, commentList, parList, commentCount = read_comments(comment_data_path + 'trainTestDataSet.txt', skip=False)
    #articleList, commentList, commentCount = read_slashdot_comments(comment_data_path + 'slashdotDataSet.txt', skip=False)
    
    out = open(comment_data_path + 'News24CommData.txt', 'w')
    #out = open(comment_data_path + 'slashdotCommData.txt', 'w')
    
    for art in commentList.items():        
        for comm in art[1]:
            out.write(comm.id + "|" + comm.body + "\n")
    
    out.close()
    print "Done with file"