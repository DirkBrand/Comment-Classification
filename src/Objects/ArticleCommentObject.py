'''
Created on 12 Mar 2014

@author: Dirk
'''

class ArticleCommentObject(object):
    '''
    Comment object
    '''


    def __init__(self, id, author, body, artBody):
        '''
        Constructor
        '''
        self.id = id
        self.body = body
        self.artBody = artBody
        self.author = author
        
    def setWords(self,words):
        self.words = words