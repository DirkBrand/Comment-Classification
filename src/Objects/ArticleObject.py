'''
Created on 12 Mar 2014

@author: Dirk
'''

class ArticleObject(object):
    '''
    Article object that contains a list of comments
    '''
    

    def __init__(self, id,  title, synopsis, body):
        '''
        Constructor
        '''
        self.id = id
        self.title = title
        self.synopsis= synopsis
        self.body = body
        
        
        
    def appendComment(self, comment):
        self.comments.append(comment)
        