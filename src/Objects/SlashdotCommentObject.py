'''
Created on 12 Mar 2014

@author: Dirk
'''

class CommentObject(object):
    '''
    Comment object
    '''


    def __init__(self,  artId, commId, superParentId, parentId, author, score, flag, date, commentTitle, hasLink, commentBody, quotedText):
        '''
        Constructor
        '''
        self.id = commId
        self.article_id = artId
        self.superParentId= superParentId
        self.parentId= parentId
        self.userId= parentId
        self.author = author
        
        self.score = score
        self.flag = flag
            
        
        self.date = date
        self.commentTitle = commentTitle
        self.hasLink = hasLink
        self.body = commentBody
        self.quotedText = quotedText
        