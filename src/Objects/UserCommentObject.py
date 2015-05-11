'''
Created on 12 Mar 2014

@author: Dirk
'''

class UserCommentObject(object):
    '''
    Comment object
    '''


    def __init__(self, usrid, commid,  parentid, author, likeCount, dislikeCount, body):
        '''
        Constructor
        '''
        self.userid = usrid
        self.parentid = parentid
        self.commid = commid
        
        if likeCount == 'null':
            self.likes = 0
        else:
            self.likes = int(likeCount)
            
        if dislikeCount == 'null':
            self.dislikes = 0
        else:
            self.dislikes = int(dislikeCount)
        
        self.author = author
        self.body = body
        