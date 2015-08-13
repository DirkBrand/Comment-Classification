'''
Created on 12 Mar 2014

@author: Dirk
'''

class CommentObject(object):
    '''
    Comment object
    '''


    def __init__(self, id, artID, parentId, userId, likeCount, dislikeCount, reportedCount, status, date, author, body, lemma_body, pos_body):
        '''
        Constructor
        '''
        self.id = id
        self.article_id = artID
        self.parentId= parentId
        self.userId = userId
        
        if likeCount == 'null':
            self.likeCount = 0
        else:
            self.likeCount = int(likeCount)
            
        if dislikeCount == 'null':
            self.dislikeCount = 0
        else:
            self.dislikeCount = int(dislikeCount)
        
        if reportedCount == 'null':
            self.reported = 0
        else:
            self.reported = int(reportedCount)
        
        if status == 'null':
            self.status = 0
        else:
            self.status = int(status)
        
        self.rating = 0
            
        self.date = date
        self.author = author
        self.body = body
        self.lemma_body = lemma_body
        self.pos_body = pos_body
        
    def setWords(self,words):
        self.words = words
        
    def setRating(self, rating):
        self.rating = int(rating)