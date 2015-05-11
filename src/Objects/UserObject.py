'''
Created on 12 Mar 2014

@author: Dirk
'''

correction= 3
class UserObject(object):


    def __init__(self, usrid, name, comments):
        self.id = usrid
        self.name = name
        self.comments = comments        
        
        self.likeSum = 0
        self.dislikeSum = 0
        ratingSum = 0
        for comm in comments:
            self.likeSum += comm.likes
            self.dislikeSum += comm.dislikes
            ratingSum += (comm.likes + correction) / (float(comm.dislikes + comm.likes + 2*correction) )
        
        self.totalVotes = self.likeSum+self.dislikeSum
        self.aveVotes = self.totalVotes/len(comments)
        self.aveLikes = self.likeSum/len(comments)
        self.aveDislikes = self.dislikeSum/len(comments)
        self.aveRating =  ratingSum/len(comments)
        self.vector = dict()
        self.bagOfWords = []
        
    def setVector(self,v):
        self.vector = v
        
        