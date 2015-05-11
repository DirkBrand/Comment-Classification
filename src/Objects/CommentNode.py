'''
Created on 12 Mar 2014

@author: Dirk
'''

class CommentNode(object):
    '''
    Comment object
    '''



    def __init__(self, usrid, commid,  parentid, author):
        '''
        Constructor
        '''
        self.userid = usrid
        self.parentid = parentid
        self.commid = commid
        self.author = author
        