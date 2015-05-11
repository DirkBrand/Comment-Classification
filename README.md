# README #
Main code repo for Master's project by Dirk Brand.

### Dependencies (Python) ###

* Numpy
* Scipy
* Scikit-Learn
* NLTK
* Gensim
* Matplotlib


### Instructions ###
Make a file config.py to reflect the absolute paths of the directories where the models, data sets and feature sets are stored.  Put it inside the src folder.

It should look something like this:

model_path = r'C:\deep_learning_models\\'

comment_data_path = r'C:\Repos\comments_stuff\\'


sentiment_path = r'D:\Sentiment\\'

feature_set_path = r'C:\featuresets\\'


### TRAINING SET ###

The training set used in all computations is unfortunately private and cannot be shared.  For testing, create a file called "trainTestDataSet.txt" and place it in comment_data_path in your own config.py.

The format of this training set is an ampersand (&) seperated file with the following columns:

Comment thread id, Comment id, Parent comment id, User id, number of likes, number of dislikes, number of reports, status number (0,1 or 2), rating (not currently used), date, author, article title, article synopsis, comment body, comment bodu lemmatized, comment body pos-tagged

After the training set is created, the feature extraction package can be used to get the relevant feature sets.

### Paper Resources ###
For additional notes on individual publications, see the text files in the Notes folder.