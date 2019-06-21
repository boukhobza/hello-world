"""
@package sentiment
Twitter sentiment analysis.
This code performs sentiment analysis on Tweets.
A custom feature extractor looks for key words and emoticons.  These are fed in
to a naive Bayes classifier to assign a label of 'positive', 'negative', or
'neutral'.  Optionally, a principle components transform (PCT) is used to lessen
the influence of covariant features.
"""
import csv, random
import nltk
import tweet_features, tweet_pca

# read all tweets and labels
fp = open( 'full-corpus.csv', 'rb' )
reader = csv.reader( fp, delimiter=',', quotechar='"', escapechar='\\' )
tweets = []
for row in reader:
    tweets.append([row[4], row[1]]);
    """try:
        #if row[1]!='irrelevant':
            tweets.append( [row[0].encode('utf-8',"replace"), row[1] ])
    except UnicodeDecodeError: 
        pass"""

#print tweets
# treat neutral and irrelevant the same
for t in tweets:
    if t[1] == 'irrelevant':
        t[1] = 'neutral'

# split in to training and test sets
random.shuffle( tweets );
fvecs = [(tweet_features.make_tweet_dict(t),s) for (t,s) in tweets]

#print fvecs
v_train = fvecs[:2500]
v_test  = fvecs[2500:]

# dump tweets which our feature selector found nothing
#for i in range(0,len(tweets)):
#    if tweet_features.is_zero_dict( fvecs[i][0] ):
#        print tweets[i][1] + ': ' + tweets[i][0]

# apply PCA reduction
#(v_train, v_test) = tweet_pca.tweet_pca_reduce( v_train, v_test, output_dim=1.0 )
#print v_train 

# train classifier
classifier = nltk.NaiveBayesClassifier.train(v_train);
#print classifier
#classifier = nltk.classify.maxent.train_maxent_classifier_with_gis(v_train);

# classify and dump results for interpretation
#print '\nAccuracy %f\n' % nltk.classify.accuracy(classifier, v_test)
#print classifier.show_most_informative_features(10)

# build confusion matrix over test set
#print t
test_truth   = [s for (t,s) in v_test]
#t le dic
#print t
#s senti
#print s
#print test_truth
test_predict = [classifier.classify(t) for (t,s) in v_test]
#print test_predict
#print t
#print s
#print test_predict

print ('Confusion Matrix')
print (nltk.ConfusionMatrix( test_truth, test_predict ))



