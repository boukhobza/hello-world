"""
@package tweet_features
Convert tweet to feature vector.
These routines help convert arbitrary tweets in to feature vectors.
"""
import numpy
# search patterns for features
testFeatures = \
    [('hasAddict',     (' addict',)), \
     ('hasAwesome',    ('awesome',)), \
     ('hasBroken',     ('broke',)), \
     ('hasBad',        (' bad',)), \
     ('hasBug',        (' bug',)), \
     ('hasCant',       ('cant','can\'t')), \
     ('hasCrash',      ('crash',)), \
     ('hasCool',       ('cool',)), \
     ('hasDifficult',  ('difficult',)), \
     ('hasDisaster',   ('disaster',)), \
     ('hasDown',       (' down',)), \
     ('hasDont',       ('dont','don\'t','do not','does not','doesn\'t')), \
     ('hasEasy',       (' easy',)), \
     ('hasExclaim',    ('!',)), \
     ('hasExcite',     (' excite',)), \
     ('hasExpense',    ('expense','expensive')), \
     ('hasFail',       (' fail',)), \
     ('hasFast',       (' fast',)), \
     ('hasFix',        (' fix',)), \
     ('hasFree',       (' free',)), \
     ('hasFrowny',     (':(', '):')), \
     ('hasFuck',       ('fuck',)), \
     ('hasGood',       ('good','great')), \
     ('hasHappy',      (' happy',' happi')), \
     ('hasHate',       ('hate',)), \
     ('hasHeart',      ('heart', '<3')), \
     ('hasIssue',      (' issue',)), \
     ('hasIncredible', ('incredible',)), \
     ('hasInterest',   ('interest',)), \
     ('hasLike',       (' like',)), \
     ('hasLol',        (' lol',)), \
     ('hasLove',       ('love','loving')), \
     ('hasLose',       (' lose',)), \
     ('hasNeat',       ('neat',)), \
     ('hasNever',      (' never',)), \
     ('hasNice',       (' nice',)), \
     ('hasPoor',       ('poor',)), \
     ('hasPerfect',    ('perfect',)), \
     ('hasPlease',     ('please',)), \
     ('hasSerious',    ('serious',)), \
     ('hasShit',       ('shit',)), \
     ('hasSlow',       (' slow',)), \
     ('hasSmiley',     (':)', ':D', '(:')), \
     ('hasSuck',       ('suck',)), \
     ('hasTerrible',   ('terrible',)), \
     ('hasThanks',     ('thank',)), \
     ('hasTrouble',    ('trouble',)), \
     ('hasUnhappy',    ('unhapp',)), \
     ('hasWin',        (' win ','winner','winning')), \
     ('hasWinky',      (';)',)), \
     ('hasgood',    ('&lt;3',':d',':dd','8)',':-)',':)',';)','(-:','(:',)), \
     ('hasbad',    (':/',':&gt;',':'')',':-(',':(',':S',':-S')), \
     ('hasWow',        ('wow','omg')) ]
    
   
def make_tweet_nparr( txt ):
    """ 
    Extract tweet feature vector as NumPy array. 
    """
    # result storage
    fvec = numpy.empty( len(testFeatures) )
    # search for each feature
    txtLow = ' ' + txt.lower() + ' '
    for i in range( 0, len(testFeatures) ):

        key = testFeatures[i][0]

        fvec[i] = False
        for tstr in testFeatures[i][1]:
            fvec[i] = fvec[i] or (txtLow.find(tstr) != -1)

    return fvec

def preProcess(tweet):
    # process the tweets
    
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet

'''emo_repl = {
    # positive emoticons
    "&lt;3": " good ",
    ":d": " good ", # :D in lower case
    ":dd": " good ", # :DD in lower case
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",
    # negative emoticons:
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":S": " bad ",
    ":-S": " bad ",
    r"\br\b": "are",
r"\bu\b": "you",
r"\bhaha\b": "ha",
r"\bhahaha\b": "ha",
r"\bdon't\b": "do not",
r"\bdoesn't\b": "does not",
r"\bdidn't\b": "did not",
r"\bhasn't\b": "has not",
r"\bhaven't\b": "have not",
r"\bhadn't\b": "had not",
r"\bwon't\b": "will not",
r"\bwouldn't\b": "would not",
r"\bcan't\b": "can not",
r"\bcannot\b": "can not",
    }'''
#return fvec


def make_tweet_dict( txt ):
    """ 
    Extract tweet feature vector as dictionary. 
    """
    txtLow = ' ' + txt.lower() + ' '

    # result storage
    fvec = {}

    # search for each feature
    for test in testFeatures:

        key = test[0]

        #fvec[key] = False;
        fvec[key] = 0;

        for tstr in test[1]:
            fvec[key] = fvec[key] or (txtLow.find(tstr) != -1)
        #affichage de dic sons sentiment print fvec
            #if (fvec[key]==True): print (key)   #moiiii
               
    return fvec


def tweet_dict_to_nparr( dict ):
    """
    Convert dictionary feature vector to numpy array
    """
    fvec = numpy.empty( len(testFeatures) )
    print 

    for i in range( 0, len(testFeatures) ):
        fvec[i] = dict[ testFeatures[i][0] ]
    return fvec



def tweet_nparr_to_dict( nparr, use_standard_features=False ):
    """
    Convert NumPy array to dictionary
    """
    fvec = {}

    if use_standard_features:
        assert len(nparr) == len(testFeatures)
        fvec = {}
        for i in range( 0, len(nparr) ):
            fvec[ testFeatures[i][0] ] = nparr[i]

    else:
        for i in range( 0, len(nparr) ):
            fvec[ str(i) ] = nparr[i]
           
   
    return fvec


def is_zero_dict( dict ):
    """
    Identifies empty feature vectors
    """
    has_any_features = False
    for key in dict:
        has_any_features = has_any_features or dict[key]
    print(has_any_features)
    return not has_any_features
