#Inspired by https://blog.chapagain.com.np/python-nltk-twitter-sentiment-analysis-natural-language-processing-nlp/

import re
from random import shuffle
import nltk
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_english = stopwords.words('english')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from nltk import classify
from nltk import NaiveBayesClassifier
import twitterAuth
import tweepy
import json

pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')
all_tweets = twitter_samples.strings('tweets.20150430-223406.json')

#Create set of emoticons
pos_emoticons = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])

neg_emoticons = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])

emoticons  = pos_emoticons.union(neg_emoticons)

def clean_tweets(tweet):
    #preserve_case - convert to lower case (if False)
    #strip_handles - removing Twtter handles
    #reduce_len - reduce len of words (heyyyyy -> hey)
    tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

    #Remove tickers which starts with $
    tweet = re.sub(r'\$\w*', '', tweet)

    #Remove old retweet text RT
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    #Remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    #Remove hashtags
    tweet = re.sub(r'#', '', tweet)

    tokenized_tweet = tweet_tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tokenized_tweet:
        #Remove stopwords (the, a, an etc.) and emoticons
        if word not in stopwords_english and word not in emoticons:
            #Stem word - method which cut words like 'worker', 'workers', 'working' to one word - 'work'
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean

#Make a BoW
def bag_of_words(tweet):
    words = clean_tweets(tweet)
    words_dictionary = dict([word, True] for word in words)
    return words_dictionary


#Make a set of positive and negative tweets
pos_tweets_set = []
for tweet in pos_tweets:
    pos_tweets_set.append((bag_of_words(tweet), 'pos'))
shuffle(pos_tweets_set)

neg_tweets_set = []
for tweet in neg_tweets:
    neg_tweets_set.append((bag_of_words(tweet), 'neg'))
shuffle(neg_tweets_set)

#Create test and train sets
test_set = pos_tweets_set[:1000] + neg_tweets_set[:1000]
train_set = pos_tweets_set[1000:] + neg_tweets_set[1000:]

classifier = NaiveBayesClassifier.train(train_set)
accurancy = classify.accuracy(classifier, test_set)
print('Classifier accurancy: %.2f' % (accurancy * 100), '%')

#Connect to Twitter
api = tweepy.API(twitterAuth.auth)

print("Enter the name of Twitter account (without @):")
scr_name = input()

#Get tweets
stuff = api.user_timeline(screen_name = scr_name, count = 8, include_rts = True)

tweets_to_analyse = []

#Prepare the output
for status in stuff:
    json_str = json.dumps(status.user._json)
    string = json.loads(json_str)
    name = string['name']
    screen_name = string['screen_name']
    text = status.text
    tweets_to_analyse.append(("%s(@%s)" % (name, screen_name), text))
    #print('%s(%s): ' % (name, screen_name) ,status.text)

#Classify our tweets and print them
for author, tweet in tweets_to_analyse:
    custom_tweet = tweet
    tweet_author = author
    custom_tweet_set = bag_of_words(custom_tweet)
    prob_result = classifier.prob_classify(custom_tweet_set)
    print('%s : %s' % (tweet_author, custom_tweet))
    print('Neg prob: %f' % prob_result.prob("neg"))
    print('Pos prob: %f' % prob_result.prob("pos"))
    if abs(prob_result.prob("neg") - prob_result.prob("pos")) < 0.2:
        print('Result: %s' % "neu")
    else:
        print('Result: %s' % prob_result.max())
    print()