import tweepy

api_key = 'api_key'
api_secret_key = 'api_secret_key'
acc_token = 'acc_token'
acc_token_secret = 'acc_token_secret'

auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(acc_token, acc_token_secret)