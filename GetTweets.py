import GetOldTweets3 as got
from flask import Flask, request, render_template
import SAEUtility as ut
import SAEngine as sae
import TweetCleaning as tc
import numpy as np


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def my_form_post():
    text = request.form['search']

    text_query = text  # Input search query to scrape tweets and name csv file
    count = 100 # pulls x amount of most recent tweets
    print("User has searched for: {}".format(str(text_query)))

    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(str(text_query))\
                                               .setMaxTweets(int(count))\
                                               .setTopTweets(1) #\
                                               #.setSince("2015-05-01")\
                                               #.setUntil("2015-09-30")

    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    tweets_df = ut.assembleDataframe(tweets)

    avgFavs = ut.getAvgScore(tweets_df['Favorites'])
    topThreeDF = ut.findTopThreeTweets(tweets_df)

    #generating cleaned tweet dataframe
    cleaned_tweets_df = tweets_df[['Datetime', 'Text']].copy()
    tc.process_text(cleaned_tweets_df)
    tc.tokenize_words((cleaned_tweets_df))


    #scoring cleaned tweets dataframe
    sae.analyze_tweets(cleaned_tweets_df)
    sae.score_by_naive_bayes(cleaned_tweets_df)
    # Copying newly added scores to other relevant dataframes
    tweets_df['Polarity'] = cleaned_tweets_df['Polarity']
    tweets_df['Subjectivity'] = cleaned_tweets_df['Subjectivity']
    tweets_df['NB-Positive Probability'] = cleaned_tweets_df['NB-Positive Probability']
    #tweets_df['NB-Neutral Probability'] = cleaned_tweets_df['NB-Neutral Probability']
    tweets_df['NB-Negative Probability'] = cleaned_tweets_df['NB-Negative Probability']
    tweets_df['NB-Sentiment Score'] = cleaned_tweets_df['NB-Sentiment Score']
    topThreeDF['Polarity'] = cleaned_tweets_df['Polarity']
    topThreeDF['Subjectivity'] = cleaned_tweets_df['Subjectivity']
    topThreeDF['NB-Positive Probability'] = cleaned_tweets_df['NB-Positive Probability']
    #topThreeDF['NB-Neutral Probability'] = cleaned_tweets_df['NB-Neutral Probability']
    topThreeDF['NB-Negative Probability'] = cleaned_tweets_df['NB-Negative Probability']
    topThreeDF['NB-Sentiment Score'] = cleaned_tweets_df['NB-Sentiment Score']
    avgScore = ut.getAvgScore(cleaned_tweets_df['NB-Sentiment Score']) * 100

    # Creating server-side directory to save report data
    ut.createDirectory(text_query)

    # Converting tweet dataframes to csv file for local storage
    tweets_df.to_csv('static/{}/{}-{}k-tweets.csv'.format(str(text_query), str(text_query), int(int(count)/1000)), sep=',')
    topThreeDF.to_csv('static/{}/{}-TopThree-tweets.csv'.format(str(text_query), str(text_query)), sep=',')
    cleaned_tweets_df.to_csv('static/{}/{}-cleaned-tweets.csv'.format(str(text_query), int(int(count) / 1000)), sep=',')

    # Saving data visualizations
    ut.generateHistogram(tweets_df, text_query)
    ut.generatePieGraph(cleaned_tweets_df['NB-Sentiment Score'], text_query)
    ut.sentiment_vs_time(text_query, avgScore)

    mostPos = ut.findPos(tweets_df)
    mostNeg = ut.findNeg(tweets_df)
    mostPos['NB-Sentiment Score'] = np.round(mostPos['NB-Sentiment Score'].astype(np.double), 4)*100
    mostNeg['NB-Sentiment Score'] = np.round(mostNeg['NB-Sentiment Score'].astype(np.double), 4)*100
    topThreeDF['NB-Sentiment Score'] = np.round(topThreeDF['NB-Sentiment Score'].astype(np.double), 4)*100

    print(tweets_df['Geo'])
    return render_template('landing.html', my_string=str(round(avgScore, 2)), topic_string=text, x=topThreeDF, pos=mostPos, neg=mostNeg)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/sentimentAnalysis")
def sentimentAnalysis():
    return render_template("sentimentAnalysis.html")

if __name__== "__main__":
    app.run()
