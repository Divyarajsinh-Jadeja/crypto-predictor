import feedparser
from textblob import TextBlob
import re

def clean_text(text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

def get_crypto_sentiment(coin_name):
    """
    Fetches news from Google News RSS for the given coin and calculates average sentiment.
    Returns a score between -1 (Negative) and 1 (Positive).
    """
    # Google News RSS URL
    rss_url = f"https://news.google.com/rss/search?q={coin_name}+crypto&hl=en-IN&gl=IN&ceid=IN:en"
    
    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries:
            return 0.0

        total_sentiment = 0
        count = 0

        # Analyze top 10 news items
        for entry in feed.entries[:10]:
            title = entry.title
            summary = entry.summary if 'summary' in entry else ""
            full_text = clean_text(title + " " + summary)
            
            analysis = TextBlob(full_text)
            total_sentiment += analysis.sentiment.polarity
            count += 1

        if count == 0:
            return 0.0
            
        avg_sentiment = total_sentiment / count
        return avg_sentiment

    except Exception as e:
        print(f"⚠️ Sentiment analysis failed for {coin_name}: {e}")
        return 0.0

def adjust_confidence_with_sentiment(base_confidence, sentiment_score):
    """
    Adjusts the confidence score based on sentiment.
    Sentiment > 0.1 boosts confidence.
    Sentiment < -0.1 reduces confidence (or boosts sell confidence).
    """
    # Normalize sentiment to a small factor (e.g., +/- 10%)
    adjustment = sentiment_score * 10 
    return min(100, max(0, base_confidence + adjustment))
