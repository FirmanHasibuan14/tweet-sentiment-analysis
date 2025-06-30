from sqlalchemy.orm import Session
from schemas.tweet import TweetCreate
from models.tweet import Tweet
from services.ml_service import SentimentAnalysisService

def analyze_and_save_sentiment(tweet_data: TweetCreate, db: Session, ml_service: SentimentAnalysisService):
    prediction_result = SentimentAnalysisService.predict(tweet_data.text)
    db_tweet = Tweet(
        text=tweet_data.text,
        sentiment_score=prediction_result["sentiment_score"],
        is_depression=prediction_result["is_depression"]
    )

    db.add(db_tweet)
    db.commit()
    db.refresh(db_tweet)

    return db_tweet

def get_prediction_history(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Tweet).offset(skip).limit(limit).all()