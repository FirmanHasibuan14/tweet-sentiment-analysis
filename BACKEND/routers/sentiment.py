from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from schemas.tweet import TweetCreate, TweetResponse
from database.connection import get_db
from controllers import sentiment
from services.ml_service import sentiment_service, SentimentAnalysisService
from typing import List

router = APIRouter()

def get_ml_service():
    return sentiment_service

@router.post("/predict", response_model=TweetResponse)
def predict_sentiment_api(tweet: TweetCreate, db: Session=Depends(get_db), ml_service: SentimentAnalysisService = Depends(get_ml_service)):
    return sentiment.analyze_and_save_sentiment(tweet, db, ml_service)

@router.get("/history", response_model=List[TweetResponse])
def get_history_api(db: Session = Depends(get_db)):
    history = sentiment.get_prediction_history(db)
    return history