from fastapi import FastAPI
from database.connection import create_tables
from core.config import settings
from routers import sentiment

create_tables()

app = FastAPI(
    title=settings.APP_NAME
)

app.include_router(sentiment.router, prefix='/sentiment', tags=['Sentiment Analysis'])

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Tweet Sentiment Prediction API. Go to /docs for documentation."}