from sqlalchemy import Column, Integer, String, Float, Boolean
from database.connection import Base

class Tweet(Base):
    __tablename__ = "tweets"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, index=True)
    sentiment_score = Column(Float)
    is_depression = Column(Boolean)

    