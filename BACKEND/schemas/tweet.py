from pydantic import BaseModel
from typing import List

class TweetBase(BaseModel):
    text: str

class TweetCreate(TweetBase):
    pass

class TweetResponse(TweetBase):
    id: int
    sentiment_score: float
    is_depression: bool

    class Config:
        from_attributes = True

class TweetHistoryResponse(BaseModel):
    history: List[TweetResponse]