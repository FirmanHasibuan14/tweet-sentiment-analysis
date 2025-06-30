from fastapi import FastAPI

app = FastAPI()

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Tweet Sentiment Prediction API. Go to /docs for documentation."}