import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd

class SentimentAnalysisService:
    def __init__(self, model_path='/ML/Model/sentiment_model.h5', data_path='/ML/Data/sentiment_tweets3.csv'):
        self.model = tf.keras.models.load_model(model_path)
        
        self._download_nltk_data()
            
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        df = pd.read_csv(data_path)
        df.rename(columns={'message to examine': 'Text', 'label (depression result)': 'Label'}, inplace=True)
        df['Processed_Text'] = df['Text'].apply(self._full_preprocess)
        
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<unk>')
        self.tokenizer.fit_on_texts(df['Processed_Text'])

    def _download_nltk_data(self):
        try:
            stopwords.words('english')
        except LookupError:
            print("Mengunduh data NLTK...")
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
    
    def _clean_text(self, text):
        text = re.sub(r"http\\S+|www\\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^\\w\\s]", "", text)
        text = text.lower()
        return emoji.demojize(text)
    
    def _remove_stopwords_lemmatize(self, text):
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word, pos='v') for word in words if word not in self.stop_words]
        return ' '.join(lemmatized_words)

    def _full_preprocess(self, text: str):
        cleaned_text = self._clean_text(text)
        final_text = self._remove_stopwords_lemmatize(cleaned_text)
        return final_text

    def predict(self, text: str):
        processed_text = self._full_preprocess(text)

        sequences = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=83, padding='post')

        prediction = self.model.predict(padded_sequences)
        sentiment_score = prediction[0][0]
        is_depression = sentiment_score > 0.5

        return {
            "sentiment_score": float(sentiment_score),
            "is_depression": bool(is_depression)
        }

sentiment_service = SentimentAnalysisService()
