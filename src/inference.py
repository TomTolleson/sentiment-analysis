import os
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        self.model_id = f"{os.getenv('HUGGINGFACE_USERNAME')}/sentiment-analysis"
        self.classifier = pipeline("sentiment-analysis", model=self.model_id)

    def predict(self, text):
        result = self.classifier(text)[0]
        return {
            "Positive": result["score"] if result["label"] == "POSITIVE" else 0,
            "Neutral": result["score"] if result["label"] == "NEUTRAL" else 0,
            "Negative": result["score"] if result["label"] == "NEGATIVE" else 0
        }
