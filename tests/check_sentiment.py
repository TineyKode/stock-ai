"""
Test sentiment analysis pipeline.
Requires: pip install transformers torch
"""
from transformers import pipeline

# Use the same model as fetch_news.py for consistency
analyzer = pipeline(
    "sentiment-analysis",
    model="finiteautomata/bertweet-base-sentiment-analysis"
)

test_headlines = [
    "Apple stock hits all-time high",
    "Market crashes amid economic uncertainty",
    "Tech stocks show steady growth"
]

for headline in test_headlines:
    result = analyzer(headline)
    print(f"'{headline}' -> {result[0]['label']} ({result[0]['score']:.4f})")
