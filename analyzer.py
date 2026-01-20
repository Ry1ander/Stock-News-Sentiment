from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import torch
import gc  # Garbage Collector

# 1. Setup the Model
model_name = "ProsusAI/finbert"
print("Loading FinBERT model...")
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Use CPU or GPU
device = 0 if torch.cuda.is_available() else -1

# Create pipeline
sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

def analyze_sentiment(texts):
    """
    Analyzes a list of texts using FinBERT.
    OPTIMIZED FOR LOW RAM / LAPTOPS.
    """
    print(f"Starting analysis on {len(texts)} articles...")
    
    results = []
    total = len(texts)
    
    # SAFETY MODE: Process 1 article at a time
    batch_size = 1 
    
    #'torch.no_grad()' prevents memory explosions
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch = texts[i : i+batch_size]
            
            # Simple progress log (e.g. "1/40", "2/40")
            if i % 5 == 0:
                print(f"Analyzing article {i+1}/{total}...")
            
            try:
                # Run AI
                batch_results = sentiment_pipeline(batch, truncation=True, padding=True, max_length=512)
                results.extend(batch_results)
            except Exception as e:
                print(f"Error on article {i}: {e}")
                # Fallback result so app doesn't crash
                results.append({'label': 'neutral', 'score': 0.0})

            # Force clear memory after every step
            gc.collect()
            
    return results

# -- TEST ---
if __name__ == "__main__":
    # Fake headlines to test
    test_headlines = [
        "Apple profits soar to record highs",  # Should be Positive
        "Factory fires delay production for months", # Should be Negative
        "Company announces annual meeting date" # Should be Neutral
    ]
    
    sentiments = analyze_sentiment(test_headlines)
    
    # Print results nicely
    for headline, result in zip(test_headlines, sentiments):
        print(f"Headline: {headline}")
        print(f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})")
        print("-" * 30)



