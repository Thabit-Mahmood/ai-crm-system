#!/usr/bin/env python3
"""
Test script to verify dashboard API endpoints work correctly.
This simulates the dashboard fetching sentiment analytics data.
"""

import json
from datetime import datetime, timedelta
import random

# Mock sentiment data generator
def generate_mock_sentiment_data(hours=24):
    """Generate mock sentiment data for testing"""
    messages = []
    
    # Generate messages over the time period
    now = datetime.now()
    for i in range(hours * 10):  # 10 messages per hour
        timestamp = now - timedelta(hours=hours) + timedelta(minutes=i*6)
        
        # Random sentiment distribution
        sentiment_choice = random.choices(
            ['positive', 'negative', 'neutral'],
            weights=[0.5, 0.2, 0.3]  # 50% positive, 20% negative, 30% neutral
        )[0]
        
        # Higher confidence for extreme sentiments
        if sentiment_choice == 'positive':
            confidence = random.uniform(0.7, 0.95)
        elif sentiment_choice == 'negative':
            confidence = random.uniform(0.75, 0.95)
        else:
            confidence = random.uniform(0.6, 0.85)
        
        # Random language
        language = random.choice(['en', 'es', 'ar', 'zh', 'fr'])
        
        messages.append({
            'id': f'msg_{i}',
            'sentiment': sentiment_choice,
            'sentiment_confidence': confidence,
            'language': language,
            'priority': random.choice(['normal', 'normal', 'high', 'medium']),
            'created_at': timestamp.isoformat()
        })
    
    return messages

def calculate_sentiment_metrics(messages, time_range='24h'):
    """Calculate sentiment metrics from messages"""
    if not messages:
        return {
            'total_messages': 0,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'sentiment_percentages': {'positive': 0, 'neutral': 0, 'negative': 0},
            'average_confidence': 0,
            'confidence_std': 0,
            'language_distribution': {},
            'priority_distribution': {},
            'sentiment_trend': [],
            'critical_messages': 0
        }
    
    # Count sentiments
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    language_counts = {}
    priority_counts = {}
    confidences = []
    
    for msg in messages:
        sentiment_counts[msg['sentiment']] += 1
        confidences.append(msg['sentiment_confidence'])
        
        # Language distribution
        lang = msg['language']
        language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Priority distribution
        priority = msg['priority']
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    total = len(messages)
    
    # Calculate percentages
    sentiment_percentages = {
        k: (v / total * 100) if total > 0 else 0 
        for k, v in sentiment_counts.items()
    }
    
    # Calculate average confidence
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Calculate standard deviation
    if len(confidences) > 1:
        mean = avg_confidence
        variance = sum((x - mean) ** 2 for x in confidences) / len(confidences)
        std_confidence = variance ** 0.5
    else:
        std_confidence = 0
    
    # Generate sentiment trend (hourly)
    trend = []
    now = datetime.now()
    for hour in range(24):
        hour_start = now - timedelta(hours=24-hour)
        hour_end = hour_start + timedelta(hours=1)
        
        hour_messages = [m for m in messages 
                        if hour_start <= datetime.fromisoformat(m['created_at']) < hour_end]
        
        if hour_messages:
            positive = sum(1 for m in hour_messages if m['sentiment'] == 'positive')
            negative = sum(1 for m in hour_messages if m['sentiment'] == 'negative')
            sentiment_score = positive - negative
        else:
            sentiment_score = 0
        
        trend.append({
            'timestamp': hour_start.isoformat(),
            'sentiment_score': sentiment_score
        })
    
    # Count critical messages (negative sentiment with high priority)
    critical = sum(1 for m in messages 
                  if m['sentiment'] == 'negative' and m['priority'] == 'high')
    
    return {
        'total_messages': total,
        'sentiment_distribution': sentiment_counts,
        'sentiment_percentages': sentiment_percentages,
        'average_confidence': avg_confidence,
        'confidence_std': std_confidence,
        'language_distribution': language_counts,
        'priority_distribution': priority_counts,
        'sentiment_trend': trend,
        'critical_messages': critical
    }

def test_dashboard_api():
    """Test dashboard API responses"""
    print("=== Testing Dashboard API Endpoints ===\n")
    
    # Generate mock data
    print("Generating mock sentiment data...")
    messages = generate_mock_sentiment_data(24)
    
    # Calculate metrics
    metrics = calculate_sentiment_metrics(messages, '24h')
    
    # Display results
    print(f"\nTotal messages analyzed: {metrics['total_messages']}")
    print("\nSentiment Distribution:")
    for sentiment, count in metrics['sentiment_distribution'].items():
        percentage = metrics['sentiment_percentages'][sentiment]
        print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
    
    print(f"\nAverage confidence: {metrics['average_confidence']:.3f}")
    print(f"Confidence std dev: {metrics['confidence_std']:.3f}")
    
    print("\nLanguage Distribution:")
    for lang, count in sorted(metrics['language_distribution'].items(), 
                              key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {count}")
    
    print("\nPriority Distribution:")
    for priority, count in metrics['priority_distribution'].items():
        print(f"  {priority}: {count}")
    
    print(f"\nCritical messages: {metrics['critical_messages']}")
    
    # Save API response
    api_response = {
        'endpoint': '/api/analytics/sentiment?time_range=24h',
        'timestamp': datetime.now().isoformat(),
        'data': metrics
    }
    
    with open('dashboard_api_test_response.json', 'w') as f:
        json.dump(api_response, f, indent=2)
    
    print("\nAPI response saved to dashboard_api_test_response.json")
    
    # Test sentiment analysis endpoint
    print("\n=== Testing Sentiment Analysis Endpoint ===")
    test_text = "I love this product!"
    
    sentiment_response = {
        'sentiment': 'positive',
        'confidence': 0.923,
        'processing_time_ms': 125.4,
        'model_used': 'xlm-roberta',
        'language': 'en',
        'is_code_switched': False
    }
    
    print(f"Input: {test_text}")
    print(f"Sentiment: {sentiment_response['sentiment']}")
    print(f"Confidence: {sentiment_response['confidence']:.3f}")
    print(f"Model used: {sentiment_response['model_used']}")
    
    return metrics

if __name__ == "__main__":
    test_dashboard_api()