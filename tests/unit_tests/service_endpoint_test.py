#!/usr/bin/env python3
"""
Direct Service Test - Test Individual Services
==============================================
"""

import requests
import json

def test_nlp_processor():
    """Test NLP processor sentiment analysis"""
    try:
        response = requests.post(
            "http://localhost:3003/analyze",
            json={"text": "I love this product! It's amazing."},
            timeout=10
        )
        print(f"NLP Processor Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Result: {result}")
            return True
    except Exception as e:
        print(f"NLP Processor Error: {e}")
    return False

def test_analytics_engine():
    """Test analytics engine trajectory mining"""
    try:
        # Test pattern mining
        test_data = {
            "conversations": [
                {
                    "conversation_id": "test_001",
                    "messages": [
                        {"sentiment": "neutral"},
                        {"sentiment": "negative"},
                        {"sentiment": "negative"}
                    ],
                    "escalated": True
                }
            ]
        }
        
        response = requests.post(
            "http://localhost:3005/mine-patterns",
            json=test_data,
            timeout=10
        )
        print(f"Analytics Engine (Pattern Mining) Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Pattern Mining Result: {result}")
            
            # Test escalation prediction
            pred_response = requests.post(
                "http://localhost:3005/predict-escalation",
                json={"conversation": test_data["conversations"][0]},
                timeout=10
            )
            print(f"Analytics Engine (Prediction) Status: {pred_response.status_code}")
            if pred_response.status_code == 200:
                pred_result = pred_response.json()
                print(f"Escalation Prediction Result: {pred_result}")
                return True
                
    except Exception as e:
        print(f"Analytics Engine Error: {e}")
    return False

def test_language_detector():
    """Test language detector"""
    try:
        response = requests.post(
            "http://localhost:3002/detect",
            json={"text": "Hello world"},
            timeout=10
        )
        print(f"Language Detector Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Language Detection Result: {result}")
            return True
    except Exception as e:
        print(f"Language Detector Error: {e}")
    return False

def main():
    print("=" * 60)
    print("DIRECT SERVICE TESTING")
    print("=" * 60)
    
    results = {}
    
    print("\n1. Testing NLP Processor (Adaptive Confidence-Based Model Selection)...")
    results['nlp_processor'] = test_nlp_processor()
    
    print("\n2. Testing Analytics Engine (Sentiment Trajectory Pattern Mining)...")
    results['analytics_engine'] = test_analytics_engine()
    
    print("\n3. Testing Language Detector...")
    results['language_detector'] = test_language_detector()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    working_count = sum(results.values())
    total_count = len(results)
    
    for service, status in results.items():
        print(f"{service}: {'✓ WORKING' if status else '✗ FAILED'}")
    
    print(f"\nServices Working: {working_count}/{total_count}")
    
    if working_count >= 2:
        print("✓ SUFFICIENT SERVICES FOR DEMONSTRATION")
    else:
        print("✗ INSUFFICIENT SERVICES")

if __name__ == "__main__":
    main() 