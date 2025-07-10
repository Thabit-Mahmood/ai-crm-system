#!/usr/bin/env python3
"""
Test script to verify sentiment analysis is working correctly
"""

import requests
import json

def test_sentiment_analysis():
    """Test sentiment analysis with various inputs"""
    
    # Test cases
    test_cases = [
        {"text": "I love this product! It's amazing and fantastic!", "expected": "positive"},
        {"text": "This is terrible! I hate it! Worst product ever!", "expected": "negative"},
        {"text": "The product is okay, nothing special.", "expected": "neutral"},
        {"text": "Me encanta este producto increíble!", "expected": "positive"},
        {"text": "Odio este producto terrible!", "expected": "negative"},
    ]
    
    print("🧪 TESTING SENTIMENT ANALYSIS FIX")
    print("="*50)
    
    correct = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            response = requests.post(
                "http://localhost:3003/analyze",
                json={"text": test_case["text"]},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted = result.get("sentiment", "unknown")
                expected = test_case["expected"]
                confidence = result.get("confidence", 0)
                model_used = result.get("model_used", "unknown")
                
                is_correct = predicted == expected
                if is_correct:
                    correct += 1
                    status = "✅ CORRECT"
                else:
                    status = "❌ INCORRECT"
                
                print(f"Test {i}: {status}")
                print(f"  Text: {test_case['text'][:40]}...")
                print(f"  Expected: {expected}")
                print(f"  Predicted: {predicted}")
                print(f"  Confidence: {confidence:.3f}")
                print(f"  Model: {model_used}")
                print()
                
            else:
                print(f"Test {i}: ❌ HTTP ERROR {response.status_code}")
                print(f"  Text: {test_case['text'][:40]}...")
                print()
                
        except Exception as e:
            print(f"Test {i}: ❌ ERROR - {str(e)}")
            print(f"  Text: {test_case['text'][:40]}...")
            print()
    
    accuracy = correct / total
    print("="*50)
    print(f"RESULTS: {correct}/{total} correct ({accuracy:.1%})")
    
    if accuracy >= 0.8:
        print("🎉 SENTIMENT ANALYSIS FIX SUCCESSFUL!")
        print("✅ Ready for production use")
        return True
    else:
        print("⚠️  SENTIMENT ANALYSIS NEEDS MORE WORK")
        print("❌ Additional fixes required")
        return False

if __name__ == "__main__":
    success = test_sentiment_analysis()
    exit(0 if success else 1) 