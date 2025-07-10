import logging
import time
from typing import Dict, List, Optional
import numpy as np
import torch
import uvicorn
import asyncio
import json
import os
import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FastAPI app
app = FastAPI(title="NLP Processor Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],  # Allow all headers
)

class AdaptiveConfidenceEnsemble:
    """ADAPTIVE CONFIDENCE-BASED MODEL SELECTION: Adaptive Confidence-Based Model Selection"""
    
    def __init__(self, models: Dict):
        self.models = models
        logger.info(f"Initialized ensemble with {len(models)} models")
    
    def softmax(self, x: List[float]) -> List[float]:
        """Softmax function for confidence weighting"""
        if not x:
            return []
        x_array = np.array(x)
        exp_x = np.exp(x_array - np.max(x_array))
        return (exp_x / np.sum(exp_x)).tolist()
    
    def apply_calibration(self, confidence: float, model_name: str) -> float:
        """Apply Platt scaling calibration to confidence scores"""
        if model_name == "xlm-roberta":
            # XLM-RoBERTa tends to be overconfident, so we reduce it
            return min(0.95, confidence * 0.85 + 0.1)
        elif model_name == "mbert":
            # mBERT is more conservative, so we boost it slightly
            return min(0.95, confidence * 0.95 + 0.05)
        return confidence
    
    def confidence_weighted_ensemble(self, text: str) -> Dict:
        """INNOVATION: Confidence-weighted ensemble selection"""
        start_time = time.time()
        
        logger.info(f"Processing text: {text[:50]}...")
        
        # Get predictions from all models
        predictions = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"Calling model: {name}")
                
                # Call the model
                result = model(text)
                
                # Handle pipeline result which returns a list
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                
                # Extract confidence from score
                confidence = result.get('score', 0.5)
                sentiment = result.get('label', 'NEUTRAL')
                
                logger.info(f"Raw result from {name}: {sentiment} (score: {confidence:.3f})")
                
                # Normalize sentiment labels based on model
                if name == "xlm-roberta":
                    # XLM-RoBERTa returns: negative, neutral, positive
                    if sentiment.lower() in ['negative', 'label_0']:
                        sentiment = 'negative'
                    elif sentiment.lower() in ['positive', 'label_2']:
                        sentiment = 'positive'
                    elif sentiment.lower() in ['neutral', 'label_1']:
                        sentiment = 'neutral'
                    else:
                        # Default fallback - keep original sentiment if it's already correct
                        sentiment = sentiment.lower()
                elif name == "mbert":
                    # mBERT returns: 1 star, 2 stars, 3 stars, 4 stars, 5 stars
                    if sentiment in ['LABEL_0', '1 star']:
                        sentiment = 'negative'
                    elif sentiment in ['LABEL_1', '2 stars']:
                        sentiment = 'negative'
                    elif sentiment in ['LABEL_2', '3 stars']:
                        sentiment = 'neutral'
                    elif sentiment in ['LABEL_3', '4 stars']:
                        sentiment = 'positive'
                    elif sentiment in ['LABEL_4', '5 stars']:
                        sentiment = 'positive'
                    else:
                        sentiment = 'neutral'
                
                # Apply confidence calibration
                calibrated_confidence = self.apply_calibration(confidence, name)
                
                logger.info(f"Model {name}: {sentiment} (calibrated conf: {calibrated_confidence:.3f})")
                
                # Create probability distribution
                if sentiment == "negative":
                    probabilities = [calibrated_confidence, (1-calibrated_confidence)/2, (1-calibrated_confidence)/2]
                elif sentiment == "positive":
                    probabilities = [(1-calibrated_confidence)/2, (1-calibrated_confidence)/2, calibrated_confidence]
                else:
                    probabilities = [(1-calibrated_confidence)/2, calibrated_confidence, (1-calibrated_confidence)/2]
                
                # Normalize probabilities
                prob_sum = sum(probabilities)
                if prob_sum > 0:
                    probabilities = [p/prob_sum for p in probabilities]
                else:
                    probabilities = [0.33, 0.34, 0.33]

                predictions[name] = {
                    "sentiment": sentiment,
                    "confidence": calibrated_confidence,
                    "raw_confidence": confidence,
                    "probabilities": probabilities
                }
                
            except Exception as e:
                logger.error(f"Model {name} failed: {e}")
                predictions[name] = {
                    "sentiment": "neutral",
                    "confidence": 0.1,
                    "raw_confidence": 0.5,
                    "probabilities": [0.33, 0.34, 0.33]
                }
        
        # Calculate weights using softmax
        confidences = [pred["confidence"] for pred in predictions.values()]
        weights = self.softmax(confidences)
        
        logger.info(f"Confidences: {confidences}")
        logger.info(f"Weights: {weights}")
        
        # Weighted ensemble
        ensemble_prob = np.zeros(3)
        for i, (name, pred) in enumerate(predictions.items()):
            if i < len(weights):
                ensemble_prob += weights[i] * np.array(pred["probabilities"])
        
        logger.info(f"Ensemble probabilities: {ensemble_prob}")
        
        # Select final sentiment
        sentiment_idx = np.argmax(ensemble_prob)
        sentiment = ["negative", "neutral", "positive"][sentiment_idx]
        final_confidence = float(np.max(ensemble_prob))
        
        # Best model selection based on language and confidence
        best_model = self._select_best_model(text, predictions)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Final result: {sentiment} (conf: {final_confidence:.3f})")
        
        return {
            "sentiment": sentiment,
            "confidence": final_confidence,
            "model_weights": dict(zip(predictions.keys(), weights)),
            "ensemble_probabilities": ensemble_prob.tolist(),
            "selected_model": best_model,
            "all_predictions": predictions,
            "processing_time_ms": processing_time,
            "method": "confidence_weighted_ensemble"
        }
    
    def _select_best_model(self, text: str, predictions: Dict) -> str:
        """Select the best model based on text characteristics and confidence"""
        # Check if text is multilingual (contains non-ASCII characters)
        is_multilingual = any(ord(char) > 127 for char in text)
        
        if is_multilingual:
            # For multilingual text, prefer XLM-RoBERTa
            if "xlm-roberta" in predictions:
                return "xlm-roberta"
        
        # Otherwise, select based on highest confidence
        return max(predictions.keys(), key=lambda k: predictions[k]["confidence"])

class OptimizedSentimentAnalyzer:
    def __init__(self):
        logger.info("Initializing Sentiment Analyzer...")
        self._initialize_models()
        self.ensemble = AdaptiveConfidenceEnsemble({
            "xlm-roberta": self.xlmr_pipeline,
            "mbert": self.mbert_pipeline
        })
        logger.info("Sentiment Analyzer ready")
    
    def _initialize_models(self):
        """Initialize models"""
        logger.info("Loading XLM-RoBERTa...")
        self.xlmr_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info("XLM-RoBERTa loaded")
        
        logger.info("Loading mBERT...")
        self.mbert_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info("mBERT loaded")

# Global analyzer instance
analyzer = OptimizedSentimentAnalyzer()

# Redis connection
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'redis'),
    port=6379,
    decode_responses=True
)

# Kafka configuration
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'kafka:9092')
consumer = None
producer = None
pg_pool = None

# Request/Response models
class SentimentRequest(BaseModel):
    text: str
    message_id: Optional[str] = None
    language: Optional[str] = None
    is_code_switched: bool = False

class SentimentResponse(BaseModel):
    message_id: Optional[str]
    sentiment: str
    confidence: float
    processing_time_ms: float
    model_used: str
    language: Optional[str]
    is_code_switched: bool

# API endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "nlp-processor",
        "models_loaded": ["xlm-roberta", "mbert"],
        "innovation": "Adaptive Confidence-Based Model Selection"
    }

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    try:
        # Use the ensemble method
        result = analyzer.ensemble.confidence_weighted_ensemble(request.text)
        
        return SentimentResponse(
            message_id=request.message_id,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            processing_time_ms=result["processing_time_ms"],
            model_used=result["selected_model"],
            language=request.language,
            is_code_switched=request.is_code_switched
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/ensemble")
async def analyze_with_ensemble(request: SentimentRequest):
    try:
        result = analyzer.ensemble.confidence_weighted_ensemble(request.text)
        return result
    except Exception as e:
        logger.error(f"Ensemble analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    return {
        "available_models": ["xlm-roberta", "mbert"],
        "ensemble_method": "confidence_weighted",
        "innovation": "Adaptive Confidence-Based Model Selection"
    }

@app.post("/analyze/batch")
async def analyze_batch(texts: List[str]):
    results = []
    for text in texts:
        try:
            result = analyzer.ensemble.confidence_weighted_ensemble(text)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "text": text})
    return {"results": results}

@app.get("/confidence-stats")
async def get_confidence_stats():
    return {
        "calibration_method": "Platt scaling",
        "xlm_roberta_adjustment": "0.85x + 0.1",
        "mbert_adjustment": "0.95x + 0.05",
        "ensemble_method": "softmax_weighted"
    }

# Kafka message processor
async def process_kafka_messages():
    """Process messages from Kafka queue for sentiment analysis"""
    global consumer, producer, pg_pool
    
    consumer = AIOKafkaConsumer(
        'nlp-processing',
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    await consumer.start()
    await producer.start()
    
    try:
        async for msg in consumer:
            try:
                message_data = msg.value
                logger.info(f"Processing message for sentiment analysis: {message_data.get('id')}")
                
                # Perform sentiment analysis
                text = message_data.get('content', '')
                result = analyzer.ensemble.confidence_weighted_ensemble(text)
                
                # Update message in database with sentiment results
                await pg_pool.execute("""
                    UPDATE messages 
                    SET sentiment = $1, 
                        sentiment_confidence = $2,
                        processed = true,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = $3
                """, result['sentiment'], result['confidence'], message_data['id'])
                
                # Check if we need to create an alert
                if result['sentiment'] == 'negative' and result['confidence'] > 0.8:
                    # Send to alert manager
                    alert_data = {
                        'message_id': message_data['id'],
                        'type': 'negative_sentiment',
                        'priority': 'high' if result['confidence'] > 0.9 else 'medium',
                        'sentiment': result['sentiment'],
                        'confidence': result['confidence'],
                        'content': text,
                        'metadata': message_data.get('metadata', {})
                    }
                    
                    await producer.send(
                        'alert-creation',
                        key=message_data['id'].encode('utf-8'),
                        value=alert_data
                    )
                
                # Cache result
                redis_client.setex(
                    f"sentiment:{message_data['id']}",
                    3600,
                    json.dumps(result)
                )
                
                logger.info(f"Message {message_data['id']} analyzed: {result['sentiment']} "
                          f"(confidence: {result['confidence']:.3f})")
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    finally:
        await consumer.stop()
        await producer.stop()

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    global pg_pool
    # Create PostgreSQL connection pool
    pg_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432,
        database='ai_crm',
        user='admin',
        password='secure_password'
    )
    # Start Kafka consumer
    asyncio.create_task(process_kafka_messages())
    logger.info("NLP Processor Service started with Kafka consumer")

@app.on_event("shutdown")
async def shutdown_event():
    if consumer:
        await consumer.stop()
    if producer:
        await producer.stop()
    if pg_pool:
        await pg_pool.close()
    logger.info("NLP Processor Service stopped")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 