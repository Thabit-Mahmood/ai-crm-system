from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import json
import logging
import uvicorn
from datetime import datetime
import redis
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import langdetect
from langdetect import detect_langs, LangDetectException
import fasttext
import numpy as np
from transformers import pipeline
import torch
import re
import os
from collections import Counter

# Initialize FastAPI app
app = FastAPI(title="Language Detection Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Redis connection
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=6379,
    decode_responses=True
)

# Kafka configuration
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
consumer = None
producer = None

# Load language detection models
class LanguageDetector:
    def __init__(self):
        logger.info("Initializing language detection models...")
        # Download FastText model if not exists
        self.fasttext_model_path = "/tmp/lid.176.bin"
        if not os.path.exists(self.fasttext_model_path):
            logger.info("Downloading FastText language identification model...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
                self.fasttext_model_path
            )
        
        self.fasttext_model = fasttext.load_model(self.fasttext_model_path)
        self.code_switch_patterns = self._compile_code_switch_patterns()
        logger.info("Language detection models initialized successfully")
    
    def _compile_code_switch_patterns(self):
        """Compile regex patterns for common code-switching scenarios"""
        return {
            'script_change': re.compile(r'[\u0600-\u06FF]+.*[\u0041-\u024F]+|[\u0041-\u024F]+.*[\u0600-\u06FF]+'),  # Arabic-Latin
            'cjk_latin': re.compile(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]+.*[\u0041-\u024F]+|[\u0041-\u024F]+.*[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]+'),  # CJK-Latin
            'devanagari_latin': re.compile(r'[\u0900-\u097F]+.*[\u0041-\u024F]+|[\u0041-\u024F]+.*[\u0900-\u097F]+'),  # Devanagari-Latin
        }
    
    def detect_language(self, text: str) -> Dict:
        """Detect language(s) in text with code-switching support"""
        results = {
            'primary_language': None,
            'confidence': 0.0,
            'all_languages': [],
            'is_code_switched': False,
            'code_switch_points': []
        }
        
        if not text or len(text.strip()) < 3:
            results['primary_language'] = 'unknown'
            return results
        
        # Check for code-switching patterns
        for pattern_name, pattern in self.code_switch_patterns.items():
            if pattern.search(text):
                results['is_code_switched'] = True
                results['code_switch_points'].append(pattern_name)
        
        # Use multiple detection methods
        # Method 1: langdetect
        try:
            lang_detect_results = detect_langs(text)
            for lang in lang_detect_results:
                results['all_languages'].append({
                    'language': lang.lang,
                    'confidence': lang.prob,
                    'method': 'langdetect'
                })
        except LangDetectException:
            logger.warning(f"langdetect failed for text: {text[:50]}...")
        
        # Method 2: FastText
        try:
            predictions = self.fasttext_model.predict(text, k=3)
            for i, (label, confidence) in enumerate(zip(predictions[0], predictions[1])):
                lang_code = label.replace('__label__', '')
                results['all_languages'].append({
                    'language': lang_code,
                    'confidence': float(confidence),
                    'method': 'fasttext'
                })
        except Exception as e:
            logger.error(f"FastText error: {e}")
        
        # If code-switching detected, analyze segments
        if results['is_code_switched']:
            segments = self._segment_by_script(text)
            for segment in segments:
                if len(segment['text'].strip()) > 2:
                    try:
                        seg_lang = detect_langs(segment['text'])[0]
                        segment['language'] = seg_lang.lang
                        segment['confidence'] = seg_lang.prob
                    except:
                        segment['language'] = 'unknown'
                        segment['confidence'] = 0.0
            results['segments'] = segments
        
        # Determine primary language
        if results['all_languages']:
            # Sort by confidence
            sorted_langs = sorted(results['all_languages'], key=lambda x: x['confidence'], reverse=True)
            results['primary_language'] = sorted_langs[0]['language']
            results['confidence'] = sorted_langs[0]['confidence']
        else:
            results['primary_language'] = 'unknown'
        
        return results
    
    def _segment_by_script(self, text: str) -> List[Dict]:
        """Segment text by script changes for code-switching analysis"""
        segments = []
        current_segment = []
        current_script = None
        
        for char in text:
            script = self._get_script(char)
            if script != current_script and current_script is not None:
                if current_segment:
                    segments.append({
                        'text': ''.join(current_segment),
                        'script': current_script,
                        'start': len(''.join([s['text'] for s in segments])),
                        'end': len(''.join([s['text'] for s in segments])) + len(current_segment)
                    })
                current_segment = [char]
                current_script = script
            else:
                current_segment.append(char)
                if current_script is None:
                    current_script = script
        
        # Add the last segment
        if current_segment:
            segments.append({
                'text': ''.join(current_segment),
                'script': current_script,
                'start': len(''.join([s['text'] for s in segments])),
                'end': len(text)
            })
        
        return segments
    
    def _get_script(self, char: str) -> str:
        """Determine the script of a character"""
        code = ord(char)
        if 0x0041 <= code <= 0x024F:
            return 'latin'
        elif 0x0600 <= code <= 0x06FF:
            return 'arabic'
        elif 0x4E00 <= code <= 0x9FFF:
            return 'chinese'
        elif 0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF:
            return 'japanese'
        elif 0x0900 <= code <= 0x097F:
            return 'devanagari'
        elif 0x0B80 <= code <= 0x0BFF:
            return 'tamil'
        else:
            return 'other'

# Initialize language detector
detector = LanguageDetector()

# Data models
class LanguageDetectionRequest(BaseModel):
    text: str
    message_id: Optional[str] = None

class LanguageDetectionResponse(BaseModel):
    message_id: Optional[str]
    primary_language: str
    confidence: float
    is_code_switched: bool
    all_languages: List[Dict]
    segments: Optional[List[Dict]] = None
    processing_time_ms: float

# API endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "language-detector"}

@app.post("/detect", response_model=LanguageDetectionResponse)
async def detect_language(request: LanguageDetectionRequest):
    """Detect language(s) in the provided text"""
    start_time = datetime.now()
    
    try:
        # Check cache
        if request.message_id:
            cached_result = redis_client.get(f"lang:{request.message_id}")
            if cached_result:
                return json.loads(cached_result)
        
        # Perform detection
        result = detector.detect_language(request.text)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = LanguageDetectionResponse(
            message_id=request.message_id,
            primary_language=result['primary_language'],
            confidence=result['confidence'],
            is_code_switched=result['is_code_switched'],
            all_languages=result['all_languages'],
            segments=result.get('segments'),
            processing_time_ms=processing_time
        )
        
        # Cache result
        if request.message_id:
            redis_client.setex(
                f"lang:{request.message_id}",
                3600,
                json.dumps(response.dict())
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Kafka message processor
async def process_kafka_messages():
    """Process messages from Kafka queue"""
    global consumer, producer
    
    consumer = AIOKafkaConsumer(
        'language-detection',
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
                logger.info(f"Processing message: {message_data['id']}")
                
                # Detect language
                result = detector.detect_language(message_data['content'])
                
                # Add language information to message
                message_data['language'] = result['primary_language']
                message_data['language_confidence'] = result['confidence']
                message_data['is_code_switched'] = result['is_code_switched']
                message_data['language_details'] = result
                
                # Send to NLP processor
                await producer.send(
                    'nlp-processing',
                    key=message_data['id'].encode('utf-8'),
                    value=message_data
                )
                
                logger.info(f"Message {message_data['id']} detected as {result['primary_language']} "
                          f"(confidence: {result['confidence']:.2f}, code-switched: {result['is_code_switched']})")
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    finally:
        await consumer.stop()
        await producer.stop()

# Start Kafka consumer on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_kafka_messages())
    logger.info("Language Detection Service started")

@app.on_event("shutdown")
async def shutdown_event():
    if consumer:
        await consumer.stop()
    if producer:
        await producer.stop()
    logger.info("Language Detection Service stopped")

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3002)
