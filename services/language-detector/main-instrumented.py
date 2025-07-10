from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
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
import time
import asyncpg

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

# Configure logging with microsecond precision
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
language_detection_duration = Histogram(
    'language_detection_duration_ms',
    'Time spent detecting language in milliseconds',
    ['method', 'status', 'is_code_switched'],
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
)

language_detected_counter = Counter(
    'languages_detected_total',
    'Total number of languages detected',
    ['language', 'is_code_switched', 'confidence_level']
)

code_switching_counter = Counter(
    'code_switching_detected_total',
    'Total number of code-switched messages',
    ['primary_language', 'secondary_language']
)

detection_errors_counter = Counter(
    'language_detection_errors_total',
    'Total number of language detection errors',
    ['error_type', 'method']
)

active_detections_gauge = Gauge(
    'active_language_detections',
    'Number of active language detections'
)

model_loading_time_gauge = Gauge(
    'model_loading_time_ms',
    'Time taken to load language models',
    ['model_name']
)

# Redis connection
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=6379,
    decode_responses=True
)

# PostgreSQL connection for metrics storage
pg_pool = None

# Kafka configuration
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
consumer = None
producer = None

# Load language detection models with timing
class LanguageDetector:
    def __init__(self):
        logger.info("Initializing language detection models...")
        
        # Model loading timing
        start_time = time.time()
        
        # Download FastText model if not exists
        self.fasttext_model_path = "/tmp/lid.176.bin"
        if not os.path.exists(self.fasttext_model_path):
            logger.info("Downloading FastText language identification model...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
                self.fasttext_model_path
            )
        
        fasttext_load_start = time.time()
        self.fasttext_model = fasttext.load_model(self.fasttext_model_path)
        fasttext_load_time = (time.time() - fasttext_load_start) * 1000
        model_loading_time_gauge.labels(model_name='fasttext').set(fasttext_load_time)
        
        self.code_switch_patterns = self._compile_code_switch_patterns()
        
        total_load_time = (time.time() - start_time) * 1000
        logger.info(f"Language detection models initialized in {total_load_time:.2f}ms")
    
    def _compile_code_switch_patterns(self):
        """Compile regex patterns for common code-switching scenarios"""
        return {
            'script_change': re.compile(r'[\u0600-\u06FF]+.*[\u0041-\u024F]+|[\u0041-\u024F]+.*[\u0600-\u06FF]+'),  # Arabic-Latin
            'cjk_latin': re.compile(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]+.*[\u0041-\u024F]+|[\u0041-\u024F]+.*[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]+'),  # CJK-Latin
            'devanagari_latin': re.compile(r'[\u0900-\u097F]+.*[\u0041-\u024F]+|[\u0041-\u024F]+.*[\u0900-\u097F]+'),  # Devanagari-Latin
            'cyrillic_latin': re.compile(r'[\u0400-\u04FF]+.*[\u0041-\u024F]+|[\u0041-\u024F]+.*[\u0400-\u04FF]+'),  # Cyrillic-Latin
        }
    
    def detect_language(self, text: str, message_id: str = None) -> Dict:
        """Detect language(s) in text with code-switching support and detailed timing"""
        active_detections_gauge.inc()
        detection_start = time.time()
        
        results = {
            'primary_language': None,
            'confidence': 0.0,
            'all_languages': [],
            'is_code_switched': False,
            'code_switch_points': [],
            'processing_stages': {}
        }
        
        try:
            # Stage 1: Input validation
            validation_start = time.time()
            if not text or len(text.strip()) < 3:
                results['primary_language'] = 'unknown'
                results['processing_stages']['validation'] = (time.time() - validation_start) * 1000
                return results
            results['processing_stages']['validation'] = (time.time() - validation_start) * 1000
            
            # Stage 2: Code-switching pattern detection
            pattern_start = time.time()
            for pattern_name, pattern in self.code_switch_patterns.items():
                if pattern.search(text):
                    results['is_code_switched'] = True
                    results['code_switch_points'].append(pattern_name)
            results['processing_stages']['pattern_detection'] = (time.time() - pattern_start) * 1000
            
            # Stage 3: langdetect method
            langdetect_start = time.time()
            try:
                lang_detect_results = detect_langs(text)
                for lang in lang_detect_results:
                    results['all_languages'].append({
                        'language': lang.lang,
                        'confidence': lang.prob,
                        'method': 'langdetect'
                    })
            except LangDetectException as e:
                logger.warning(f"langdetect failed for text: {text[:50]}... Error: {e}")
                detection_errors_counter.labels(error_type='langdetect_exception', method='langdetect').inc()
            results['processing_stages']['langdetect'] = (time.time() - langdetect_start) * 1000
            
            # Stage 4: FastText method
            fasttext_start = time.time()
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
                detection_errors_counter.labels(error_type='fasttext_exception', method='fasttext').inc()
            results['processing_stages']['fasttext'] = (time.time() - fasttext_start) * 1000
            
            # Stage 5: Segment analysis for code-switching
            if results['is_code_switched']:
                segment_start = time.time()
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
                results['processing_stages']['segmentation'] = (time.time() - segment_start) * 1000
            
            # Stage 6: Determine primary language
            consolidation_start = time.time()
            if results['all_languages']:
                # Sort by confidence
                sorted_langs = sorted(results['all_languages'], key=lambda x: x['confidence'], reverse=True)
                results['primary_language'] = sorted_langs[0]['language']
                results['confidence'] = sorted_langs[0]['confidence']
                
                # Record language detection metrics
                confidence_level = 'high' if results['confidence'] > 0.8 else 'medium' if results['confidence'] > 0.5 else 'low'
                language_detected_counter.labels(
                    language=results['primary_language'],
                    is_code_switched=str(results['is_code_switched']),
                    confidence_level=confidence_level
                ).inc()
                
                # Record code-switching metrics
                if results['is_code_switched'] and len(sorted_langs) > 1:
                    code_switching_counter.labels(
                        primary_language=sorted_langs[0]['language'],
                        secondary_language=sorted_langs[1]['language']
                    ).inc()
            else:
                results['primary_language'] = 'unknown'
            results['processing_stages']['consolidation'] = (time.time() - consolidation_start) * 1000
            
            # Calculate total processing time
            total_time = (time.time() - detection_start) * 1000
            results['processing_time_ms'] = total_time
            
            # Record timing metrics
            language_detection_duration.observe(
                total_time,
                method='combined',
                status='success',
                is_code_switched=str(results['is_code_switched'])
            )
            
            # Store detailed metrics if message_id provided
            if message_id and pg_pool:
                await self._store_detection_metrics(message_id, results)
            
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            detection_errors_counter.labels(error_type='general_exception', method='combined').inc()
            total_time = (time.time() - detection_start) * 1000
            language_detection_duration.observe(
                total_time,
                method='combined',
                status='error',
                is_code_switched='false'
            )
            results['error'] = str(e)
        finally:
            active_detections_gauge.dec()
        
        return results
    
    def _segment_by_script(self, text: str) -> List[Dict]:
        """Segment text by script changes for code-switching analysis"""
        segments = []
        current_segment = []
        current_script = None
        
        for i, char in enumerate(text):
            script = self._get_script(char)
            if script != current_script and current_script is not None:
                if current_segment:
                    segments.append({
                        'text': ''.join(current_segment),
                        'script': current_script,
                        'start': len(''.join([s['text'] for s in segments])),
                        'end': len(''.join([s['text'] for s in segments])) + len(current_segment),
                        'char_position': i - len(current_segment)
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
                'end': len(text),
                'char_position': len(text) - len(current_segment)
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
        elif 0x0400 <= code <= 0x04FF:
            return 'cyrillic'
        elif 0xAC00 <= code <= 0xD7AF:
            return 'korean'
        else:
            return 'other'
    
    async def _store_detection_metrics(self, message_id: str, results: Dict):
        """Store detailed detection metrics in database"""
        try:
            async with pg_pool.acquire() as conn:
                await conn.execute('''
                    UPDATE processing_metrics
                    SET 
                        language_detection_start = NOW(),
                        language_detection_complete = NOW() + INTERVAL '%s milliseconds',
                        language = $2,
                        is_code_switched = $3
                    WHERE message_id = $1
                ''', message_id, results['primary_language'], results['is_code_switched'],
                    results['processing_time_ms'])
        except Exception as e:
            logger.error(f"Error storing detection metrics: {e}")

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
    processing_stages: Dict[str, float]

# API endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "language-detector",
        "uptime": time.time(),
        "models_loaded": True
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/detect", response_model=LanguageDetectionResponse)
async def detect_language(request: LanguageDetectionRequest):
    """Detect language(s) in the provided text with comprehensive metrics"""
    start_time = time.time()
    
    try:
        # Check cache first
        cache_key = f"lang:{request.message_id}" if request.message_id else None
        if cache_key:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for message {request.message_id}")
                return json.loads(cached_result)
        
        # Perform detection
        result = detector.detect_language(request.text, request.message_id)
        
        # Create response
        response = LanguageDetectionResponse(
            message_id=request.message_id,
            primary_language=result['primary_language'],
            confidence=result['confidence'],
            is_code_switched=result['is_code_switched'],
            all_languages=result['all_languages'],
            segments=result.get('segments'),
            processing_time_ms=result.get('processing_time_ms', (time.time() - start_time) * 1000),
            processing_stages=result.get('processing_stages', {})
        )
        
        # Cache result
        if cache_key:
            redis_client.setex(
                cache_key,
                3600,
                json.dumps(response.dict())
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        detection_errors_counter.labels(error_type='api_error', method='api').inc()
        raise HTTPException(status_code=500, detail=str(e))

# Kafka message processor with detailed metrics
async def process_kafka_messages():
    """Process messages from Kafka queue with comprehensive timing"""
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
            processing_start = time.time()
            
            try:
                message_data = msg.value
                message_id = message_data['id']
                
                # Update metrics with queue wait time
                if 'metrics' in message_data and 'queueTime' in message_data['metrics']:
                    queue_wait_time = time.time() * 1000 - message_data['metrics']['queueTime']
                    message_data['metrics']['queueWaitMs'] = queue_wait_time
                
                logger.info(f"Processing message: {message_id}")
                
                # Detect language with timing
                detection_start = time.time()
                result = detector.detect_language(message_data['content'], message_id)
                detection_time = (time.time() - detection_start) * 1000
                
                # Add language information to message
                message_data['language'] = result['primary_language']
                message_data['language_confidence'] = result['confidence']
                message_data['is_code_switched'] = result['is_code_switched']
                message_data['language_details'] = result
                
                # Update metrics
                if 'metrics' not in message_data:
                    message_data['metrics'] = {}
                message_data['metrics']['languageDetectionMs'] = detection_time
                message_data['metrics']['languageDetectionComplete'] = time.time() * 1000
                
                # Send to NLP processor
                kafka_start = time.time()
                await producer.send(
                    'nlp-processing',
                    key=message_id.encode('utf-8'),
                    value=message_data
                )
                kafka_time = (time.time() - kafka_start) * 1000
                
                # Log detailed metrics
                total_time = (time.time() - processing_start) * 1000
                logger.info(f"Message {message_id} processed", {
                    'language': result['primary_language'],
                    'confidence': result['confidence'],
                    'is_code_switched': result['is_code_switched'],
                    'detection_time_ms': detection_time,
                    'kafka_time_ms': kafka_time,
                    'total_time_ms': total_time
                })
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                detection_errors_counter.labels(error_type='kafka_processing', method='kafka').inc()
                
    finally:
        await consumer.stop()
        await producer.stop()

# Initialize database connection
async def init_db():
    global pg_pool
    pg_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=5432,
        database='ai_crm',
        user='admin',
        password='secure_password'
    )

# Start Kafka consumer on startup
@app.on_event("startup")
async def startup_event():
    await init_db()
    asyncio.create_task(process_kafka_messages())
    logger.info("Language Detection Service started with comprehensive metrics")

@app.on_event("shutdown")
async def shutdown_event():
    if consumer:
        await consumer.stop()
    if producer:
        await producer.stop()
    if pg_pool:
        await pg_pool.close()
    logger.info("Language Detection Service stopped")

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3002)
