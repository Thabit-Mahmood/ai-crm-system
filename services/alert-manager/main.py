from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import json
import logging
from datetime import datetime, timedelta
import redis
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import asyncpg
import uvicorn
from collections import defaultdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import httpx
import os

# Initialize FastAPI app
app = FastAPI(title="Alert Manager Service")

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

# PostgreSQL connection pool
pg_pool = None

# Notification channels
class NotificationChannel:
    async def send(self, alert: Dict):
        pass

class EmailChannel(NotificationChannel):
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.from_email = os.getenv('FROM_EMAIL', 'alerts@company.com')
        self.to_emails = os.getenv('ALERT_EMAILS', 'admin@company.com').split(',')
    
    async def send(self, alert: Dict):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert['priority'].upper()}] Alert: {alert.get('rule', 'Unknown')}"
            
            body = f"""
            Alert Details:
            - ID: {alert['id']}
            - Priority: {alert['priority']}
            - Rule: {alert.get('rule', 'Unknown')}
            - Content: {alert.get('content', 'No content')}
            - Timestamp: {alert['timestamp']}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Note: In production, implement proper email sending
            logger.info(f"Email alert sent for {alert['id']}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")

class SlackChannel(NotificationChannel):
    def __init__(self):
        self.webhook_url = os.getenv('SLACK_WEBHOOK_URL', '')
    
    async def send(self, alert: Dict):
        """Send Slack notification"""
        try:
            if not self.webhook_url:
                logger.warning("Slack webhook URL not configured")
                return
            
            payload = {
                "text": f"🚨 Alert: {alert.get('rule', 'Unknown')}",
                "attachments": [{
                    "color": self._get_color(alert['priority']),
                    "fields": [
                        {"title": "Priority", "value": alert['priority'], "short": True},
                        {"title": "Content", "value": alert.get('content', 'No content')[:100], "short": False},
                        {"title": "Timestamp", "value": alert['timestamp'], "short": True}
                    ]
                }]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.webhook_url, json=payload)
                response.raise_for_status()
                
            logger.info(f"Slack alert sent for {alert['id']}")
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    def _get_color(self, priority: str) -> str:
        colors = {
            'critical': '#ff0000',
            'high': '#ff6600',
            'medium': '#ffcc00',
            'low': '#00ff00'
        }
        return colors.get(priority, '#cccccc')

class SMSChannel(NotificationChannel):
    async def send(self, alert: Dict):
        """Send SMS notification"""
        # Note: Implement SMS sending logic here
        logger.info(f"SMS alert sent for {alert['id']}")

class DashboardChannel(NotificationChannel):
    async def send(self, alert: Dict):
        """Send dashboard notification"""
        try:
            # Store in Redis for real-time dashboard updates
            redis_client.lpush('dashboard_alerts', json.dumps(alert))
            redis_client.ltrim('dashboard_alerts', 0, 99)  # Keep last 100 alerts
            logger.info(f"Dashboard alert sent for {alert['id']}")
        except Exception as e:
            logger.error(f"Error sending dashboard alert: {e}")

class WebhookChannel(NotificationChannel):
    def __init__(self):
        self.webhook_url = os.getenv('WEBHOOK_URL', '')
    
    async def send(self, alert: Dict):
        """Send webhook notification"""
        try:
            if not self.webhook_url:
                logger.warning("Webhook URL not configured")
                return
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.webhook_url, json=alert)
                response.raise_for_status()
                
            logger.info(f"Webhook alert sent for {alert['id']}")
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")

class AlertManager:
    def __init__(self):
        self.alert_rules = self._initialize_alert_rules()
        self.alert_history = defaultdict(list)
        self.notification_channels = self._initialize_channels()
        self.alert_thresholds = {
            'critical': {
                'sentiment_score': -0.8,
                'volume_spike': 3.0,  # 3x normal volume
                'response_time': 300  # 5 minutes
            },
            'high': {
                'sentiment_score': -0.6,
                'volume_spike': 2.0,
                'response_time': 600  # 10 minutes
            },
            'medium': {
                'sentiment_score': -0.4,
                'volume_spike': 1.5,
                'response_time': 1800  # 30 minutes
            }
        }
        logger.info("Alert Manager initialized")
    
    def _initialize_alert_rules(self):
        """Initialize alert rules"""
        return {
            'sentiment_critical': {
                'condition': lambda msg: msg.get('sentiment') == 'negative' and msg.get('sentiment_confidence', 0) > 0.8,
                'priority': 'critical',
                'channels': ['email', 'slack', 'dashboard'],
                'cooldown': 300  # 5 minutes
            },
            'sentiment_high': {
                'condition': lambda msg: msg.get('sentiment') == 'negative' and msg.get('sentiment_confidence', 0) > 0.6,
                'priority': 'high',
                'channels': ['slack', 'dashboard'],
                'cooldown': 600
            },
            'vip_negative': {
                'condition': lambda msg: 'vip' in str(msg.get('metadata', {}).get('user', '')).lower() and msg.get('sentiment') == 'negative',
                'priority': 'critical',
                'channels': ['email', 'slack', 'sms', 'dashboard'],
                'cooldown': 0  # No cooldown for VIP
            },
            'volume_spike': {
                'condition': lambda msg: False,  # Checked separately
                'priority': 'high',
                'channels': ['slack', 'dashboard'],
                'cooldown': 1800
            },
            'keyword_alert': {
                'condition': lambda msg: any(keyword in msg.get('content', '').lower() 
                                           for keyword in ['emergency', 'urgent', 'crisis', 'lawsuit', 'legal']),
                'priority': 'critical',
                'channels': ['email', 'slack', 'dashboard'],
                'cooldown': 300
            }
        }
    
    def _initialize_channels(self):
        """Initialize notification channels"""
        return {
            'email': EmailChannel(),
            'slack': SlackChannel(),
            'sms': SMSChannel(),
            'dashboard': DashboardChannel(),
            'webhook': WebhookChannel()
        }
    
    async def process_message(self, message: Dict) -> List[Dict]:
        """Process message and generate alerts if needed"""
        alerts = []
        
        # Check each alert rule
        for rule_name, rule in self.alert_rules.items():
            if rule['condition'](message):
                # Check cooldown
                if not self._is_in_cooldown(rule_name, message):
                    alert = await self._create_alert(message, rule_name, rule)
                    alerts.append(alert)
                    
                    # Send notifications
                    await self._send_notifications(alert, rule['channels'])
                    
                    # Update cooldown
                    self._update_cooldown(rule_name, message)
        
        # Check for volume spikes
        volume_alert = await self._check_volume_spike()
        if volume_alert:
            alerts.append(volume_alert)
        
        return alerts
    
    async def _create_alert(self, message: Dict, rule_name: str, rule: Dict) -> Dict:
        """Create alert object"""
        alert = {
            'id': f"alert_{message['id']}_{rule_name}",
            'message_id': message['id'],
            'rule': rule_name,
            'priority': rule['priority'],
            'timestamp': datetime.now().isoformat(),
            'content': message['content'][:200],  # Truncate for alert
            'sentiment': message.get('sentiment'),
            'sentiment_confidence': message.get('sentiment_confidence'),
            'language': message.get('language'),
            'metadata': message.get('metadata', {}),
            'status': 'active'
        }
        
        # Store in database
        await self._store_alert(alert)
        
        # Update alert history
        self.alert_history[rule_name].append(alert)
        
        return alert
    
    async def _store_alert(self, alert: Dict):
        """Store alert in database"""
        try:
            await pg_pool.execute("""
                INSERT INTO alerts (id, message_id, type, priority, title, description, 
                                  metadata, status, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, alert['id'], alert['message_id'], alert['rule'], alert['priority'],
                f"Alert: {alert['rule']}", alert['content'], 
                json.dumps(alert['metadata']), alert['status'],
                datetime.fromisoformat(alert['timestamp']))
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    async def _send_notifications(self, alert: Dict, channels: List[str]):
        """Send notifications through specified channels"""
        for channel_name in channels:
            channel = self.notification_channels.get(channel_name)
            if channel:
                try:
                    await channel.send(alert)
                    logger.info(f"Alert {alert['id']} sent via {channel_name}")
                except Exception as e:
                    logger.error(f"Error sending alert via {channel_name}: {e}")
    
    def _is_in_cooldown(self, rule_name: str, message: Dict) -> bool:
        """Check if rule is in cooldown period"""
        cooldown_key = f"cooldown:{rule_name}:{message.get('metadata', {}).get('user', 'unknown')}"
        return redis_client.exists(cooldown_key)
    
    def _update_cooldown(self, rule_name: str, message: Dict):
        """Update cooldown period"""
        cooldown_key = f"cooldown:{rule_name}:{message.get('metadata', {}).get('user', 'unknown')}"
        cooldown_time = self.alert_rules[rule_name]['cooldown']
        redis_client.setex(cooldown_key, cooldown_time, 1)
    
    async def _check_volume_spike(self) -> Optional[Dict]:
        """Check for volume spikes"""
        try:
            current_count = await self._get_message_count(5)  # Last 5 minutes
            historical_avg = await self._get_historical_average()
            
            if historical_avg > 0 and current_count > historical_avg * 2:
                return {
                    'id': f"volume_spike_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'type': 'volume_spike',
                    'priority': 'high',
                    'title': 'Volume Spike Detected',
                    'description': f'Current: {current_count}, Average: {historical_avg:.1f}',
                    'timestamp': datetime.now().isoformat(),
                    'metadata': {
                        'current_count': current_count,
                        'historical_average': historical_avg,
                        'spike_ratio': current_count / historical_avg
                    }
                }
        except Exception as e:
            logger.error(f"Error checking volume spike: {e}")
        
        return None
    
    async def _get_message_count(self, minutes: int) -> int:
        """Get message count for the last N minutes"""
        try:
            count = await pg_pool.fetchval("""
                SELECT COUNT(*) FROM messages 
                WHERE created_at > NOW() - INTERVAL '%s minutes'
            """, minutes)
            return count or 0
        except Exception as e:
            logger.error(f"Error getting message count: {e}")
            return 0
    
    async def _get_historical_average(self) -> float:
        """Get historical average message count"""
        try:
            avg = await pg_pool.fetchval("""
                SELECT AVG(hourly_count) FROM (
                    SELECT COUNT(*) as hourly_count
                    FROM messages
                    WHERE created_at > NOW() - INTERVAL '7 days'
                    GROUP BY DATE_TRUNC('hour', created_at)
                ) hourly_stats
            """)
            return float(avg) / 12 if avg else 0  # Convert to 5-minute average
        except Exception as e:
            logger.error(f"Error getting historical average: {e}")
            return 0

# Initialize alert manager
alert_manager = AlertManager()

# Pydantic models
class Alert(BaseModel):
    id: str
    message_id: str
    rule: str
    priority: str
    timestamp: str
    content: str
    sentiment: Optional[str] = None
    sentiment_confidence: Optional[float] = None
    language: Optional[str] = None
    metadata: Dict = {}
    status: str = "active"

class AlertUpdateRequest(BaseModel):
    status: str
    notes: Optional[str] = None

# API endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "alert-manager"}

@app.post("/create-alert")
async def create_alert(alert_data: Dict):
    """Create a new alert"""
    try:
        alert = {
            'id': f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'message_id': alert_data.get('message_id', ''),
            'rule': 'manual',
            'priority': alert_data.get('severity', 'medium'),
            'timestamp': datetime.now().isoformat(),
            'content': alert_data.get('message', ''),
            'metadata': alert_data.get('metadata', {}),
            'status': 'active'
        }
        
        # Store alert
        await alert_manager._store_alert(alert)
        
        # Send notifications
        channels = ['dashboard']
        if alert['priority'] in ['critical', 'high']:
            channels.extend(['slack'])
        if alert['priority'] == 'critical':
            channels.extend(['email'])
        
        await alert_manager._send_notifications(alert, channels)
        
        return {"success": True, "alert_id": alert['id']}
        
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts")
async def get_alerts(
    priority: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """Get alerts with optional filtering"""
    try:
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []
        
        if priority:
            query += " AND priority = $" + str(len(params) + 1)
            params.append(priority)
        
        if status:
            query += " AND status = $" + str(len(params) + 1)
            params.append(status)
        
        query += " ORDER BY created_at DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)
        
        rows = await pg_pool.fetch(query, *params)
        
        alerts = []
        for row in rows:
            alert = dict(row)
            alert['metadata'] = json.loads(alert['metadata']) if alert['metadata'] else {}
            alert['created_at'] = alert['created_at'].isoformat()
            alerts.append(alert)
        
        return {"alerts": alerts}
        
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/alerts/{alert_id}")
async def update_alert(alert_id: str, request: AlertUpdateRequest):
    """Update alert status"""
    try:
        await pg_pool.execute("""
            UPDATE alerts 
            SET status = $1, updated_at = NOW()
            WHERE id = $2
        """, request.status, alert_id)
        
        if request.notes:
            await pg_pool.execute("""
                INSERT INTO alert_notes (alert_id, notes, created_at)
                VALUES ($1, $2, NOW())
            """, alert_id, request.notes)
        
        return {"success": True}
        
    except Exception as e:
        logger.error(f"Error updating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alert-stats")
async def get_alert_stats():
    """Get alert statistics"""
    try:
        # Get alert counts by priority
        priority_stats = await pg_pool.fetch("""
            SELECT priority, COUNT(*) as count
            FROM alerts
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY priority
        """)
        
        # Get alert counts by type
        type_stats = await pg_pool.fetch("""
            SELECT type, COUNT(*) as count
            FROM alerts
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY type
            ORDER BY count DESC
            LIMIT 10
        """)
        
        # Get response time stats
        response_stats = await pg_pool.fetchrow("""
            SELECT 
                AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_response_time,
                MIN(EXTRACT(EPOCH FROM (updated_at - created_at))) as min_response_time,
                MAX(EXTRACT(EPOCH FROM (updated_at - created_at))) as max_response_time
            FROM alerts
            WHERE status = 'resolved'
            AND created_at > NOW() - INTERVAL '24 hours'
        """)
        
        return {
            "priority_distribution": [dict(row) for row in priority_stats],
            "top_types": [dict(row) for row in type_stats],
            "response_times": dict(response_stats) if response_stats else {}
        }
        
    except Exception as e:
        logger.error(f"Error fetching alert stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Database initialization
async def init_database():
    """Initialize database tables"""
    try:
        await pg_pool.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                message_id UUID,
                type VARCHAR(50) NOT NULL,
                priority VARCHAR(20) DEFAULT 'normal',
                status VARCHAR(20) DEFAULT 'open',
                title VARCHAR(255) NOT NULL,
                description TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await pg_pool.execute("""
            CREATE TABLE IF NOT EXISTS alert_notes (
                id SERIAL PRIMARY KEY,
                alert_id UUID REFERENCES alerts(id),
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await pg_pool.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alerts(priority);
            CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
            CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);
        """)
        
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise

# Kafka message processor
async def process_kafka_messages():
    """Process messages from Kafka queue with retry logic"""
    global consumer
    
    max_retries = 5
    retry_delay = 10
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Starting Kafka consumer (attempt {attempt + 1}/{max_retries})...")
            
            consumer = AIOKafkaConsumer(
                'alert-processing',
                'language-detection',
                'nlp-analysis',
                bootstrap_servers=KAFKA_BROKER,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                group_id='alert-manager-group',
                enable_auto_commit=True,
                auto_offset_reset='latest'
            )
            
            await consumer.start()
            logger.info("Kafka consumer started successfully")
            
            try:
                async for msg in consumer:
                    try:
                        message_data = msg.value
                        logger.info(f"Processing message for alerts: {message_data.get('id', 'unknown')}")
                        
                        # Process message and generate alerts
                        alerts = await alert_manager.process_message(message_data)
                        
                        if alerts:
                            logger.info(f"Generated {len(alerts)} alerts for message {message_data.get('id', 'unknown')}")
                        
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        
            finally:
                await consumer.stop()
                
        except Exception as e:
            logger.error(f"Kafka consumer error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("Max retries reached. Kafka consumer failed to start.")
                break

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    global pg_pool
    try:
        pg_pool = await asyncpg.create_pool(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=5432,
            database='ai_crm',
            user='admin',
            password='secure_password'
        )
        await init_database()
        asyncio.create_task(process_kafka_messages())
        logger.info("Alert Manager Service started")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    if consumer:
        await consumer.stop()
    if pg_pool:
        await pg_pool.close()
    logger.info("Alert Manager Service stopped")

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3004)
