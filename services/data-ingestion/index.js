const express = require('express');
const { Pool } = require('pg');
const Redis = require('ioredis');
const { Kafka } = require('kafkajs');
const cors = require('cors');
const bodyParser = require('body-parser');
const axios = require('axios');
const winston = require('winston');
const { v4: uuidv4 } = require('uuid');
const rateLimit = require('express-rate-limit');
const { performance } = require('perf_hooks');
const client = require('prom-client');

// Prometheus metrics
const register = new client.Registry();
const messageIngestionDuration = new client.Histogram({
  name: 'message_ingestion_duration_ms',
  help: 'Duration of message ingestion in milliseconds',
  labelNames: ['status', 'source'],
  buckets: [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
});
const messagesIngested = new client.Counter({
  name: 'messages_ingested_total',
  help: 'Total number of messages ingested',
  labelNames: ['source', 'priority']
});
const activeConnections = new client.Gauge({
  name: 'active_data_sources',
  help: 'Number of active data source connections',
  labelNames: ['type']
});
const queueDepth = new client.Gauge({
  name: 'ingestion_queue_depth',
  help: 'Current depth of ingestion queue'
});

register.registerMetric(messageIngestionDuration);
register.registerMetric(messagesIngested);
register.registerMetric(activeConnections);
register.registerMetric(queueDepth);
client.collectDefaultMetrics({ register });

// Initialize Express app
const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true }));

// Logger configuration
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }),
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});

// Database connection
const pgPool = new Pool({
  host: process.env.POSTGRES_HOST || 'localhost',
  port: 5432,
  database: 'ai_crm',
  user: 'admin',
  password: 'secure_password'
});

// Redis connection
const redis = new Redis({
  host: process.env.REDIS_HOST || 'localhost',
  port: 6379
});

// Kafka configuration
const kafka = new Kafka({
  clientId: 'data-ingestion-service',
  brokers: [process.env.KAFKA_BROKER || 'localhost:9092']
});

const producer = kafka.producer();

// Rate limiting configuration
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // limit each IP to 1000 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});

// Initialize database tables
async function initializeDatabase() {
  try {
    await pgPool.query(`
      CREATE TABLE IF NOT EXISTS data_sources (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name VARCHAR(255) NOT NULL,
        type VARCHAR(50) NOT NULL,
        config JSONB NOT NULL,
        status VARCHAR(50) DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    await pgPool.query(`
      CREATE TABLE IF NOT EXISTS messages (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source_id UUID REFERENCES data_sources(id),
        content TEXT NOT NULL,
        language VARCHAR(10),
        sentiment VARCHAR(20),
        sentiment_confidence FLOAT,
        metadata JSONB,
        processed BOOLEAN DEFAULT FALSE,
        priority VARCHAR(20) DEFAULT 'normal',
        is_code_switched BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    await pgPool.query(`
      CREATE TABLE IF NOT EXISTS alerts (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        message_id UUID REFERENCES messages(id),
        type VARCHAR(50) NOT NULL,
        priority VARCHAR(20) DEFAULT 'normal',
        status VARCHAR(20) DEFAULT 'open',
        title VARCHAR(255) NOT NULL,
        description TEXT,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    await pgPool.query(`
      CREATE INDEX IF NOT EXISTS idx_messages_processed ON messages(processed);
      CREATE INDEX IF NOT EXISTS idx_messages_priority ON messages(priority);
      CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
      CREATE INDEX IF NOT EXISTS idx_messages_sentiment ON messages(sentiment);
      CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
      CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alerts(priority);
      CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);
    `);

    logger.info('Database initialized successfully');
  } catch (error) {
    logger.error('Database initialization error:', error);
    throw error;
  }
}

// Data source connectors
class DataSourceConnector {
  constructor(config) {
    this.config = config;
    this.active = true;
  }

  async connect() {
    // Override in subclasses
  }

  async disconnect() {
    this.active = false;
  }

  async fetchData() {
    // Override in subclasses
  }
}

// Twitter connector
class TwitterConnector extends DataSourceConnector {
  async fetchData() {
    // Return empty array - no synthetic data
    // Real Twitter API integration would go here
    return [];
  }
}

// Data ingestion manager
class DataIngestionManager {
  constructor() {
    this.sources = new Map();
    this.intervals = new Map();
  }

  async addSource(sourceConfig) {
    const { id, type, config } = sourceConfig;
    let connector;

    switch (type) {
      case 'twitter':
        connector = new TwitterConnector(config);
        break;
      case 'facebook':
        // Add Facebook connector
        break;
      case 'email':
        // Add email connector
        break;
      default:
        throw new Error(`Unknown source type: ${type}`);
    }

    await connector.connect();
    this.sources.set(id, connector);

    // Start polling
    const interval = setInterval(async () => {
      await this.pollSource(id);
    }, config.pollInterval || 60000);

    this.intervals.set(id, interval);
    logger.info(`Added data source: ${id} (${type})`);
  }

  async removeSource(sourceId) {
    const connector = this.sources.get(sourceId);
    if (connector) {
      await connector.disconnect();
      this.sources.delete(sourceId);
    }

    const interval = this.intervals.get(sourceId);
    if (interval) {
      clearInterval(interval);
      this.intervals.delete(sourceId);
    }

    logger.info(`Removed data source: ${sourceId}`);
  }

  async pollSource(sourceId) {
    const connector = this.sources.get(sourceId);
    if (!connector || !connector.active) return;

    try {
      const data = await connector.fetchData();
      
      for (const item of data) {
        // Check for duplicates
        const cacheKey = `msg:${item.id}`;
        const exists = await redis.get(cacheKey);
        if (exists) continue;

        // Process and queue message
        const message = {
          id: item.id,
          sourceId: sourceId,
          content: item.text || item.content,
          metadata: {
            user: item.user,
            timestamp: item.timestamp,
            originalData: item
          },
          priority: this.calculatePriority(item),
          createdAt: new Date()
        };

        // Save to database
        await pgPool.query(
          `INSERT INTO messages (id, source_id, content, metadata, priority) 
           VALUES ($1, $2, $3, $4, $5)`,
          [message.id, message.sourceId, message.content, JSON.stringify(message.metadata), message.priority]
        );

        // Send to Kafka for processing
        try {
          await producer.send({
            topic: 'language-detection',
            messages: [{
              key: message.id,
              value: JSON.stringify(message)
            }]
          });
        } catch (kafkaError) {
          logger.warn(`Failed to send message to Kafka: ${kafkaError.message}`);
          // Continue processing even if Kafka fails
        }

        // Cache to prevent duplicates
        await redis.set(cacheKey, '1', 'EX', 3600);

        logger.info(`Ingested message: ${message.id} with priority: ${message.priority}`);
      }
    } catch (error) {
      logger.error(`Error polling source ${sourceId}:`, error);
    }
  }

  calculatePriority(item) {
    const content = (item.text || item.content || '').toLowerCase();
    
    // High priority keywords
    if (content.includes('urgent') || content.includes('emergency') || 
        content.includes('critical') || content.includes('complaint')) {
      return 'high';
    }
    
    // Check for VIP users (demo logic)
    if (item.user && item.user.includes('vip')) {
      return 'high';
    }
    
    // Negative sentiment keywords
    if (content.includes('disappointed') || content.includes('angry') || 
        content.includes('terrible') || content.includes('worst')) {
      return 'medium';
    }
    
    return 'normal';
  }
}

const ingestionManager = new DataIngestionManager();

// API Routes
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'data-ingestion' });
});

// Add data source
app.post('/api/sources', async (req, res) => {
  try {
    const { name, type, config } = req.body;
    
    const result = await pgPool.query(
      `INSERT INTO data_sources (name, type, config) 
       VALUES ($1, $2, $3) RETURNING *`,
      [name, type, JSON.stringify(config)]
    );
    
    const source = result.rows[0];
    await ingestionManager.addSource({
      id: source.id,
      type: source.type,
      config: { ...config, ...JSON.parse(source.config) }
    });
    
    res.json({ success: true, source });
  } catch (error) {
    logger.error('Error adding source:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get all sources
app.get('/api/sources', async (req, res) => {
  try {
    const result = await pgPool.query('SELECT * FROM data_sources ORDER BY created_at DESC');
    res.json(result.rows);
  } catch (error) {
    logger.error('Error fetching sources:', error);
    res.status(500).json({ error: error.message });
  }
});

// Delete source
app.delete('/api/sources/:id', async (req, res) => {
  try {
    const { id } = req.params;
    await ingestionManager.removeSource(id);
    await pgPool.query('UPDATE data_sources SET status = $1 WHERE id = $2', ['inactive', id]);
    res.json({ success: true });
  } catch (error) {
    logger.error('Error removing source:', error);
    res.status(500).json({ error: error.message });
  }
});

// Manual message ingestion
app.post('/api/messages', limiter, async (req, res) => {
  try {
    const { content, sourceId, metadata } = req.body;
    const messageId = uuidv4();
    
    const message = {
      id: messageId,
      sourceId: sourceId || null,  // Use null instead of 'manual' for UUID column
      content,
      metadata: metadata || {},
      priority: 'normal',
      createdAt: new Date()
    };
    
    // Save to database
    await pgPool.query(
      `INSERT INTO messages (id, source_id, content, metadata, priority) 
       VALUES ($1, $2, $3, $4, $5)`,
      [message.id, message.sourceId, message.content, JSON.stringify(message.metadata), message.priority]
    );
    
    // Send to Kafka
    try {
      await producer.send({
        topic: 'language-detection',
        messages: [{
          key: message.id,
          value: JSON.stringify(message)
        }]
      });
    } catch (kafkaError) {
      logger.warn(`Failed to send message to Kafka: ${kafkaError.message}`);
      // Continue processing even if Kafka fails
    }
    
    res.json({ success: true, messageId });
  } catch (error) {
    logger.error('Error ingesting message:', error);
    res.status(500).json({ error: error.message });
  }
});

// Test message ingestion with immediate sentiment analysis
app.post('/api/test-sentiment', async (req, res) => {
  try {
    const { content } = req.body;
    const messageId = uuidv4();
    
    const message = {
      id: messageId,
      sourceId: null,
      content,
      metadata: {
        source: 'dashboard-test',
        timestamp: new Date().toISOString()
      },
      priority: 'normal',
      createdAt: new Date()
    };
    
    // Save to database
    await pgPool.query(
      `INSERT INTO messages (id, source_id, content, metadata, priority) 
       VALUES ($1, $2, $3, $4, $5)`,
      [message.id, message.sourceId, message.content, JSON.stringify(message.metadata), message.priority]
    );
    
    // Call language detector directly
    let language = 'en';
    let isCodeSwitched = false;
    
    try {
      const langResponse = await axios.post('http://language-detector:3002/detect', { 
        text: content,
        message_id: messageId 
      });
      language = langResponse.data.primary_language;
      isCodeSwitched = langResponse.data.is_code_switched;
    } catch (error) {
      logger.warn('Language detection failed, using default');
    }
    
    // Call NLP processor directly
    let sentimentResult = null;
    try {
      const nlpResponse = await axios.post('http://nlp-processor:8000/analyze', { 
        text: content,
        message_id: messageId,
        language: language,
        is_code_switched: isCodeSwitched
      });
      sentimentResult = nlpResponse.data;
      
      // Update message with results
      await pgPool.query(
        `UPDATE messages 
         SET language = $1, 
             sentiment = $2, 
             sentiment_confidence = $3,
             is_code_switched = $4,
             processed = true,
             updated_at = CURRENT_TIMESTAMP
         WHERE id = $5`,
        [language, sentimentResult.sentiment, sentimentResult.confidence, isCodeSwitched, messageId]
      );
      
      // Check if we need to create an alert
      if (sentimentResult.sentiment === 'negative' && sentimentResult.confidence > 0.8) {
        await pgPool.query(
          `INSERT INTO alerts (message_id, type, priority, status, title, description, metadata)
           VALUES ($1, $2, $3, $4, $5, $6, $7)`,
          [
            messageId,
            'negative_sentiment',
            sentimentResult.confidence > 0.9 ? 'critical' : 'high',
            'open',
            'Negative Sentiment Detected',
            `High confidence (${(sentimentResult.confidence * 100).toFixed(1)}%) negative sentiment in message`,
            JSON.stringify({
              sentiment: sentimentResult.sentiment,
              confidence: sentimentResult.confidence,
              content: content.substring(0, 200)
            })
          ]
        );
      }
    } catch (error) {
      logger.error('Sentiment analysis failed:', error);
      // Return partial result even if sentiment fails
      return res.json({
        success: true,
        messageId,
        language,
        isCodeSwitched,
        sentiment: 'unknown',
        confidence: 0,
        error: 'Sentiment analysis failed'
      });
    }
    
    // Return complete result with sentiment data
    res.json({
      success: true,
      messageId,
      language,
      isCodeSwitched,
      sentiment: sentimentResult?.sentiment || 'unknown',
      confidence: sentimentResult?.confidence || 0,
      processing_time_ms: sentimentResult?.processing_time_ms || 0,
      model_used: sentimentResult?.model_used || 'unknown'
    });
    
  } catch (error) {
    logger.error('Error processing test sentiment:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get ingestion statistics
app.get('/api/stats', async (req, res) => {
  try {
    const stats = await pgPool.query(`
      SELECT 
        COUNT(*) as total_messages,
        COUNT(CASE WHEN processed = true THEN 1 END) as processed_messages,
        COUNT(CASE WHEN priority = 'high' THEN 1 END) as high_priority,
        COUNT(CASE WHEN priority = 'medium' THEN 1 END) as medium_priority,
        COUNT(CASE WHEN priority = 'normal' THEN 1 END) as normal_priority,
        COUNT(CASE WHEN created_at > NOW() - INTERVAL '1 hour' THEN 1 END) as last_hour
      FROM messages
    `);
    
    const sources = await pgPool.query(`
      SELECT type, COUNT(*) as count 
      FROM data_sources 
      WHERE status = 'active' 
      GROUP BY type
    `);
    
    res.json({
      messages: stats.rows[0],
      sources: sources.rows
    });
  } catch (error) {
    logger.error('Error fetching stats:', error);
    res.status(500).json({ error: error.message });
  }
});

// Start server
async function start() {
  try {
    await initializeDatabase();
    
    // Connect to Kafka with retry logic
    const maxRetries = 5;
    let retryDelay = 5000;
    let kafkaConnected = false;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        logger.info(`Connecting to Kafka (attempt ${attempt}/${maxRetries})...`);
        await producer.connect();
        logger.info('Kafka producer connected successfully');
        kafkaConnected = true;
        break;
      } catch (error) {
        logger.error(`Kafka connection attempt ${attempt}/${maxRetries} failed:`, error.message);
        if (attempt < maxRetries) {
          logger.info(`Retrying in ${retryDelay / 1000} seconds...`);
          await new Promise(resolve => setTimeout(resolve, retryDelay));
          retryDelay *= 2; // Exponential backoff
        }
      }
    }
    
    if (!kafkaConnected) {
      logger.warn('Kafka connection failed after all retries. Service will start without Kafka.');
    }
    
    const PORT = process.env.PORT || 3001;
    app.listen(PORT, () => {
      logger.info(`Data Ingestion Service running on port ${PORT}`);
    });
  } catch (error) {
    logger.error('Failed to start service:', error);
    process.exit(1);
  }
}

start();
