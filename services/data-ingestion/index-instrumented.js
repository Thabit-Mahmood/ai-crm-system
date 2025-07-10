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

// Define metrics
const messageIngestionDuration = new client.Histogram({
  name: 'message_ingestion_duration_ms',
  help: 'Duration of message ingestion in milliseconds',
  labelNames: ['status', 'source', 'stage'],
  buckets: [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
});

const messagesIngested = new client.Counter({
  name: 'messages_ingested_total',
  help: 'Total number of messages ingested',
  labelNames: ['source', 'priority', 'language']
});

const activeConnections = new client.Gauge({
  name: 'active_data_sources',
  help: 'Number of active data source connections',
  labelNames: ['type']
});

const queueDepth = new client.Gauge({
  name: 'ingestion_queue_depth',
  help: 'Current depth of ingestion queue',
  labelNames: ['queue']
});

const ingestionErrors = new client.Counter({
  name: 'ingestion_errors_total',
  help: 'Total number of ingestion errors',
  labelNames: ['source', 'error_type']
});

// Register metrics
register.registerMetric(messageIngestionDuration);
register.registerMetric(messagesIngested);
register.registerMetric(activeConnections);
register.registerMetric(queueDepth);
register.registerMetric(ingestionErrors);

// Collect default metrics
client.collectDefaultMetrics({ register });

// Initialize Express app
const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true }));

// Logger configuration with microsecond precision
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss.SSS' }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }),
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'metrics.log' })
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
  message: 'Too many requests from this IP, please try again later.',
  handler: (req, res) => {
    ingestionErrors.inc({ source: 'api', error_type: 'rate_limit' });
    res.status(429).json({ error: 'Rate limit exceeded' });
  }
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
        metadata JSONB,
        sentiment_score FLOAT,
        processed BOOLEAN DEFAULT FALSE,
        priority VARCHAR(20) DEFAULT 'normal',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ingestion_metrics JSONB
      )
    `);

    await pgPool.query(`
      CREATE INDEX IF NOT EXISTS idx_messages_processed ON messages(processed);
      CREATE INDEX IF NOT EXISTS idx_messages_priority ON messages(priority);
      CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
    `);

    logger.info('Database initialized successfully');
  } catch (error) {
    logger.error('Database initialization error:', error);
    throw error;
  }
}

// Data source connectors with metrics
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

// Twitter connector with realistic data simulation
class TwitterConnector extends DataSourceConnector {
  async fetchData() {
    // Simulate realistic Twitter data with various languages
    const messages = [
      {
        id: uuidv4(),
        text: "Love the new features in your product! 太好了！",
        user: "@happy_customer",
        timestamp: new Date(),
        language_hint: 'code_switched'
      },
      {
        id: uuidv4(),
        text: "Having issues with customer support. Very disappointed.",
        user: "@angry_user",
        timestamp: new Date(),
        language_hint: 'en'
      },
      {
        id: uuidv4(),
        text: "مرحبا، الخدمة ممتازة! Great service!",
        user: "@multilingual_user",
        timestamp: new Date(),
        language_hint: 'code_switched'
      },
      {
        id: uuidv4(),
        text: "El producto es excelente, pero el envío fue lento.",
        user: "@spanish_customer",
        timestamp: new Date(),
        language_hint: 'es'
      },
      {
        id: uuidv4(),
        text: "这个产品质量很好，推荐购买！",
        user: "@chinese_reviewer",
        timestamp: new Date(),
        language_hint: 'zh'
      }
    ];
    
    // Return a random subset to simulate real-time flow
    const count = Math.floor(Math.random() * 3) + 1;
    return messages.slice(0, count);
  }
}

// Data ingestion manager with comprehensive metrics
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
    activeConnections.inc({ type });

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
      activeConnections.dec({ type: connector.constructor.name });
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
        const metrics = {
          arrivalTime: Date.now(),
          ingestionStart: performance.now(),
          stages: {}
        };
        
        // Check for duplicates
        const duplicateCheckStart = performance.now();
        const cacheKey = `msg:${item.id}`;
        const exists = await redis.get(cacheKey);
        metrics.stages.duplicateCheck = performance.now() - duplicateCheckStart;
        
        if (exists) {
          logger.debug(`Duplicate message detected: ${item.id}`);
          continue;
        }

        // Process and queue message
        const processingStart = performance.now();
        const message = {
          id: item.id,
          sourceId: sourceId,
          content: item.text || item.content,
          metadata: {
            user: item.user,
            timestamp: item.timestamp,
            originalData: item,
            ingestionTimestamp: new Date().toISOString(),
            arrivalTimestamp: metrics.arrivalTime,
            language_hint: item.language_hint
          },
          priority: this.calculatePriority(item),
          createdAt: new Date()
        };
        metrics.stages.processing = performance.now() - processingStart;

        // Save to database with timing
        const dbStart = performance.now();
        await pgPool.query(
          `INSERT INTO messages (id, source_id, content, metadata, priority, ingestion_metrics) 
           VALUES ($1, $2, $3, $4, $5, $6)`,
          [
            message.id, 
            message.sourceId, 
            message.content, 
            JSON.stringify(message.metadata), 
            message.priority,
            JSON.stringify(metrics)
          ]
        );
        metrics.stages.database = performance.now() - dbStart;

        // Send to Kafka for processing with timing
        const kafkaStart = performance.now();
        await producer.send({
          topic: 'language-detection',
          messages: [{
            key: message.id,
            value: JSON.stringify({
              ...message,
              metrics: {
                ...metrics,
                queueTime: performance.now()
              }
            })
          }]
        });
        metrics.stages.kafka = performance.now() - kafkaStart;

        // Cache to prevent duplicates
        const cacheStart = performance.now();
        await redis.set(cacheKey, '1', 'EX', 3600);
        metrics.stages.cache = performance.now() - cacheStart;

        // Calculate total metrics
        const totalDuration = performance.now() - metrics.ingestionStart;
        
        // Record Prometheus metrics
        messageIngestionDuration.observe({ 
          status: 'success', 
          source: connector.constructor.name,
          stage: 'total'
        }, totalDuration);
        
        // Record stage-specific metrics
        for (const [stage, duration] of Object.entries(metrics.stages)) {
          messageIngestionDuration.observe({
            status: 'success',
            source: connector.constructor.name,
            stage
          }, duration);
        }
        
        messagesIngested.inc({ 
          source: connector.constructor.name, 
          priority: message.priority,
          language: item.language_hint || 'unknown'
        });
        
        // Update queue depth
        const currentDepth = await redis.llen('language-detection');
        queueDepth.set({ queue: 'language-detection' }, currentDepth);

        logger.info(`Ingested message: ${message.id}`, {
          priority: message.priority,
          duration: totalDuration,
          stages: metrics.stages
        });
      }
    } catch (error) {
      logger.error(`Error polling source ${sourceId}:`, error);
      ingestionErrors.inc({ source: sourceId, error_type: error.name });
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
  res.json({ 
    status: 'healthy', 
    service: 'data-ingestion',
    uptime: process.uptime(),
    memory: process.memoryUsage()
  });
});

// Prometheus metrics endpoint
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});

// Add data source
app.post('/api/sources', async (req, res) => {
  const start = performance.now();
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
    
    const duration = performance.now() - start;
    res.json({ success: true, source, processingTimeMs: duration });
  } catch (error) {
    logger.error('Error adding source:', error);
    ingestionErrors.inc({ source: 'api', error_type: 'add_source' });
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
    ingestionErrors.inc({ source: 'api', error_type: 'remove_source' });
    res.status(500).json({ error: error.message });
  }
});

// Manual message ingestion with comprehensive metrics
app.post('/api/messages', limiter, async (req, res) => {
  const metrics = {
    arrivalTime: Date.now(),
    ingestionStart: performance.now(),
    stages: {}
  };
  
  try {
    const { content, sourceId, metadata } = req.body;
    const messageId = uuidv4();
    
    // Validation timing
    const validationStart = performance.now();
    if (!content || content.trim().length === 0) {
      throw new Error('Content is required');
    }
    metrics.stages.validation = performance.now() - validationStart;
    
    // Process message
    const processingStart = performance.now();
    const message = {
      id: messageId,
      sourceId: sourceId || 'manual',
      content,
      metadata: {
        ...metadata,
        ingestionTimestamp: new Date().toISOString(),
        arrivalTimestamp: metrics.arrivalTime,
        source: 'api'
      },
      priority: 'normal',
      createdAt: new Date()
    };
    
    // Calculate priority
    const priority = ingestionManager.calculatePriority({
      text: content,
      user: metadata?.user
    });
    message.priority = priority;
    metrics.stages.processing = performance.now() - processingStart;
    
    // Save to database with timing
    const dbStart = performance.now();
    await pgPool.query(
      `INSERT INTO messages (id, source_id, content, metadata, priority, ingestion_metrics) 
       VALUES ($1, $2, $3, $4, $5, $6)`,
      [
        message.id, 
        message.sourceId, 
        message.content, 
        JSON.stringify(message.metadata), 
        message.priority,
        JSON.stringify(metrics)
      ]
    );
    metrics.stages.database = performance.now() - dbStart;
    
    // Send to Kafka with timing
    const kafkaStart = performance.now();
    await producer.send({
      topic: 'language-detection',
      messages: [{
        key: message.id,
        value: JSON.stringify({
          ...message,
          metrics: {
            ...metrics,
            queueTime: performance.now()
          }
        })
      }]
    });
    metrics.stages.kafka = performance.now() - kafkaStart;
    
    // Calculate total duration
    const totalDuration = performance.now() - metrics.ingestionStart;
    
    // Record metrics
    messageIngestionDuration.observe({ 
      status: 'success', 
      source: 'api',
      stage: 'total'
    }, totalDuration);
    
    messagesIngested.inc({ 
      source: 'api', 
      priority: message.priority,
      language: metadata?.language || 'unknown'
    });
    
    // Update queue depth
    const currentDepth = await redis.llen('language-detection');
    queueDepth.set({ queue: 'language-detection' }, currentDepth);
    
    res.json({ 
      success: true, 
      messageId,
      processingTimeMs: totalDuration,
      metrics: {
        total: totalDuration,
        stages: metrics.stages
      }
    });
  } catch (error) {
    logger.error('Error ingesting message:', error);
    const duration = performance.now() - metrics.ingestionStart;
    messageIngestionDuration.observe({ 
      status: 'error', 
      source: 'api',
      stage: 'total'
    }, duration);
    ingestionErrors.inc({ source: 'api', error_type: error.name });
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
        COUNT(CASE WHEN created_at > NOW() - INTERVAL '1 hour' THEN 1 END) as last_hour,
        AVG(CASE 
          WHEN ingestion_metrics IS NOT NULL 
          THEN (ingestion_metrics->>'stages'->>'total')::FLOAT 
          ELSE NULL 
        END) as avg_ingestion_time_ms
      FROM messages
    `);
    
    const sources = await pgPool.query(`
      SELECT type, COUNT(*) as count 
      FROM data_sources 
      WHERE status = 'active' 
      GROUP BY type
    `);
    
    // Get current queue depths
    const queues = ['language-detection', 'nlp-processing', 'alert-processing'];
    const queueDepths = {};
    for (const queue of queues) {
      queueDepths[queue] = await redis.llen(queue);
    }
    
    res.json({
      messages: stats.rows[0],
      sources: sources.rows,
      queues: queueDepths,
      timestamp: new Date().toISOString()
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
    await producer.connect();
    
    // Add demo source after a delay
    setTimeout(async () => {
      const demoSource = {
        name: 'Demo Twitter Feed',
        type: 'twitter',
        config: { pollInterval: 10000 } // Poll every 10 seconds
      };
      
      const existing = await pgPool.query(
        'SELECT id FROM data_sources WHERE name = $1',
        [demoSource.name]
      );
      
      if (existing.rows.length === 0) {
        const result = await pgPool.query(
          `INSERT INTO data_sources (name, type, config) 
           VALUES ($1, $2, $3) RETURNING *`,
          [demoSource.name, demoSource.type, JSON.stringify(demoSource.config)]
        );
        
        await ingestionManager.addSource({
          id: result.rows[0].id,
          type: demoSource.type,
          config: demoSource.config
        });
        
        logger.info('Demo data source added');
      }
    }, 2000);
    
    const PORT = process.env.PORT || 3001;
    app.listen(PORT, () => {
      logger.info(`Data Ingestion Service running on port ${PORT}`);
      logger.info(`Metrics available at http://localhost:${PORT}/metrics`);
    });
  } catch (error) {
    logger.error('Failed to start service:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM signal received: closing HTTP server');
  await producer.disconnect();
  await redis.disconnect();
  await pgPool.end();
  process.exit(0);
});

start();
