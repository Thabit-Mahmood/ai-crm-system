import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost';

// Create a single, unified axios instance for the API gateway
const api = axios.create({
  baseURL: API_BASE_URL
});

// Add a request interceptor for error handling and resilience
// Set timeout for all requests
api.defaults.timeout = 15000; // 15 seconds

api.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error);
    
    // For connection errors, return a fallback response
    if (error.code === 'ERR_NETWORK' || error.code === 'ECONNREFUSED' || error.message === 'Network Error' || error.code === 'ECONNABORTED') {
      console.warn(`Service temporarily unavailable: ${error.config?.baseURL}${error.config?.url}`);
      // Don't return a resolved promise - let each API function handle it
      return Promise.reject({
        ...error,
        isServiceUnavailable: true
      });
    }
    
    throw error;
  }
);

// Dashboard API
export const fetchDashboardData = async () => {
  try {
    const response = await api.get('/api/analytics/dashboard');
    return response.data;
  } catch (error) {
    console.warn('Dashboard data unavailable, returning empty data');
    return { summary: {}, charts: [], alerts: [] };
  }
};

// Sentiment Analysis APIs
export const fetchSentimentAnalytics = async (timeRange = '24h') => {
  try {
    const response = await api.get(`/api/analytics/sentiment?time_range=${timeRange}`);
    return response.data;
  } catch (error) {
    console.warn('Sentiment analytics unavailable, returning empty data');
    return { sentiment_distribution: [], trends: [], total_analyzed: 0 };
  }
};

export const analyzeSentiment = async (text, language = null) => {
  try {
    // Use the new test-sentiment endpoint that saves data to database
    const response = await api.post('/api/ingestion/test-sentiment', { content: text });
    return response.data;
  } catch (error) {
    console.warn('Sentiment analysis failed, trying direct NLP service');
    try {
      // Fallback to direct NLP service
      const response = await api.post('/api/nlp/analyze', { text, language });
      return response.data;
    } catch (fallbackError) {
      console.warn('NLP analysis unavailable, returning neutral sentiment');
      return { 
        sentiment: 'neutral',
        confidence: 0,
        processing_time_ms: 0,
        model_used: 'service-unavailable',
        language: language || 'unknown',
        error: 'NLP service is starting up...'
      };
    }
  }
};

// Alert APIs
export const fetchAlerts = async (params = {}) => {
  try {
    const queryString = new URLSearchParams(params).toString();
    const response = await api.get(`/api/alerts/alerts${queryString ? '?' + queryString : ''}`);
    return response.data;
  } catch (error) {
    console.warn('Alert fetch failed, returning empty data');
    return { alerts: [] };
  }
};

export const updateAlert = async (alertId, status, notes = null) => {
  const response = await api.put(`/api/alerts/alerts/${alertId}`, { status, notes });
  return response.data;
};

export const fetchAlertStats = async () => {
  try {
    const response = await api.get('/api/analytics/alerts');
    // Transform the data to match expected format
    const data = response.data;
    const priority_distribution = Object.entries(data.alert_summary || {}).map(([priority, count]) => ({
      priority,
      count
    }));
    return {
      priority_distribution,
      response_times: data.response_times || data.resolution_times || {},
      total_alerts_7d: data.total_alerts_7d || 0
    };
  } catch (error) {
    console.warn('Alert analytics unavailable, returning empty data');
    return { priority_distribution: [], response_times: {}, total_alerts_7d: 0 };
  }
};

// Language APIs
export const fetchLanguageAnalytics = async () => {
  try {
    const response = await api.get('/api/analytics/languages');
    return response.data;
  } catch (error) {
    console.warn('Language analytics unavailable, returning empty data');
    return { languages: [], total_detected: 0 };
  }
};

export const detectLanguage = async (text) => {
  const response = await api.post('/api/language/detect', { text });
  return response.data;
};

// Performance APIs
export const fetchPerformanceMetrics = async () => {
  try {
    const response = await api.get('/api/analytics/performance');
    return response.data;
  } catch (error) {
    console.warn('Performance metrics unavailable, returning empty data');
    return { throughput: [], response_times: [], error_rates: [] };
  }
};

export const fetchNLPStats = async () => {
  try {
    const response = await api.get('/api/analytics/nlp-stats');
    return response.data;
  } catch (error) {
    return {
      status: 'loading',
      message: 'NLP service is starting up...',
      gpu_utilization: 0,
      cpu_utilization: 0,
      memory_usage: 0
    };
  }
};

// ROI APIs
export const fetchROIAnalytics = async (periodDays = 30) => {
  try {
    const response = await api.get(`/api/analytics/roi?period_days=${periodDays}`);
    return response.data;
  } catch (error) {
    console.warn('ROI analytics unavailable, returning empty data');
    return { 
      roi_percentage: 0, 
      estimated_revenue_impact: 0, 
      support_cost_savings: 0,
      estimated_churn_reduction: 0,
      sentiment_improvement: 0,
      response_time_reduction: 0,
      critical_issues_prevented: 0
    };
  }
};

// Data Source APIs
export const fetchDataSources = async () => {
  const response = await api.get('/api/ingestion/sources');
  return response.data;
};

export const addDataSource = async (source) => {
  const response = await api.post('/api/ingestion/sources', source);
  return response.data;
};

export const deleteDataSource = async (sourceId) => {
  const response = await api.delete(`/api/ingestion/sources/${sourceId}`);
  return response.data;
};

export const fetchIngestionStats = async () => {
  try {
    const response = await api.get('/api/ingestion/stats');
    return response.data;
  } catch (error) {
    console.warn('Ingestion stats unavailable, returning empty data');
    return { messages_processed: 0, sources_active: 0, processing_rate: 0 };
  }
};

// Message APIs
export const ingestMessage = async (content, metadata = {}) => {
  const response = await api.post('/api/ingestion/messages', { content, metadata });
  return response.data;
};

export const searchMessages = async (query, filters = {}) => {
  const response = await api.post('/api/ingestion/search', { query, filters });
  return response.data;
};

export const getMessageById = async (id) => {
  const response = await api.get(`/api/ingestion/messages/${id}`);
  return response.data;
};

export const getMessageTrajectory = async (id) => {
  const response = await api.get(`/api/analytics/trajectories/${id}`);
  return response.data;
};

// Settings APIs
export const getSettings = async () => {
  try {
    const response = await api.get('/api/ingestion/settings');
    return response.data;
  } catch (error) {
    console.error("Failed to fetch settings, returning defaults", error);
    return {
      general: { siteName: "AI-Powered CRM", defaultLanguage: "en" },
      sentiment: { confidenceThreshold: 0.75, escalationTrigger: -0.5 },
      notifications: { emailEnabled: true, alertLevels: ["critical"] }
    };
  }
};

export const saveSettings = async (settings) => {
  const response = await api.post('/api/ingestion/settings', settings);
  return response.data;
};

export const resetSettings = async () => {
  const response = await api.post('/api/ingestion/settings/reset');
  return response.data;
};

// WebSocket Connection
export const createWebSocketConnection = (onMessage) => {
  const wsUrl = API_BASE_URL.replace(/^http/, 'ws') + '/ws';
  
  const connect = () => {
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connection established');
    };
    
    ws.onmessage = (event) => {
      onMessage(JSON.parse(event.data));
    };
    
    ws.onclose = (event) => {
      console.warn(`WebSocket closed (code: ${event.code}). Reconnecting in 3 seconds...`);
      setTimeout(connect, 3000);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      ws.close();
    };
    
    return ws;
  };
  
  return connect();
};

export default api;
