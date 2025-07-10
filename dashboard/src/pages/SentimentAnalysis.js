import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { Brain, TrendingUp, TrendingDown, BarChart3, MessageSquare, RefreshCw, Globe } from 'lucide-react';
import { Line, Bar, Radar } from 'react-chartjs-2';
import { fetchSentimentAnalytics, analyzeSentiment } from '../services/api';
import toast from 'react-hot-toast';

const SentimentAnalysis = () => {
  const [timeRange, setTimeRange] = useState('24h');
  const [testText, setTestText] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [testResult, setTestResult] = useState(null);

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['sentiment', timeRange],
    queryFn: () => fetchSentimentAnalytics(timeRange)
  });

  const handleAnalyzeText = async () => {
    if (!testText.trim()) {
      toast.error('Please enter some text to analyze');
      return;
    }

    setAnalyzing(true);
    try {
      const result = await analyzeSentiment(testText);
      setTestResult(result);
      toast.success('Analysis complete!');
      // Refresh the dashboard data to show the new sentiment in graphs
      setTimeout(() => {
        refetch();
      }, 500); // Wait 0.5 seconds for data to propagate
      // Also refresh after a bit more time in case processing takes longer
      setTimeout(() => {
        refetch();
      }, 2000);
    } catch (error) {
      toast.error('Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  // Generate sentiment trend data with proper values - ONLY REAL DATA
  const generateSentimentTrendData = () => {
    // Check if we have real meaningful data (not just zeros)
    if (data?.sentiment_trend && 
        data.sentiment_trend.length > 0 && 
        data.sentiment_trend.some(d => d.sentiment_score !== 0)) {
      // If we have real meaningful data, use it
      return {
        labels: data.sentiment_trend.map(d => {
          const date = new Date(d.timestamp);
          return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }),
        datasets: [{
          label: 'Sentiment Score',
          data: data.sentiment_trend.map(d => d.sentiment_score),
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4
        }]
      };
    }
    
    // NO FAKE DATA - return empty dataset when no meaningful data available
    return {
      labels: [],
      datasets: [{
        label: 'Sentiment Score',
        data: [],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4
      }]
    };
  };

  const sentimentTrendData = generateSentimentTrendData();

  const languageSentimentData = {
    labels: Object.keys(data?.language_distribution || {}),
    datasets: [{
      label: 'Message Count',
      data: Object.values(data?.language_distribution || {}),
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(34, 197, 94, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(168, 85, 247, 0.8)',
        'rgba(251, 191, 36, 0.8)'
      ],
      borderWidth: 0
    }]
  };

  // Calculate confidence distribution from actual data
  const calculateConfidenceDistribution = () => {
    if (!data?.confidence_distribution) {
      // If no data, calculate from sentiment_trend or return empty
      return [0, 0, 0, 0, 0];
    }
    
    // If confidence_distribution exists in data, use it
    return data.confidence_distribution;
  };

  const confidenceData = {
    labels: ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
    datasets: [{
      label: 'Confidence Distribution',
      data: calculateConfidenceDistribution(),
      backgroundColor: 'rgba(168, 85, 247, 0.5)',
      borderColor: 'rgba(168, 85, 247, 1)',
      borderWidth: 2
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#fff' }
      }
    },
    scales: {
      x: {
        ticks: { color: '#9ca3af' },
        grid: { color: 'rgba(75, 85, 99, 0.3)' }
      },
      y: {
        ticks: { color: '#9ca3af' },
        grid: { color: 'rgba(75, 85, 99, 0.3)' }
      }
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Sentiment Analysis</h1>
          <p className="text-gray-400 mt-1">Deep insights into customer emotions</p>
        </div>
        <div className="flex items-center space-x-4">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-sm text-white"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
          <button
            onClick={() => refetch()}
            className="p-2 bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors"
          >
            <RefreshCw className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-effect rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Total Messages</span>
            <MessageSquare className="h-5 w-5 text-blue-400" />
          </div>
          <p className="text-3xl font-bold">{data?.total_messages || 0}</p>
          <p className="text-sm text-gray-400 mt-1">Analyzed today</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-effect rounded-xl p-6 success-glow"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Positive</span>
            <TrendingUp className="h-5 w-5 text-green-400" />
          </div>
          <p className="text-3xl font-bold text-green-400">
            {data?.sentiment_percentages?.positive?.toFixed(1) || 0}%
          </p>
          <p className="text-sm text-gray-400 mt-1">
            {data?.sentiment_distribution?.positive || 0} messages
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-effect rounded-xl p-6 danger-glow"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Negative</span>
            <TrendingDown className="h-5 w-5 text-red-400" />
          </div>
          <p className="text-3xl font-bold text-red-400">
            {data?.sentiment_percentages?.negative?.toFixed(1) || 0}%
          </p>
          <p className="text-sm text-gray-400 mt-1">
            {data?.sentiment_distribution?.negative || 0} messages
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-effect rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Avg Confidence</span>
            <Brain className="h-5 w-5 text-purple-400" />
          </div>
          <p className="text-3xl font-bold text-purple-400">
            {(data?.average_confidence * 100)?.toFixed(1) || 0}%
          </p>
          <p className="text-sm text-gray-400 mt-1">Model certainty</p>
        </motion.div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sentiment Trend */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="glass-effect rounded-xl p-6"
        >
          <h2 className="text-xl font-semibold mb-4">Sentiment Trend</h2>
          <div className="h-64">
            {data?.sentiment_trend && 
             data.sentiment_trend.length > 0 && 
             data.sentiment_trend.some(d => d.sentiment_score !== 0) ? (
            <Line data={sentimentTrendData} options={chartOptions} />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-400">
                <div className="text-center">
                  <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No trend data available</p>
                  <p className="text-sm mt-2">Process some messages to see sentiment trends</p>
                </div>
              </div>
            )}
          </div>
        </motion.div>

        {/* Language Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="glass-effect rounded-xl p-6"
        >
          <h2 className="text-xl font-semibold mb-4">Language Distribution</h2>
          <div className="h-64">
            {data?.language_distribution && Object.keys(data.language_distribution).length > 0 ? (
            <Bar data={languageSentimentData} options={chartOptions} />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-400">
                <div className="text-center">
                  <Globe className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No language data available</p>
                  <p className="text-sm mt-2">Process some messages to see language distribution</p>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Confidence Distribution */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="glass-effect rounded-xl p-6"
      >
        <h2 className="text-xl font-semibold mb-4">Confidence Distribution</h2>
        <div className="h-64">
          {data?.confidence_distribution && data.confidence_distribution.some(val => val > 0) ? (
          <Bar data={confidenceData} options={chartOptions} />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-400">
              <div className="text-center">
                <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No confidence data available</p>
                <p className="text-sm mt-2">Process some messages to see confidence distribution</p>
              </div>
            </div>
          )}
        </div>
      </motion.div>

      {/* Test Sentiment Analysis */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="glass-effect rounded-xl p-6"
      >
        <h2 className="text-xl font-semibold mb-4">Test Sentiment Analysis</h2>
        <div className="space-y-4">
          <textarea
            value={testText}
            onChange={(e) => setTestText(e.target.value)}
            placeholder="Enter text to analyze sentiment..."
            className="w-full h-32 bg-gray-700/50 border border-gray-600 rounded-lg p-4 text-white placeholder-gray-400 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleAnalyzeText}
            disabled={analyzing}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-medium hover:opacity-90 transition-opacity disabled:opacity-50"
          >
            {analyzing ? 'Analyzing...' : 'Analyze Sentiment'}
          </button>
          
          {testResult && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 p-4 bg-gray-700/50 rounded-lg"
            >
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-400">Sentiment</p>
                  <p className={`text-2xl font-bold ${
                    testResult.sentiment === 'positive' ? 'text-green-400' :
                    testResult.sentiment === 'negative' ? 'text-red-400' : 'text-blue-400'
                  }`}>
                    {testResult.sentiment}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">Confidence</p>
                  <p className="text-2xl font-bold">
                    {(testResult.confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">Processing Time</p>
                  <p className="text-2xl font-bold">
                    {testResult.processing_time_ms.toFixed(0)}ms
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">Model Used</p>
                  <p className="text-sm font-medium">
                    {testResult.model_used}
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </motion.div>
    </div>
  );
};

export default SentimentAnalysis;
