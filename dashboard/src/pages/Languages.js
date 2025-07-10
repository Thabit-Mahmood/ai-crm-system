import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { Globe, Code, TrendingUp, BarChart3, Languages as LanguageIcon, RefreshCw } from 'lucide-react';
import { Doughnut, Bar, Radar } from 'react-chartjs-2';
import { fetchLanguageAnalytics, detectLanguage } from '../services/api';
import toast from 'react-hot-toast';

const Languages = () => {
  const [testText, setTestText] = useState('');
  const [detecting, setDetecting] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['languages'],
    queryFn: fetchLanguageAnalytics,
    refetchInterval: 30000
  });

  const handleDetectLanguage = async () => {
    if (!testText.trim()) {
      toast.error('Please enter some text to analyze');
      return;
    }

    setDetecting(true);
    try {
      const result = await detectLanguage(testText);
      setDetectionResult(result);
      toast.success('Language detection complete!');
    } catch (error) {
      toast.error('Detection failed');
    } finally {
      setDetecting(false);
    }
  };

  // Language distribution chart
  const languageDistributionData = {
    labels: data?.languages?.slice(0, 10).map(l => l.language.toUpperCase()) || [],
    datasets: [{
      data: data?.languages?.slice(0, 10).map(l => l.message_count) || [],
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(34, 197, 94, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(168, 85, 247, 0.8)',
        'rgba(251, 191, 36, 0.8)',
        'rgba(236, 72, 153, 0.8)',
        'rgba(20, 184, 166, 0.8)',
        'rgba(251, 146, 60, 0.8)',
        'rgba(100, 116, 139, 0.8)',
        'rgba(250, 204, 21, 0.8)'
      ],
      borderWidth: 0
    }]
  };

  // Sentiment by language chart
  const sentimentByLanguageData = {
    labels: data?.languages?.slice(0, 8).map(l => l.language.toUpperCase()) || [],
    datasets: [{
      label: 'Average Sentiment Score',
      data: data?.languages?.slice(0, 8).map(l => l.average_sentiment * 100) || [],
      backgroundColor: data?.languages?.slice(0, 8).map(l => 
        l.average_sentiment > 0.2 ? 'rgba(34, 197, 94, 0.8)' :
        l.average_sentiment < -0.2 ? 'rgba(239, 68, 68, 0.8)' :
        'rgba(59, 130, 246, 0.8)'
      ) || [],
      borderColor: data?.languages?.slice(0, 8).map(l => 
        l.average_sentiment > 0.2 ? 'rgba(34, 197, 94, 1)' :
        l.average_sentiment < -0.2 ? 'rgba(239, 68, 68, 1)' :
        'rgba(59, 130, 246, 1)'
      ) || [],
      borderWidth: 2
    }]
  };

  // Code-switching patterns
  const codeSwitchingData = {
    labels: ['Single Language', 'Code-Switched'],
    datasets: [{
      data: [
        100 - (data?.code_switching_rate || 0),
        data?.code_switching_rate || 0
      ],
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(168, 85, 247, 0.8)'
      ],
      borderColor: [
        'rgba(59, 130, 246, 1)',
        'rgba(168, 85, 247, 1)'
      ],
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

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
        labels: {
          color: '#fff',
          padding: 20
        }
      }
    }
  };

  // Language flags mapping
  const languageFlags = {
    en: '🇬🇧', es: '🇪🇸', fr: '🇫🇷', de: '🇩🇪', zh: '🇨🇳',
    ja: '🇯🇵', ar: '🇸🇦', hi: '🇮🇳', pt: '🇧🇷', ru: '🇷🇺',
    ko: '🇰🇷', it: '🇮🇹', nl: '🇳🇱', tr: '🇹🇷', pl: '🇵🇱'
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Language Analytics</h1>
          <p className="text-gray-400 mt-1">Multilingual insights and code-switching patterns</p>
        </div>
        <button
          onClick={() => refetch()}
          className="p-2 bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors"
        >
          <RefreshCw className="h-5 w-5" />
        </button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-effect rounded-xl p-6 neon-glow"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Total Languages</span>
            <Globe className="h-5 w-5 text-blue-400" />
          </div>
          <p className="text-3xl font-bold">{data?.total_languages || 0}</p>
          <p className="text-sm text-gray-400 mt-1">Active languages</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-effect rounded-xl p-6 shadow-lg shadow-purple-500/20"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Code-Switching</span>
            <Code className="h-5 w-5 text-purple-400" />
          </div>
          <p className="text-3xl font-bold text-purple-400">
            {data?.code_switching_rate?.toFixed(1) || 0}%
          </p>
          <p className="text-sm text-gray-400 mt-1">Mixed language messages</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-effect rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Top Language</span>
            <LanguageIcon className="h-5 w-5 text-green-400" />
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-2xl">
              {languageFlags[data?.languages?.[0]?.language] || '🌐'}
            </span>
            <p className="text-2xl font-bold">
              {data?.languages?.[0]?.language?.toUpperCase() || '-'}
            </p>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            {data?.languages?.[0]?.percentage?.toFixed(1) || 0}% of messages
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-effect rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Message Volume</span>
            <BarChart3 className="h-5 w-5 text-orange-400" />
          </div>
          <p className="text-3xl font-bold">
            {data?.languages?.reduce((sum, l) => sum + l.message_count, 0) || 0}
          </p>
          <p className="text-sm text-gray-400 mt-1">Total analyzed</p>
        </motion.div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Language Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="glass-effect rounded-xl p-6"
        >
          <h2 className="text-xl font-semibold mb-4">Language Distribution</h2>
          <div className="h-64">
            <Doughnut data={languageDistributionData} options={doughnutOptions} />
          </div>
        </motion.div>

        {/* Code-Switching Rate */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="glass-effect rounded-xl p-6"
        >
          <h2 className="text-xl font-semibold mb-4">Code-Switching Analysis</h2>
          <div className="h-64">
            <Doughnut data={codeSwitchingData} options={doughnutOptions} />
          </div>
          <div className="mt-4 grid grid-cols-2 gap-4 text-center">
            <div className="bg-gray-700/30 rounded-lg p-3">
              <p className="text-sm text-gray-400">Mixed Messages</p>
              <p className="text-2xl font-bold text-purple-400">
                {Math.round((data?.code_switching_rate || 0) * 10)}
              </p>
            </div>
            <div className="bg-gray-700/30 rounded-lg p-3">
              <p className="text-sm text-gray-400">Language Pairs</p>
              <p className="text-2xl font-bold text-blue-400">
                {Math.min(data?.total_languages || 0, 5)}
              </p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Sentiment by Language */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="glass-effect rounded-xl p-6"
      >
        <h2 className="text-xl font-semibold mb-4">Sentiment by Language</h2>
        <div className="h-64">
          <Bar data={sentimentByLanguageData} options={chartOptions} />
        </div>
      </motion.div>

      {/* Language Details Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="glass-effect rounded-xl p-6"
      >
        <h2 className="text-xl font-semibold mb-4">Language Details</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Language</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-gray-400">Messages</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-gray-400">Percentage</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-gray-400">Avg Sentiment</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-gray-400">Code-Switched</th>
              </tr>
            </thead>
            <tbody>
              {data?.languages?.map((lang, index) => (
                <tr key={lang.language} className="border-b border-gray-700/50">
                  <td className="py-3 px-4">
                    <div className="flex items-center space-x-2">
                      <span className="text-xl">{languageFlags[lang.language] || '🌐'}</span>
                      <span className="font-medium">{lang.language.toUpperCase()}</span>
                    </div>
                  </td>
                  <td className="text-right py-3 px-4">{lang.message_count}</td>
                  <td className="text-right py-3 px-4">{lang.percentage.toFixed(1)}%</td>
                  <td className="text-right py-3 px-4">
                    <span className={`font-medium ${
                      lang.average_sentiment > 0.2 ? 'text-green-400' :
                      lang.average_sentiment < -0.2 ? 'text-red-400' : 'text-blue-400'
                    }`}>
                      {lang.average_sentiment > 0 ? '+' : ''}{(lang.average_sentiment * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="text-right py-3 px-4">
                    {lang.code_switched_messages || 0}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Language Detection Test */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="glass-effect rounded-xl p-6"
      >
        <h2 className="text-xl font-semibold mb-4">Test Language Detection</h2>
        <div className="space-y-4">
          <textarea
            value={testText}
            onChange={(e) => setTestText(e.target.value)}
            placeholder="Enter text to detect language and code-switching... Try mixing languages like: Hello! ¿Cómo estás? 你好吗？"
            className="w-full h-32 bg-gray-700/50 border border-gray-600 rounded-lg p-4 text-white placeholder-gray-400 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleDetectLanguage}
            disabled={detecting}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-medium hover:opacity-90 transition-opacity disabled:opacity-50"
          >
            {detecting ? 'Detecting...' : 'Detect Language'}
          </button>
          
          {detectionResult && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 p-4 bg-gray-700/50 rounded-lg"
            >
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-400">Primary Language</p>
                  <div className="flex items-center space-x-2 mt-1">
                    <span className="text-2xl">
                      {languageFlags[detectionResult.primary_language] || '🌐'}
                    </span>
                    <p className="text-2xl font-bold">
                      {detectionResult.primary_language.toUpperCase()}
                    </p>
                  </div>
                </div>
                <div>
                  <p className="text-sm text-gray-400">Confidence</p>
                  <p className="text-2xl font-bold mt-1">
                    {(detectionResult.confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">Code-Switching</p>
                  <p className={`text-2xl font-bold mt-1 ${
                    detectionResult.is_code_switched ? 'text-purple-400' : 'text-gray-400'
                  }`}>
                    {detectionResult.is_code_switched ? 'Yes' : 'No'}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">Processing Time</p>
                  <p className="text-2xl font-bold mt-1">
                    {detectionResult.processing_time_ms.toFixed(0)}ms
                  </p>
                </div>
              </div>
              
              {detectionResult.all_languages.length > 1 && (
                <div className="mt-4">
                  <p className="text-sm text-gray-400 mb-2">All Detected Languages:</p>
                  <div className="space-y-2">
                    {detectionResult.all_languages.map((lang, idx) => (
                      <div key={idx} className="flex items-center justify-between bg-gray-800/50 rounded p-2">
                        <div className="flex items-center space-x-2">
                          <span>{languageFlags[lang.language] || '🌐'}</span>
                          <span className="font-medium">{lang.language.toUpperCase()}</span>
                          <span className="text-xs text-gray-500">({lang.method})</span>
                        </div>
                        <span className="text-sm">
                          {(lang.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {detectionResult.segments && (
                <div className="mt-4">
                  <p className="text-sm text-gray-400 mb-2">Language Segments:</p>
                  <div className="space-y-2">
                    {detectionResult.segments.map((segment, idx) => (
                      <div key={idx} className="bg-gray-800/50 rounded p-2">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-gray-500">
                            {segment.script} script
                          </span>
                          <span className="text-xs">
                            {languageFlags[segment.language] || '🌐'} {segment.language?.toUpperCase() || 'Unknown'}
                          </span>
                        </div>
                        <p className="text-sm">{segment.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </div>
      </motion.div>
    </div>
  );
};

export default Languages;
