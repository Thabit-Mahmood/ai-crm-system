import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Database, 
  Plus, 
  Trash2, 
  CheckCircle, 
  XCircle,
  Twitter,
  Facebook,
  Mail,
  MessageSquare,
  Globe,
  RefreshCw,
  Settings,
  Activity
} from 'lucide-react';
import { fetchDataSources, addDataSource, deleteDataSource, fetchIngestionStats, ingestMessage } from '../services/api';
import toast from 'react-hot-toast';

const DataSources = () => {
  const [showAddModal, setShowAddModal] = useState(false);
  const [showTestModal, setShowTestModal] = useState(false);
  const [testMessage, setTestMessage] = useState({
    content: '',
    language: 'auto',
    source: 'manual'
  });
  const [formData, setFormData] = useState({
    name: '',
    type: 'twitter',
    config: {
      pollInterval: 60000,
      apiKey: '',
      apiSecret: ''
    }
  });

  const queryClient = useQueryClient();

  const { data: sources, isLoading: sourcesLoading } = useQuery({
    queryKey: ['data-sources'],
    queryFn: fetchDataSources,
    refetchInterval: 10000
  });

  const { data: stats } = useQuery({
    queryKey: ['ingestion-stats'],
    queryFn: fetchIngestionStats,
    refetchInterval: 5000
  });

  const addSourceMutation = useMutation({
    mutationFn: addDataSource,
    onSuccess: () => {
      queryClient.invalidateQueries(['data-sources']);
      queryClient.invalidateQueries(['ingestion-stats']);
      toast.success('Data source added successfully');
      setShowAddModal(false);
      resetForm();
    },
    onError: () => {
      toast.error('Failed to add data source');
    }
  });

  const deleteSourceMutation = useMutation({
    mutationFn: deleteDataSource,
    onSuccess: () => {
      queryClient.invalidateQueries(['data-sources']);
      queryClient.invalidateQueries(['ingestion-stats']);
      toast.success('Data source removed');
    },
    onError: () => {
      toast.error('Failed to remove data source');
    }
  });

  const testMessageMutation = useMutation({
    mutationFn: ingestMessage,
    onSuccess: () => {
      queryClient.invalidateQueries(['ingestion-stats']);
      toast.success('Test message processed successfully!');
      setShowTestModal(false);
      setTestMessage({ content: '', language: 'auto', source: 'manual' });
    },
    onError: (error) => {
      toast.error(`Failed to process message: ${error.message}`);
    }
  });

  const resetForm = () => {
    setFormData({
      name: '',
      type: 'twitter',
      config: {
        pollInterval: 60000,
        apiKey: '',
        apiSecret: ''
      }
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!formData.name) {
      toast.error('Please provide a name for the data source');
      return;
    }
    addSourceMutation.mutate(formData);
  };

  const handleTestMessage = (e) => {
    e.preventDefault();
    if (!testMessage.content.trim()) {
      toast.error('Please enter a message to test');
      return;
    }
    testMessageMutation.mutate(testMessage.content, {
      source: testMessage.source,
      language: testMessage.language,
      timestamp: new Date().toISOString()
    });
  };

  const getSourceIcon = (type) => {
    const icons = {
      twitter: <Twitter className="h-5 w-5 text-blue-400" />,
      facebook: <Facebook className="h-5 w-5 text-blue-600" />,
      email: <Mail className="h-5 w-5 text-gray-400" />,
      chat: <MessageSquare className="h-5 w-5 text-green-400" />,
      api: <Globe className="h-5 w-5 text-purple-400" />
    };
    return icons[type] || <Database className="h-5 w-5 text-gray-400" />;
  };

  const getStatusColor = (status) => {
    return status === 'active' ? 'text-green-400' : 'text-gray-500';
  };

  const formatInterval = (ms) => {
    const seconds = ms / 1000;
    if (seconds < 60) return `${seconds}s`;
    const minutes = seconds / 60;
    if (minutes < 60) return `${minutes}m`;
    const hours = minutes / 60;
    return `${hours.toFixed(1)}h`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Data Sources</h1>
          <p className="text-gray-400 mt-1">Manage your data ingestion connections</p>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={() => setShowTestModal(true)}
            className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
          >
            <MessageSquare className="h-5 w-5" />
            <span>Test Message</span>
          </button>
        <button
          onClick={() => setShowAddModal(true)}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
        >
          <Plus className="h-5 w-5" />
          <span>Add Source</span>
        </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-effect rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Total Messages</span>
            <Database className="h-5 w-5 text-blue-400" />
          </div>
          <p className="text-3xl font-bold">{stats?.messages?.total_messages || 0}</p>
          <div className="mt-2 text-sm">
            <span className="text-green-400">{stats?.messages?.last_hour || 0}</span>
            <span className="text-gray-400 ml-2">last hour</span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-effect rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Active Sources</span>
            <Activity className="h-5 w-5 text-green-400" />
          </div>
          <p className="text-3xl font-bold text-green-400">
            {sources?.filter(s => s.status === 'active').length || 0}
          </p>
          <p className="text-sm text-gray-400 mt-1">
            of {sources?.length || 0} total
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-effect rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Processing Rate</span>
            <RefreshCw className="h-5 w-5 text-purple-400" />
          </div>
          <p className="text-3xl font-bold">
            {((stats?.messages?.last_hour || 0) / 60).toFixed(1)}
            <span className="text-lg font-normal text-gray-400">/min</span>
          </p>
          <p className="text-sm text-gray-400 mt-1">Messages ingested</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-effect rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Processed</span>
            <CheckCircle className="h-5 w-5 text-blue-400" />
          </div>
          <p className="text-3xl font-bold">
            {stats?.messages?.processed_messages || 0}
          </p>
          <p className="text-sm text-gray-400 mt-1">
            {((stats?.messages?.processed_messages / stats?.messages?.total_messages) * 100).toFixed(0) || 0}% completion
          </p>
        </motion.div>
      </div>

      {/* Source Type Distribution */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="glass-effect rounded-xl p-6"
      >
        <h2 className="text-xl font-semibold mb-4">Source Distribution</h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {stats?.sources?.map((source) => (
            <div key={source.type} className="text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-lg bg-gray-700/50 mb-2">
                {getSourceIcon(source.type)}
              </div>
              <p className="text-sm font-medium capitalize">{source.type}</p>
              <p className="text-2xl font-bold mt-1">{source.count}</p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Data Sources List */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="glass-effect rounded-xl p-6"
      >
        <h2 className="text-xl font-semibold mb-4">Configured Sources</h2>
        {sourcesLoading ? (
          <div className="flex justify-center py-12">
            <div className="spinner"></div>
          </div>
        ) : sources?.length === 0 ? (
          <div className="text-center py-12">
            <Database className="h-12 w-12 text-gray-500 mx-auto mb-4" />
            <p className="text-gray-400">No data sources configured</p>
            <button
              onClick={() => setShowAddModal(true)}
              className="mt-4 text-blue-400 hover:text-blue-300"
            >
              Add your first data source →
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            {sources.map((source, index) => (
              <motion.div
                key={source.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className="flex items-center justify-between p-4 bg-gray-700/30 rounded-lg"
              >
                <div className="flex items-center space-x-4">
                  <div className="flex-shrink-0">
                    {getSourceIcon(source.type)}
                  </div>
                  <div>
                    <h3 className="font-medium">{source.name}</h3>
                    <div className="flex items-center space-x-4 text-sm text-gray-400 mt-1">
                      <span className="flex items-center">
                        <span className={`w-2 h-2 rounded-full mr-2 ${
                          source.status === 'active' ? 'bg-green-400' : 'bg-gray-500'
                        }`}></span>
                        <span className={getStatusColor(source.status)}>
                          {source.status}
                        </span>
                      </span>
                      <span>Poll: {formatInterval(source.config?.pollInterval || 60000)}</span>
                      <span>Added: {new Date(source.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
                    title="Configure"
                  >
                    <Settings className="h-4 w-4" />
                  </button>
                  <button
                    onClick={() => {
                      if (window.confirm('Are you sure you want to remove this data source?')) {
                        deleteSourceMutation.mutate(source.id);
                      }
                    }}
                    className="p-2 text-gray-400 hover:text-red-400 hover:bg-gray-700 rounded-lg transition-colors"
                    title="Remove"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </motion.div>

      {/* Add Source Modal */}
      <AnimatePresence>
        {showAddModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
            onClick={() => setShowAddModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="glass-effect rounded-xl p-6 max-w-md w-full mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <h2 className="text-xl font-semibold mb-4">Add Data Source</h2>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Source Name
                  </label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., Main Twitter Feed"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Source Type
                  </label>
                  <select
                    value={formData.type}
                    onChange={(e) => setFormData({ ...formData, type: e.target.value })}
                    className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="twitter">Twitter</option>
                    <option value="facebook">Facebook</option>
                    <option value="email">Email</option>
                    <option value="chat">Live Chat</option>
                    <option value="api">Custom API</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Poll Interval
                  </label>
                  <select
                    value={formData.config.pollInterval}
                    onChange={(e) => setFormData({
                      ...formData,
                      config: { ...formData.config, pollInterval: Number(e.target.value) }
                    })}
                    className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value={10000}>10 seconds</option>
                    <option value={30000}>30 seconds</option>
                    <option value={60000}>1 minute</option>
                    <option value={300000}>5 minutes</option>
                    <option value={600000}>10 minutes</option>
                  </select>
                </div>

                {formData.type !== 'twitter' && (
                  <div className="p-3 bg-yellow-900/20 border border-yellow-700/50 rounded-lg">
                    <p className="text-sm text-yellow-400">
                      Note: Additional data source integrations require API configuration. Only manual message ingestion is fully functional without external API keys.
                    </p>
                  </div>
                )}

                <div className="flex justify-end space-x-3 pt-4">
                  <button
                    type="button"
                    onClick={() => {
                      setShowAddModal(false);
                      resetForm();
                    }}
                    className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={addSourceMutation.isLoading}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50"
                  >
                    {addSourceMutation.isLoading ? 'Adding...' : 'Add Source'}
                  </button>
                </div>
              </form>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Test Message Modal */}
      <AnimatePresence>
        {showTestModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
            onClick={() => setShowTestModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="glass-effect rounded-xl p-6 max-w-lg w-full mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <h2 className="text-xl font-semibold mb-4">Test Message Processing</h2>
              <p className="text-gray-400 text-sm mb-4">
                Send a test message through the AI CRM pipeline to see sentiment analysis, language detection, and alert processing in action.
              </p>
              
              <form onSubmit={handleTestMessage} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Message Content
                  </label>
                  <textarea
                    value={testMessage.content}
                    onChange={(e) => setTestMessage({ ...testMessage, content: e.target.value })}
                    className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500 resize-none"
                    placeholder="Enter a message to test (e.g., 'I love this new feature!' or 'This is terrible and broken!')"
                    rows={4}
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Language
                    </label>
                    <select
                      value={testMessage.language}
                      onChange={(e) => setTestMessage({ ...testMessage, language: e.target.value })}
                      className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-green-500"
                    >
                      <option value="auto">Auto-detect</option>
                      <option value="en">English</option>
                      <option value="es">Spanish</option>
                      <option value="fr">French</option>
                      <option value="de">German</option>
                      <option value="pt">Portuguese</option>
                      <option value="it">Italian</option>
                      <option value="zh">Chinese</option>
                      <option value="ja">Japanese</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Source Type
                    </label>
                    <select
                      value={testMessage.source}
                      onChange={(e) => setTestMessage({ ...testMessage, source: e.target.value })}
                      className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-green-500"
                    >
                      <option value="manual">Manual Test</option>
                      <option value="twitter">Twitter</option>
                      <option value="facebook">Facebook</option>
                      <option value="email">Email</option>
                      <option value="chat">Live Chat</option>
                    </select>
                  </div>
                </div>

                <div className="p-3 bg-blue-900/20 border border-blue-700/50 rounded-lg">
                  <p className="text-sm text-blue-400">
                    <strong>This will:</strong> Process your message through language detection → sentiment analysis → alert evaluation → dashboard updates
                  </p>
                </div>

                <div className="flex justify-end space-x-3 pt-4">
                  <button
                    type="button"
                    onClick={() => {
                      setShowTestModal(false);
                      setTestMessage({ content: '', language: 'auto', source: 'manual' });
                    }}
                    className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={testMessageMutation.isLoading}
                    className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors disabled:opacity-50"
                  >
                    {testMessageMutation.isLoading ? 'Processing...' : 'Send Test Message'}
                  </button>
                </div>
              </form>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default DataSources;
