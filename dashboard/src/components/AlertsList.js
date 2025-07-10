import React from 'react';
import { motion } from 'framer-motion';
import { AlertCircle, AlertTriangle, Info, CheckCircle, Clock } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

const AlertsList = ({ alerts = [] }) => {
  const getPriorityIcon = (priority) => {
    switch (priority) {
      case 'critical':
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      case 'high':
        return <AlertTriangle className="h-5 w-5 text-orange-500" />;
      case 'medium':
        return <Info className="h-5 w-5 text-yellow-500" />;
      default:
        return <CheckCircle className="h-5 w-5 text-green-500" />;
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'critical':
        return 'border-red-500/50 bg-red-900/20';
      case 'high':
        return 'border-orange-500/50 bg-orange-900/20';
      case 'medium':
        return 'border-yellow-500/50 bg-yellow-900/20';
      default:
        return 'border-green-500/50 bg-green-900/20';
    }
  };

  const getPriorityBadge = (priority) => {
    const colors = {
      critical: 'bg-red-500/20 text-red-400 border-red-500/50',
      high: 'bg-orange-500/20 text-orange-400 border-orange-500/50',
      medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50',
      low: 'bg-green-500/20 text-green-400 border-green-500/50'
    };

    return `${colors[priority] || colors.low} border`;
  };

  // Only show real alerts, no mock data
  if (alerts.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <AlertCircle className="h-12 w-12 mx-auto mb-3 opacity-50" />
        <p className="text-sm">No alerts to display</p>
        <p className="text-xs text-gray-600 mt-1">Real alerts will appear here when conditions are met</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {alerts.map((alert, index) => (
        <motion.div
          key={alert.id}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className={`p-4 rounded-lg border ${getPriorityColor(alert.priority)} transition-all hover:shadow-lg`}
        >
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 mt-1">
              {getPriorityIcon(alert.priority)}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 text-xs font-medium rounded-md ${getPriorityBadge(alert.priority)}`}>
                    {alert.priority.toUpperCase()}
                  </span>
                  <span className="text-xs text-gray-500">
                    {(alert.type || alert.rule || '').replace(/_/g, ' ')}
                  </span>
                </div>
                <div className="flex items-center text-xs text-gray-500">
                  <Clock className="h-3 w-3 mr-1" />
                  {formatDistanceToNow(new Date(alert.created_at || alert.timestamp), { addSuffix: true })}
                </div>
              </div>
              
              {/* Show the actual message content that triggered the alert */}
              {alert.message_content && (
                <div className="mb-2">
                  <p className="text-xs text-gray-400 mb-1">Original Message:</p>
                  <p className="text-sm text-gray-200 bg-gray-800/50 rounded px-2 py-1 line-clamp-2">
                    "{alert.message_content}"
                  </p>
                </div>
              )}
              
              {/* Show alert description */}
              <p className="text-sm text-gray-300 mb-2">
                {alert.description || alert.title || 'Alert triggered'}
              </p>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4 text-xs">
                  {(alert.message_sentiment || alert.sentiment) && (
                    <div className="flex items-center">
                      <span className="text-gray-500 mr-1">Sentiment:</span>
                      <span className={`font-medium ${
                        (alert.message_sentiment || alert.sentiment) === 'negative' ? 'text-red-400' :
                        (alert.message_sentiment || alert.sentiment) === 'positive' ? 'text-green-400' : 'text-blue-400'
                      }`}>
                        {alert.message_sentiment || alert.sentiment} ({((alert.sentiment_confidence || 0) * 100).toFixed(0)}%)
                      </span>
                    </div>
                  )}
                  {alert.message_language && (
                    <div className="flex items-center">
                      <span className="text-gray-500 mr-1">Language:</span>
                      <span className="text-gray-300">{alert.message_language.toUpperCase()}</span>
                    </div>
                  )}
                  {alert.metadata?.user && (
                    <div className="flex items-center">
                      <span className="text-gray-500 mr-1">User:</span>
                      <span className="text-gray-300">{alert.metadata.user}</span>
                    </div>
                  )}
                </div>
                
                <div className="flex items-center space-x-2">
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    alert.status === 'active' ? 'bg-red-500/20 text-red-400' :
                    alert.status === 'acknowledged' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-green-500/20 text-green-400'
                  }`}>
                    {alert.status}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
};

export default AlertsList;
