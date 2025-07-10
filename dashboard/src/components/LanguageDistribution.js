import React from 'react';
import { motion } from 'framer-motion';
import { Globe, Code } from 'lucide-react';

const LanguageDistribution = ({ data }) => {
  // Language flag emojis mapping
  const languageFlags = {
    en: '🇬🇧',
    es: '🇪🇸',
    fr: '🇫🇷',
    de: '🇩🇪',
    zh: '🇨🇳',
    ja: '🇯🇵',
    ar: '🇸🇦',
    hi: '🇮🇳',
    pt: '🇧🇷',
    ru: '🇷🇺',
    ko: '🇰🇷',
    it: '🇮🇹',
    nl: '🇳🇱',
    tr: '🇹🇷',
    pl: '🇵🇱'
  };

  // ONLY REAL DATA - no fake fallback data
  const languages = data?.languages || [];

  const getSentimentColor = (sentiment) => {
    if (sentiment > 0.2) return 'text-green-400';
    if (sentiment < -0.2) return 'text-red-400';
    return 'text-blue-400';
  };

  const getSentimentBg = (sentiment) => {
    if (sentiment > 0.2) return 'bg-green-500';
    if (sentiment < -0.2) return 'bg-red-500';
    return 'bg-blue-500';
  };

  // Show "No data available" when no real data exists
  if (!languages || languages.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-gray-400">
        <div className="text-center">
          <Globe className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No language data available</p>
          <p className="text-sm mt-2">Process some messages to see language distribution</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Code-switching indicator */}
      <div className="mb-4 p-3 bg-purple-900/20 border border-purple-500/50 rounded-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Code className="h-5 w-5 text-purple-400" />
            <span className="text-sm font-medium text-purple-400">Code-Switching Rate</span>
          </div>
          <span className="text-lg font-bold text-purple-400">
            {data?.code_switching_rate?.toFixed(1) || 0}%
          </span>
        </div>
      </div>

      {/* Language list */}
      <div className="space-y-3 flex-1">
        {languages.slice(0, 5).map((lang, index) => (
          <motion.div
            key={lang.language}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className="relative"
          >
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center space-x-2">
                <span className="text-xl">{languageFlags[lang.language] || '🌐'}</span>
                <span className="text-sm font-medium uppercase">{lang.language}</span>
                <span className="text-xs text-gray-500">
                  {lang.message_count} messages
                </span>
              </div>
              <span className={`text-sm font-medium ${getSentimentColor(lang.average_sentiment)}`}>
                {lang.average_sentiment > 0 ? '+' : ''}{(lang.average_sentiment * 100).toFixed(0)}%
              </span>
            </div>
            
            {/* Progress bar */}
            <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${lang.percentage}%` }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                className={`h-full rounded-full ${getSentimentBg(lang.average_sentiment)} opacity-80`}
              />
            </div>
            
            <div className="flex justify-between mt-1">
              <span className="text-xs text-gray-500">{lang.percentage}%</span>
              {lang.code_switched_messages > 0 && (
                <span className="text-xs text-purple-400">
                  {lang.code_switched_messages} mixed
                </span>
              )}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Summary stats */}
      <div className="mt-4 pt-4 border-t border-gray-700 grid grid-cols-2 gap-4 text-center">
        <div>
          <div className="flex items-center justify-center space-x-1 text-gray-400 text-xs mb-1">
            <Globe className="h-3 w-3" />
            <span>Total Languages</span>
          </div>
          <span className="text-2xl font-bold text-blue-400">
            {data?.total_languages || 0}
          </span>
        </div>
        <div>
          <div className="flex items-center justify-center space-x-1 text-gray-400 text-xs mb-1">
            <Code className="h-3 w-3" />
            <span>Mixed Messages</span>
          </div>
          <span className="text-2xl font-bold text-purple-400">
            {data?.code_switched_messages || 0}
          </span>
        </div>
      </div>
    </div>
  );
};

export default LanguageDistribution;
