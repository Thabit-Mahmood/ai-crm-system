import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  Zap,
  Menu,
  X,
  Activity,
  Search
} from 'lucide-react';

const Layout = ({ children }) => {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const navigation = [
    { name: 'Sentiment Analysis', href: '/sentiment', icon: Brain },
    { name: 'Performance', href: '/performance', icon: Zap },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <div className="flex h-screen bg-gray-900">
      {/* Sidebar */}
      <AnimatePresence mode="wait">
        {sidebarOpen && (
          <motion.div
            initial={{ x: -300 }}
            animate={{ x: 0 }}
            exit={{ x: -300 }}
            transition={{ duration: 0.3 }}
            className="fixed inset-y-0 left-0 z-50 w-64 bg-gray-800/95 backdrop-blur-xl border-r border-gray-700 lg:relative lg:z-0"
          >
            <div className="flex flex-col h-full">
              <div className="flex items-center justify-between p-6 border-b border-gray-700">
                <h1 className="text-xl font-bold gradient-text">AI CRM</h1>
                <button
                  onClick={() => setSidebarOpen(false)}
                  className="lg:hidden p-2 rounded-lg hover:bg-gray-700"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              
              <nav className="flex-1 px-4 py-6 space-y-2">
                {navigation.map((item) => {
                  const Icon = item.icon;
                  return (
                    <Link
                      key={item.name}
                      to={item.href}
                      className={`flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                        isActive(item.href)
                          ? 'bg-blue-600 text-white shadow-lg'
                          : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                      }`}
                    >
                      <Icon className="mr-3 h-5 w-5" />
                      {item.name}
                    </Link>
                  );
                })}
              </nav>
              
              <div className="p-4 border-t border-gray-700">
                <div className="flex items-center space-x-3 text-sm text-gray-400">
                  <div className="live-indicator">
                    <div className="relative">
                      <div className="absolute -inset-1 bg-green-500 rounded-full opacity-75 animate-ping"></div>
                      <div className="relative w-2 h-2 bg-green-500 rounded-full"></div>
                    </div>
                  </div>
                  <span>System Online</span>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-gray-800/50 backdrop-blur-xl border-b border-gray-700">
          <div className="flex items-center justify-between px-6 py-4">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="p-2 rounded-lg hover:bg-gray-700 transition-colors"
              >
                <Menu className="h-5 w-5" />
              </button>
              
              <div className="hidden md:flex items-center space-x-2 text-sm text-gray-400">
                <Search className="h-4 w-4" />
                <span>Search...</span>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Activity indicator */}
              <div className="flex items-center space-x-2 text-sm">
                <Activity className="h-4 w-4 text-green-400 animate-pulse" />
                <span className="text-gray-300">Live Processing</span>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto">
          <div className="p-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;
