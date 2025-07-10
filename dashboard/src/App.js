import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import Layout from './components/Layout';

import SentimentAnalysis from './pages/SentimentAnalysis';
import Performance from './pages/Performance';

import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchInterval: 5000, // Default, will be updated by settings
      retry: 2,
      staleTime: 2000,
    },
  },
});

// Make queryClient available globally for settings to modify
window.queryClient = queryClient;

function App() {
  useEffect(() => {
    // Load and apply settings on app startup
    const loadSystemSettings = async () => {
      try {
        const savedSettings = localStorage.getItem('aiCrmSettings');
        if (savedSettings) {
          const settings = JSON.parse(savedSettings);
          
          // Apply theme
          document.documentElement.setAttribute('data-theme', settings.system?.theme || 'dark');
          
          // Apply refresh interval
          if (settings.dashboard?.autoRefresh) {
            queryClient.setDefaultOptions({
              queries: {
                refetchInterval: settings.system?.refreshInterval || 5000,
                retry: settings.system?.maxRetries || 2,
                staleTime: 2000,
              }
            });
          } else {
            queryClient.setDefaultOptions({
              queries: {
                refetchInterval: false,
                retry: settings.system?.maxRetries || 2,
                staleTime: 2000,
              }
            });
          }
          
          // Store settings globally
          window.systemSettings = settings;
        } else {
          // Set default dark theme
          document.documentElement.setAttribute('data-theme', 'dark');
        }
      } catch (error) {
        console.error('Failed to load system settings:', error);
        document.documentElement.setAttribute('data-theme', 'dark');
      }
    };

    loadSystemSettings();
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
          <div className="absolute inset-0 bg-gradient-mesh opacity-10"></div>
          <div className="relative z-10">
            <Layout>
              <Routes>
                <Route path="/" element={<Navigate to="/sentiment" replace />} />
                <Route path="/sentiment" element={<SentimentAnalysis />} />
                <Route path="/performance" element={<Performance />} />
              </Routes>
            </Layout>
          </div>
          <Toaster 
            position="top-right"
            toastOptions={{
              className: '',
              style: {
                border: '1px solid #374151',
                padding: '16px',
                color: '#fff',
                background: '#1f2937',
                zIndex: 10000,
              },
              success: {
                iconTheme: {
                  primary: '#10b981',
                  secondary: '#fff',
                },
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#fff',
                },
              },
            }}
          />
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
