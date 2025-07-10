import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './utils/chartConfig'; // Initialize Chart.js configuration

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
