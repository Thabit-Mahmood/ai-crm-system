import React, { useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import { format } from 'date-fns';

const RealtimeChart = ({ data }) => {
  const chartRef = useRef(null);

  // Process data for the chart - SAFE ARRAY CHECK
  const processedData = Array.isArray(data) ? data.slice(-24) : []; // Last 24 data points

  const chartData = {
    labels: processedData.map(d => {
      const date = new Date(d.timestamp);
      return format(date, 'HH:mm');
    }),
    datasets: [
      {
        label: 'Sentiment Score',
        data: processedData.map(d => d.sentiment_score),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 6,
        pointBackgroundColor: 'rgb(59, 130, 246)',
        pointBorderColor: '#fff',
        pointBorderWidth: 2
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: 'rgba(31, 41, 55, 0.95)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: 'rgba(75, 85, 99, 0.3)',
        borderWidth: 1,
        padding: 12,
        displayColors: false,
        callbacks: {
          label: (context) => {
            const value = context.parsed.y;
            const sentiment = value > 0 ? 'Positive' : value < 0 ? 'Negative' : 'Neutral';
            return `Score: ${value.toFixed(2)} (${sentiment})`;
          }
        }
      }
    },
    scales: {
      x: {
        ticks: { 
          color: '#9ca3af',
          maxRotation: 0,
          autoSkip: true,
          maxTicksLimit: 8
        },
        grid: { 
          color: 'rgba(75, 85, 99, 0.3)',
          drawBorder: false
        }
      },
      y: {
        min: -100,
        max: 100,
        ticks: { 
          color: '#9ca3af',
          callback: (value) => value > 0 ? `+${value}` : value
        },
        grid: { 
          color: 'rgba(75, 85, 99, 0.3)',
          drawBorder: false
        }
      }
    },
    animation: {
      duration: 750,
      easing: 'easeInOutQuart'
    }
  };

  // Add gradient - AFTER data check
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const ctx = chart.ctx;
    const gradient = ctx.createLinearGradient(0, 0, 0, chart.height);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
    
    chart.data.datasets[0].backgroundColor = gradient;
    chart.update();
  }, []);

  // Only render chart if we have valid data
  if (!processedData || processedData.length === 0) {
    return (
      <div className="relative h-full flex items-center justify-center text-gray-400">
        <div className="text-center">
          <div className="text-sm">No chart data available</div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-full">
      <Line ref={chartRef} data={chartData} options={options} />
      
      {/* Zero line indicator */}
      <div className="absolute inset-0 pointer-events-none">
        <div 
          className="absolute w-full border-t border-gray-600 border-dashed"
          style={{ top: '50%' }}
        >
          <span className="absolute -top-3 left-2 text-xs text-gray-500 bg-gray-800 px-1">
            Neutral
          </span>
        </div>
      </div>
    </div>
  );
};

export default RealtimeChart;
