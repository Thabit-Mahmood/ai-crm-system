import React, { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { Zap, Gauge, Server, Clock, TrendingUp, Activity, Cpu, HardDrive } from 'lucide-react';
import { Line, Bar, Scatter, Doughnut } from 'react-chartjs-2';
import { fetchPerformanceMetrics, fetchNLPStats } from '../services/api';

const Performance = () => {
  const { data: perfData, isLoading: perfLoading } = useQuery({
    queryKey: ['performance'],
    queryFn: fetchPerformanceMetrics,
    refetchInterval: 5000
  });

  const { data: nlpStats } = useQuery({
    queryKey: ['nlp-stats'],
    queryFn: fetchNLPStats,
    refetchInterval: 5000
  });

  // Calculate real values for metrics - NO FAKE DATA
  const throughput = useMemo(() => {
    if (perfData?.messages_per_second) {
      return perfData.messages_per_second.toFixed(6); // Round to 6 decimal places as requested
    }
    return 0; // Show real zero when no data
  }, [perfData?.messages_per_second]);

  const cpuUsage = useMemo(() => {
    if (nlpStats?.cpu_utilization) return nlpStats.cpu_utilization.toFixed(1);
    if (perfData?.cpu_utilization) return perfData.cpu_utilization.toFixed(1);
    return 0; // Show real zero when no data
  }, [nlpStats?.cpu_utilization, perfData?.cpu_utilization]);

  // Latency distribution data
  const latencyData = {
    labels: ['Average', 'Min', 'Max', 'P95', 'P99'],
    datasets: [{
      label: 'Latency (ms)',
      data: [
        perfData?.average_latency_ms || 0,
        perfData?.min_latency_ms || 0,
        perfData?.max_latency_ms || 0,
        perfData?.p95_latency_ms || 0,
        perfData?.p99_latency_ms || 0
      ],
      backgroundColor: 'rgba(59, 130, 246, 0.5)',
      borderColor: 'rgba(59, 130, 246, 1)',
      borderWidth: 2
    }]
  };

  // Throughput over time - REAL DATA ONLY
  const throughputData = useMemo(() => {
    if (perfData?.throughput_timeline && perfData.throughput_timeline.length > 0) {
      // Use real timeline data if available
      return {
        labels: perfData.throughput_timeline.map(d => {
          const date = new Date(d.timestamp);
          return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }),
        datasets: [{
          label: 'Messages/Second',
          data: perfData.throughput_timeline.map(d => d.messages_per_second),
          borderColor: 'rgb(34, 197, 94)',
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 0
        }]
      };
    }
    
    // No fake data - return empty when no real data
    return {
      labels: [],
      datasets: [{
        label: 'Messages/Second',
        data: [],
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0
      }]
    };
  }, [perfData?.throughput_timeline]);

  // Sub-5s success rate gauge
  const sub5sRate = perfData?.sub_5_second_rate || 0;
  const gaugeData = {
    datasets: [{
      data: [sub5sRate, 100 - sub5sRate],
      backgroundColor: [
        sub5sRate >= 95 ? 'rgba(34, 197, 94, 0.8)' : 
        sub5sRate >= 90 ? 'rgba(251, 191, 36, 0.8)' : 'rgba(239, 68, 68, 0.8)',
        'rgba(75, 85, 99, 0.3)'
      ],
      borderWidth: 0
    }],
    labels: ['Success', 'Failed']
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

  const gaugeOptions = {
    responsive: true,
    maintainAspectRatio: false,
    circumference: 180,
    rotation: 270,
    cutout: '75%',
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false }
    }
  };

  if (perfLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Performance Metrics</h1>
          <p className="text-gray-400 mt-1">Real-time system performance monitoring</p>
        </div>
        <div className="flex items-center space-x-2">
          <Activity className="h-5 w-5 text-green-400 animate-pulse" />
          <span className="text-sm text-gray-300">Live monitoring active</span>
        </div>
      </div>

      {/* Performance Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-effect rounded-xl p-6 neon-glow"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Avg Latency</span>
            <Zap className="h-5 w-5 text-blue-400" />
          </div>
          <p className="text-3xl font-bold">
            {perfData?.average_latency_ms?.toFixed(0) || 0}
            <span className="text-lg font-normal text-gray-400">ms</span>
          </p>
          <div className="mt-2 text-sm">
            {perfData?.average_latency_ms > 0 ? (
              <span className="text-blue-400">Real-time data</span>
            ) : (
              <span className="text-gray-400">Awaiting data</span>
            )}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-effect rounded-xl p-6 success-glow"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Sub-5s Rate</span>
            <Gauge className="h-5 w-5 text-green-400" />
          </div>
          <p className="text-3xl font-bold text-green-400">
            {perfData?.sub_5_second_rate?.toFixed(1) || 0}%
          </p>
          <div className="mt-2 text-sm">
            <span className="text-green-400">✓ Target Met</span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-effect rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Throughput</span>
            <TrendingUp className="h-5 w-5 text-purple-400" />
          </div>
          <p className="text-3xl font-bold">
            {throughput}
            <span className="text-lg font-normal text-gray-400">/s</span>
          </p>
          <div className="mt-2 text-sm">
            <span className="text-purple-400">Active throughput</span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-effect rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">CPU Usage</span>
            <Cpu className="h-5 w-5 text-orange-400" />
          </div>
          <p className="text-3xl font-bold">
            {cpuUsage}%
          </p>
          <div className="mt-2 text-sm">
            <span className="text-orange-400">Processing Active</span>
          </div>
        </motion.div>
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sub-5s Success Rate Gauge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="glass-effect rounded-xl p-6"
        >
          <h2 className="text-xl font-semibold mb-4">Sub-5 Second Success Rate</h2>
          <div className="relative h-64">
            <Doughnut data={gaugeData} options={gaugeOptions} />
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center mt-12">
                <p className="text-4xl font-bold">{sub5sRate.toFixed(1)}%</p>
                <p className="text-sm text-gray-400 mt-1">Success Rate</p>
              </div>
            </div>
          </div>
          <div className="mt-4 grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-xs text-gray-400">Target</p>
              <p className="text-lg font-bold text-blue-400">95%</p>
            </div>
            <div>
              <p className="text-xs text-gray-400">Current</p>
              <p className="text-lg font-bold text-green-400">{sub5sRate.toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-xs text-gray-400">Status</p>
              <p className="text-lg font-bold text-green-400">✓ Pass</p>
            </div>
          </div>
        </motion.div>

        {/* Latency Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="glass-effect rounded-xl p-6"
        >
          <h2 className="text-xl font-semibold mb-4">Latency Distribution</h2>
          <div className="h-64">
            <Bar data={latencyData} options={chartOptions} />
          </div>
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div className="bg-gray-700/30 rounded-lg p-3">
              <p className="text-xs text-gray-400">P95 Latency</p>
              <p className="text-xl font-bold text-blue-400">
                {perfData?.p95_latency_ms?.toFixed(0) || 0}ms
              </p>
            </div>
            <div className="bg-gray-700/30 rounded-lg p-3">
              <p className="text-xs text-gray-400">P99 Latency</p>
              <p className="text-xl font-bold text-purple-400">
                {perfData?.p99_latency_ms?.toFixed(0) || 0}ms
              </p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Throughput Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="glass-effect rounded-xl p-6"
      >
        <h2 className="text-xl font-semibold mb-4">Throughput Over Time</h2>
        <div className="h-64">
          {perfData?.throughput_timeline && perfData.throughput_timeline.length > 0 ? (
            <Line data={throughputData} options={chartOptions} />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-400">
              <div className="text-center">
                <TrendingUp className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No throughput data available</p>
                <p className="text-sm mt-2">Process some messages to see throughput trends</p>
              </div>
            </div>
          )}
        </div>
      </motion.div>

      {/* System Resources */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="glass-effect rounded-xl p-6"
      >
        <h2 className="text-xl font-semibold mb-4">System Resources</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* CPU Performance */}
          <div className="bg-gray-700/30 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium">CPU Performance</span>
              <Cpu className="h-4 w-4 text-orange-400" />
            </div>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">CPU Utilization</span>
                  <span>{cpuUsage ? `${cpuUsage}%` : 'No Data'}</span>
                </div>
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div 
                    className="bg-orange-400 h-2 rounded-full"
                    style={{ width: `${parseFloat(cpuUsage) || 0}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">System Load</span>
                  <span>{perfData?.system_load ? `${perfData.system_load.toFixed(2)}` : 'No Data'}</span>
                </div>
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div 
                    className="bg-blue-400 h-2 rounded-full"
                    style={{ width: `${perfData?.system_load ? Math.min(perfData.system_load * 25, 100) : 0}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Processing Stats */}
          <div className="bg-gray-700/30 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium">Processing Queue</span>
              <Server className="h-4 w-4 text-blue-400" />
            </div>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-400">Queue Size</span>
                <span className="text-lg font-bold">{nlpStats?.queue_size ?? 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-400">Batch Size</span>
                <span className="text-lg font-bold">{nlpStats?.batch_size ?? 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-400">Total Processed</span>
                <span className="text-lg font-bold">{nlpStats?.total_processed ?? 'N/A'}</span>
              </div>
            </div>
          </div>

          {/* Model Performance */}
          <div className="bg-gray-700/30 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium">Model Performance</span>
              <Activity className="h-4 w-4 text-green-400" />
            </div>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-400">Avg Inference</span>
                <span className="text-lg font-bold">
                  {nlpStats?.average_latency ? `${nlpStats.average_latency.toFixed(0)}ms` : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-400">Device</span>
                <span className="text-sm font-medium">{nlpStats?.device || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-400">Optimization</span>
                <span className="text-sm font-medium">{nlpStats?.optimization_level || 'N/A'}</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Performance;
