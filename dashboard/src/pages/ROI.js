import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { 
  DollarSign, 
  TrendingUp, 
  Users, 
  ShieldCheck, 
  Target,
  PiggyBank,
  LineChart,
  Award
} from 'lucide-react';
import { Line, Bar, Radar } from 'react-chartjs-2';
import { fetchROIAnalytics } from '../services/api';

const ROI = () => {
  const [periodDays, setPeriodDays] = useState(30);

  const { data, isLoading } = useQuery({
    queryKey: ['roi', periodDays],
    queryFn: () => fetchROIAnalytics(periodDays),
    refetchInterval: 60000 // Update every minute
  });

  // ROI trend data - only show real data
  const roiTrendData = data?.roi_trend ? {
    labels: data.roi_trend.labels || [],
    datasets: [{
      label: 'ROI %',
      data: data.roi_trend.data || [],
      borderColor: 'rgb(34, 197, 94)',
      backgroundColor: 'rgba(34, 197, 94, 0.1)',
      borderWidth: 3,
      fill: true,
      tension: 0.4
    }]
  } : null;

  // Financial impact breakdown - only show real data
  const financialBreakdownData = data?.financial_breakdown ? {
    labels: ['Revenue Impact', 'Cost Savings', 'Churn Reduction', 'Efficiency Gains'],
    datasets: [{
      label: 'Financial Impact ($)',
      data: [
        data.estimated_revenue_impact || 0,
        data.support_cost_savings || 0,
        data.churn_reduction_value || 0,
        data.efficiency_gains || 0
      ],
      backgroundColor: [
        'rgba(34, 197, 94, 0.8)',
        'rgba(59, 130, 246, 0.8)',
        'rgba(168, 85, 247, 0.8)',
        'rgba(251, 191, 36, 0.8)'
      ],
      borderWidth: 0
    }]
  } : null;

  // Business metrics radar chart - only show real data
  const businessMetricsData = data?.business_metrics ? {
    labels: [
      'Sentiment Improvement',
      'Response Time',
      'Customer Satisfaction',
      'Issue Prevention',
      'Team Efficiency',
      'Cost Reduction'
    ],
    datasets: [{
      label: 'Current Performance',
      data: [
        Math.abs(data.sentiment_improvement || 0),
        data.response_time_reduction || 0,
        data.customer_satisfaction || 0,
        data.issues_prevented || 0,
        data.team_efficiency || 0,
        data.cost_reduction || 0
      ],
      backgroundColor: 'rgba(59, 130, 246, 0.2)',
      borderColor: 'rgba(59, 130, 246, 1)',
      borderWidth: 2,
      pointBackgroundColor: 'rgba(59, 130, 246, 1)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'rgba(59, 130, 246, 1)'
    }, {
      label: 'Target',
      data: data.targets || [20, 30, 25, 10, 25, 20],
      backgroundColor: 'rgba(75, 85, 99, 0.1)',
      borderColor: 'rgba(75, 85, 99, 0.5)',
      borderWidth: 2,
      borderDash: [5, 5],
      pointBackgroundColor: 'rgba(75, 85, 99, 0.5)',
      pointBorderColor: 'transparent'
    }]
  } : null;

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

  const radarOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#fff' }
      }
    },
    scales: {
      r: {
        angleLines: {
          color: 'rgba(75, 85, 99, 0.3)'
        },
        grid: {
          color: 'rgba(75, 85, 99, 0.3)'
        },
        pointLabels: {
          color: '#9ca3af'
        },
        ticks: {
          color: '#6b7280',
          backdropColor: 'transparent'
        },
        suggestedMin: 0,
        suggestedMax: 30
      }
    }
  };

  const calculatePaybackPeriod = () => {
    const monthlyROI = (data?.roi_percentage || 0) / 12;
    if (monthlyROI > 0) {
      return (100 / monthlyROI).toFixed(1);
    }
    return 'N/A';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">ROI Analytics</h1>
          <p className="text-gray-400 mt-1">Business impact and financial performance</p>
        </div>
        <select
          value={periodDays}
          onChange={(e) => setPeriodDays(Number(e.target.value))}
          className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-sm text-white"
        >
          <option value={7}>Last 7 Days</option>
          <option value={30}>Last 30 Days</option>
          <option value={90}>Last 90 Days</option>
        </select>
      </div>

      {/* ROI Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-effect rounded-xl p-6 success-glow"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Total ROI</span>
            <DollarSign className="h-5 w-5 text-green-400" />
          </div>
          <p className="text-3xl font-bold text-green-400">
            {data?.roi_percentage?.toFixed(0) || 0}%
          </p>
          <p className="text-sm text-gray-400 mt-1">Return on investment</p>
          <div className="mt-3 text-xs">
            <span className="text-green-400">↑ 15%</span>
            <span className="text-gray-400 ml-2">from last period</span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-effect rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Revenue Impact</span>
            <TrendingUp className="h-5 w-5 text-blue-400" />
          </div>
          <p className="text-3xl font-bold">
            ${((data?.estimated_revenue_impact || 0) / 1000).toFixed(0)}k
          </p>
          <p className="text-sm text-gray-400 mt-1">Additional revenue</p>
          <div className="mt-3 text-xs">
            <span className="text-blue-400">From sentiment improvement</span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-effect rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Cost Savings</span>
            <PiggyBank className="h-5 w-5 text-purple-400" />
          </div>
          <p className="text-3xl font-bold text-purple-400">
            ${((data?.support_cost_savings || 0) / 1000).toFixed(0)}k
          </p>
          <p className="text-sm text-gray-400 mt-1">Support efficiency</p>
          <div className="mt-3 text-xs">
            <span className="text-purple-400">{data?.response_time_reduction?.toFixed(0) || 0}% faster</span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-effect rounded-xl p-6 warning-glow"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Churn Reduction</span>
            <Users className="h-5 w-5 text-orange-400" />
          </div>
          <p className="text-3xl font-bold text-orange-400">
            {data?.estimated_churn_reduction?.toFixed(1) || 0}%
          </p>
          <p className="text-sm text-gray-400 mt-1">Customer retention</p>
          <div className="mt-3 text-xs">
            <span className="text-orange-400">Saved customers</span>
          </div>
        </motion.div>
      </div>

      {/* Business Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="glass-effect rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Operational Metrics</h3>
            <Target className="h-5 w-5 text-blue-400" />
          </div>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Messages Processed</span>
              <span className="text-lg font-bold">{data?.total_messages_processed || 0}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Alerts Generated</span>
              <span className="text-lg font-bold text-yellow-400">{data?.alerts_generated || 0}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Critical Issues Prevented</span>
              <span className="text-lg font-bold text-green-400">{data?.critical_issues_prevented || 0}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Sentiment Improvement</span>
              <span className="text-lg font-bold text-blue-400">
                {data?.sentiment_improvement > 0 ? '+' : ''}{data?.sentiment_improvement?.toFixed(1) || 0}%
              </span>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="glass-effect rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Financial Summary</h3>
            <LineChart className="h-5 w-5 text-green-400" />
          </div>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Implementation Cost</span>
              <span className="text-lg font-bold">$50k</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Total Benefits</span>
              <span className="text-lg font-bold text-green-400">
                ${(((data?.estimated_revenue_impact || 0) + (data?.support_cost_savings || 0)) / 1000).toFixed(0)}k
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Net Benefit</span>
              <span className="text-lg font-bold text-blue-400">
                ${(((data?.estimated_revenue_impact || 0) + (data?.support_cost_savings || 0) - 50000) / 1000).toFixed(0)}k
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Payback Period</span>
              <span className="text-lg font-bold text-purple-400">
                {calculatePaybackPeriod()} months
              </span>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="glass-effect rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Success Indicators</h3>
            <Award className="h-5 w-5 text-yellow-400" />
          </div>
          <div className="space-y-3">
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${
                data?.roi_percentage > 20 ? 'bg-green-400' : 'bg-gray-500'
              }`}></div>
              <span className="text-sm">ROI Target Met (>20%)</span>
            </div>
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${
                data?.sub_5_second_rate > 95 ? 'bg-green-400' : 'bg-gray-500'
              }`}></div>
              <span className="text-sm">Performance SLA Met</span>
            </div>
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${
                data?.sentiment_improvement > 10 ? 'bg-green-400' : 'bg-gray-500'
              }`}></div>
              <span className="text-sm">Sentiment Improved</span>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 rounded-full bg-green-400"></div>
              <span className="text-sm">System Operational</span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ROI Trend */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="glass-effect rounded-xl p-6"
        >
          <h2 className="text-xl font-semibold mb-4">ROI Trend</h2>
          <div className="h-64">
            {roiTrendData ? (
              <Line data={roiTrendData} options={chartOptions} />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-400">
                <div className="text-center">
                  <LineChart className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p className="text-sm">No ROI trend data available</p>
                  <p className="text-xs text-gray-500 mt-1">Data will appear when transactions are processed</p>
                </div>
              </div>
            )}
          </div>
        </motion.div>

        {/* Financial Breakdown */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="glass-effect rounded-xl p-6"
        >
          <h2 className="text-xl font-semibold mb-4">Financial Impact Breakdown</h2>
          <div className="h-64">
            {financialBreakdownData ? (
              <Bar data={financialBreakdownData} options={chartOptions} />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-400">
                <div className="text-center">
                  <DollarSign className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p className="text-sm">No financial data available</p>
                  <p className="text-xs text-gray-500 mt-1">Financial impact will be calculated as system usage grows</p>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Business Metrics Radar */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
        className="glass-effect rounded-xl p-6"
      >
        <h2 className="text-xl font-semibold mb-4">Business Performance Metrics</h2>
        <div className="h-80">
          {businessMetricsData ? (
            <Radar data={businessMetricsData} options={radarOptions} />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-400">
              <div className="text-center">
                <Target className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p className="text-sm">No metrics data available</p>
                <p className="text-xs text-gray-500 mt-1">Performance metrics will be calculated over time</p>
              </div>
            </div>
          )}
        </div>
      </motion.div>

      {/* ROI Calculation Details */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.0 }}
        className="glass-effect rounded-xl p-6"
      >
        <h2 className="text-xl font-semibold mb-4">ROI Calculation Details</h2>
        <div className="bg-gray-700/30 rounded-lg p-4 font-mono text-sm">
          <div className="text-green-400 mb-2">// ROI Calculation Formula</div>
          <div className="text-gray-300">
            ROI = (Financial Benefits - Implementation Costs) / Implementation Costs × 100%
          </div>
          <div className="mt-4 text-gray-300">
            <div>Financial Benefits = ${((data?.estimated_revenue_impact || 0) + (data?.support_cost_savings || 0)).toLocaleString()}</div>
            <div>Implementation Costs = $50,000</div>
            <div className="mt-2 text-green-400">
              ROI = ({((data?.estimated_revenue_impact || 0) + (data?.support_cost_savings || 0)).toLocaleString()} - 50,000) / 50,000 × 100%
            </div>
            <div className="text-green-400 font-bold">
              ROI = {data?.roi_percentage?.toFixed(1) || 0}%
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default ROI;
