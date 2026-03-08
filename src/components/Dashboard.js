import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Warning,
  CheckCircle,
  AccountBalance,
  Refresh as RefreshIcon,
  Timeline as TimelineIcon
} from '@mui/icons-material';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';
import axios from 'axios';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchDashboardData = async () => {
    setRefreshing(true);
    try {
      const response = await axios.get('http://localhost:5000/api/risk-metrics');
      setMetrics(response.data);
      setError(null);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setError('Failed to load dashboard data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Box sx={{ width: '100%', mt: 4 }}>
        <LinearProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  const summaryCards = [
    {
      title: 'Total Transactions',
      value: metrics?.total_transactions?.toLocaleString() || '0',
      change: '+12.3%',
      icon: <AccountBalance />,
      color: '#1976d2'
    },
    {
      title: 'High Risk',
      value: metrics?.high_risk_count?.toLocaleString() || '0',
      change: `${metrics?.high_risk_percentage?.toFixed(1)}% of total`,
      icon: <Warning />,
      color: '#d32f2f'
    },
    {
      title: 'Avg Risk Score',
      value: metrics?.avg_risk_score ? `${(metrics.avg_risk_score * 100).toFixed(1)}%` : '0%',
      change: '+2.5%',
      icon: <TrendingUp />,
      color: '#ed6c02'
    },
    {
      title: 'Audit Findings',
      value: metrics?.audit_findings_count || '0',
      change: '3 critical',
      icon: <CheckCircle />,
      color: '#2e7d32'
    }
  ];

  const radarData = Object.entries(metrics?.risk_by_type || {}).map(([key, value]) => ({
    type: key,
    value: value.percentage || 0
  }));

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" fontWeight="bold">
          Risk Dashboard
        </Typography>
        <Tooltip title="Refresh">
          <IconButton onClick={fetchDashboardData} disabled={refreshing}>
            <RefreshIcon className={refreshing ? 'spin' : ''} />
          </IconButton>
        </Tooltip>
      </Box>

      <Grid container spacing={3}>
        {/* Summary Cards */}
        {summaryCards.map((card, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ bgcolor: card.color, mr: 2 }}>
                    {card.icon}
                  </Avatar>
                  <Typography color="textSecondary" variant="body2">
                    {card.title}
                  </Typography>
                </Box>
                <Typography variant="h4" component="div" gutterBottom>
                  {card.value}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  {card.change}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}

        {/* Risk Distribution Chart */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Risk Distribution by Type
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={Object.entries(metrics?.risk_by_type || {}).map(([key, value]) => ({
                    name: key,
                    value: value.high_risk
                  }))}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {Object.entries(metrics?.risk_by_type || {}).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <RechartsTooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Risk Radar Chart */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Risk Profile Radar
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart outerRadius={90} data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="type" />
                <PolarRadiusAxis angle={30} domain={[0, 100]} />
                <Radar
                  name="Risk Score"
                  dataKey="value"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.6}
                />
                <RechartsTooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Risk Trend */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Risk Trend (Last 30 Days)
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={metrics?.time_series_risk || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <RechartsTooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="total"
                  stackId="1"
                  stroke="#8884d8"
                  fill="#8884d8"
                  name="Total Transactions"
                />
                <Area
                  type="monotone"
                  dataKey="high_risk"
                  stackId="2"
                  stroke="#ff4d4f"
                  fill="#ff4d4f"
                  name="High Risk"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Risk by Location */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Risk by Location
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={Object.entries(metrics?.risk_by_location || {}).map(([key, value]) => ({
                  location: key,
                  'High Risk': value.high_risk,
                  'Total': value.count
                }))}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="location" />
                <YAxis />
                <RechartsTooltip />
                <Legend />
                <Bar dataKey="High Risk" fill="#ff4d4f" />
                <Bar dataKey="Total" fill="#1976d2" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Recent Alerts */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Recent High Risk Alerts
            </Typography>
            <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
              {metrics?.recent_alerts?.map((alert, index) => (
                <Box
                  key={index}
                  sx={{
                    p: 2,
                    mb: 1,
                    bgcolor: 'background.default',
                    borderRadius: 1,
                    border: '1px solid',
                    borderColor: 'divider'
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <WarningIcon color="error" sx={{ mr: 1, fontSize: 20 }} />
                    <Typography variant="subtitle2">
                      {alert.transaction_id}
                    </Typography>
                    <Chip
                      label={`${(alert.risk_score * 100).toFixed(0)}%`}
                      size="small"
                      color="error"
                      sx={{ ml: 'auto' }}
                    />
                  </Box>
                  <Typography variant="body2" color="textSecondary">
                    Amount: ${alert.amount?.toFixed(2)} | Type: {alert.transaction_type}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Paper>
        </Grid>
      </Grid>

      <style jsx>{`
        @keyframes spin {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }
        .spin {
          animation: spin 1s linear infinite;
        }
      `}</style>
    </Box>
  );
}

export default Dashboard;