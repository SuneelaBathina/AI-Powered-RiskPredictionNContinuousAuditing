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
  Tooltip,
  Alert,
  Avatar,
  Button
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Warning as WarningIcon,
  CheckCircle,
  AccountBalance,
  Refresh as RefreshIcon,
  Timeline as TimelineIcon,
  Error as ErrorIcon,
  Security as SecurityIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchDashboardData = async () => {
    setRefreshing(true);
    setError(null);
    
    try {
      console.log('Fetching dashboard data from backend API...');
      
      // Fetch risk metrics
      const response = await axios.get(`${API_BASE}/api/risk-metrics`);
      console.log('Risk metrics received:', response.data);
      
      if (!response.data) {
        throw new Error('No data received from API');
      }
      
      setMetrics(response.data);
      setLastUpdated(new Date());
      
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setError(`Failed to load dashboard data: ${error.message}`);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Box sx={{ width: '100%', mt: 4 }}>
        <LinearProgress />
        <Typography sx={{ mt: 2, textAlign: 'center' }}>Loading dashboard data...</Typography>
      </Box>
    );
  }

  if (error || !metrics) {
    return (
      <Box sx={{ mt: 4 }}>
        <Alert 
          severity="error" 
          action={
            <Button color="inherit" size="small" onClick={fetchDashboardData}>
              Retry
            </Button>
          }
        >
          {error || 'Unable to load dashboard data'}
        </Alert>
      </Box>
    );
  }

  // Extract metrics with safe defaults
  const totalTransactions = metrics.total_transactions || 0;
  const highRiskCount = metrics.high_risk_count || 0;
  const mediumRiskCount = metrics.medium_risk_count || 0;
  const lowRiskCount = metrics.low_risk_count || 0;
  const avgRiskScore = metrics.avg_risk_score || 0;
  const highRiskPercentage = metrics.high_risk_percentage || 0;
  const auditFindingsCount = metrics.audit_findings_count || 0;
  const recentAlerts = metrics.recent_alerts || [];
  const riskByType = metrics.risk_by_type || {};
  const riskByLocation = metrics.risk_by_location || {};
  const timeSeriesRisk = metrics.time_series_risk || [];

  console.log('Rendering with data:', {
    totalTransactions,
    highRiskCount,
    mediumRiskCount,
    lowRiskCount,
    riskByTypeKeys: Object.keys(riskByType),
    riskByLocationKeys: Object.keys(riskByLocation),
    timeSeriesLength: timeSeriesRisk.length,
    alertsLength: recentAlerts.length
  });

  // Prepare pie chart data
  const pieData = [
    { name: 'High Risk', value: highRiskCount },
    { name: 'Medium Risk', value: mediumRiskCount },
    { name: 'Low Risk', value: lowRiskCount }
  ].filter(item => item.value > 0);

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" fontWeight="bold">
          Risk Dashboard
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {lastUpdated && (
            <Typography variant="caption" color="text.secondary">
              Last updated: {lastUpdated.toLocaleTimeString()}
            </Typography>
          )}
          <Tooltip title="Refresh">
            <IconButton onClick={fetchDashboardData} disabled={refreshing}>
              <RefreshIcon className={refreshing ? 'spin' : ''} />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%', bgcolor: '#e3f2fd' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: '#1976d2', mr: 2 }}>
                  <AccountBalance />
                </Avatar>
                <Typography color="textSecondary" variant="body2">
                  Total Transactions
                </Typography>
              </Box>
              <Typography variant="h4" component="div" gutterBottom>
                {totalTransactions.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                From backend API
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%', bgcolor: '#ffebee' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: '#d32f2f', mr: 2 }}>
                  <WarningIcon />
                </Avatar>
                <Typography color="textSecondary" variant="body2">
                  High Risk
                </Typography>
              </Box>
              <Typography variant="h4" component="div" gutterBottom>
                {highRiskCount.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {highRiskPercentage.toFixed(1)}% of total
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%', bgcolor: '#fff3e0' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: '#ed6c02', mr: 2 }}>
                  <AssessmentIcon />
                </Avatar>
                <Typography color="textSecondary" variant="body2">
                  Avg Risk Score
                </Typography>
              </Box>
              <Typography variant="h4" component="div" gutterBottom>
                {(avgRiskScore * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="textSecondary">
                From ML model
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%', bgcolor: '#e8f5e9' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: '#2e7d32', mr: 2 }}>
                  <SecurityIcon />
                </Avatar>
                <Typography color="textSecondary" variant="body2">
                  Audit Findings
                </Typography>
              </Box>
              <Typography variant="h4" component="div" gutterBottom>
                {auditFindingsCount.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {recentAlerts.length} recent alerts
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Risk Distribution Pie Chart */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%', minHeight: 400 }}>
            <Typography variant="h6" gutterBottom>
              Risk Distribution
            </Typography>
            {pieData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
                <Typography color="text.secondary">No risk distribution data available</Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Risk by Location */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%', minHeight: 400 }}>
            <Typography variant="h6" gutterBottom>
              Risk by Location
            </Typography>
            {Object.keys(riskByLocation).length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={Object.entries(riskByLocation).map(([key, value]) => ({
                    location: key,
                    'High Risk': value.high_risk || 0,
                    'Total': value.count || 0,
                    'Risk %': value.percentage?.toFixed(1) || 0
                  }))}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="location" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <RechartsTooltip />
                  <Legend />
                  <Bar yAxisId="left" dataKey="High Risk" fill="#ff4d4f" />
                  <Bar yAxisId="left" dataKey="Total" fill="#1976d2" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
                <Typography color="text.secondary">No location data available</Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Risk Trend */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Risk Trend (Last 30 Days)
            </Typography>
            {timeSeriesRisk.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={timeSeriesRisk}>
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
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
                <Typography color="text.secondary">No trend data available</Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Recent Alerts */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Recent High Risk Alerts
            </Typography>
            {recentAlerts.length > 0 ? (
              <Grid container spacing={2}>
                {recentAlerts.map((alert, index) => (
                  <Grid item xs={12} md={4} key={index}>
                    <Card variant="outlined" sx={{ bgcolor: '#fff3f3' }}>
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                          <WarningIcon color="error" sx={{ mr: 1 }} />
                          <Typography variant="subtitle2" noWrap>
                            {alert.transaction_id}
                          </Typography>
                        </Box>
                        <Typography variant="body2" gutterBottom>
                          Amount: ${alert.amount?.toLocaleString()}
                        </Typography>
                        <Typography variant="body2" gutterBottom>
                          Type: {alert.transaction_type}
                        </Typography>
                        <Typography variant="body2" gutterBottom>
                          Location: {alert.location}
                        </Typography>
                        <Chip
                          size="small"
                          label={`Risk: ${(alert.risk_score * 100).toFixed(0)}%`}
                          color="error"
                        />
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Typography color="text.secondary" align="center" sx={{ py: 4 }}>
                No recent alerts
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* Risk by Transaction Type */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Risk by Transaction Type
            </Typography>
            {Object.keys(riskByType).length > 0 ? (
              <Grid container spacing={2}>
                {Object.entries(riskByType).map(([type, data]) => (
                  <Grid item xs={12} sm={6} md={3} key={type}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom fontWeight="bold">
                          {type}
                        </Typography>
                        <Typography variant="body2">
                          Total: {data.count?.toLocaleString()}
                        </Typography>
                        <Typography variant="body2" color="error">
                          High Risk: {data.high_risk}
                        </Typography>
                        <Box sx={{ mt: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={data.percentage || 0}
                            color={data.percentage > 2 ? 'error' : data.percentage > 1 ? 'warning' : 'success'}
                            sx={{ height: 8, borderRadius: 4 }}
                          />
                        </Box>
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                          Risk: {data.percentage?.toFixed(1)}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Typography color="text.secondary" align="center" sx={{ py: 4 }}>
                No transaction type data available
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .spin {
          animation: spin 1s linear infinite;
        }
      `}</style>
    </Box>
  );
}

export default Dashboard;