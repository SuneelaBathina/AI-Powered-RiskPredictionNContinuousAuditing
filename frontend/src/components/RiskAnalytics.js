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
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress
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
  Assessment as AssessmentIcon,
  Download as DownloadIcon,
  LocationOn as LocationIcon,
  Category as CategoryIcon,
  Speed as SpeedIcon
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

const COLORS = {
  high: '#f44336',
  medium: '#ff9800',
  low: '#4caf50',
  primary: '#1976d2',
  info: '#2196f3'
};

const CHART_COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

function RiskAnalytics() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [riskMetrics, setRiskMetrics] = useState(null);

  const fetchRiskData = async () => {
    setRefreshing(true);
    setError(null);

    try {
      console.log('Fetching risk analytics data from backend...');
      
      const response = await axios.get(`${API_BASE}/api/risk-metrics`);
      console.log('Risk Metrics received:', response.data);
      
      setRiskMetrics(response.data);
      setLastUpdated(new Date());

    } catch (error) {
      console.error('Error fetching risk data:', error);
      setError(`Failed to load risk analytics: ${error.message}`);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchRiskData();
    const interval = setInterval(fetchRiskData, 60000);
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    fetchRiskData();
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading risk analytics...</Typography>
      </Box>
    );
  }

  if (error || !riskMetrics) {
    return (
      <Box sx={{ mt: 4 }}>
        <Alert 
          severity="error" 
          action={
            <Button color="inherit" size="small" onClick={handleRefresh}>
              Retry
            </Button>
          }
        >
          {error || 'Unable to load risk analytics'}
        </Alert>
      </Box>
    );
  }

  // Extract metrics with safe defaults
  const totalTransactions = riskMetrics.total_transactions || 0;
  const highRiskCount = riskMetrics.high_risk_count || 0;
  const mediumRiskCount = riskMetrics.medium_risk_count || 0;
  const lowRiskCount = riskMetrics.low_risk_count || 0;
  const avgRiskScore = riskMetrics.avg_risk_score || 0;
  const highRiskPercentage = riskMetrics.high_risk_percentage || 0;
  
  const riskByType = riskMetrics.risk_by_type || {};
  const riskByLocation = riskMetrics.risk_by_location || {};
  const timeSeriesRisk = riskMetrics.time_series_risk || [];
  const recentAlerts = riskMetrics.recent_alerts || [];

  // Prepare chart data
  const riskDistributionData = [
    { name: 'High Risk', value: highRiskCount },
    { name: 'Medium Risk', value: mediumRiskCount },
    { name: 'Low Risk', value: lowRiskCount }
  ].filter(item => item.value > 0);

  const riskByTypeData = Object.entries(riskByType).map(([type, data]) => ({
    type,
    count: data.count || 0,
    highRisk: data.high_risk || 0,
    riskPercentage: data.percentage || 0
  }));

  const riskByLocationData = Object.entries(riskByLocation).map(([loc, data]) => ({
    location: loc,
    count: data.count || 0,
    highRisk: data.high_risk || 0
  }));

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            Risk Analytics Dashboard
          </Typography>
          {lastUpdated && (
            <Typography variant="caption" color="text.secondary">
              Last updated: {lastUpdated.toLocaleString()}
            </Typography>
          )}
        </Box>
        <Tooltip title="Refresh">
          <IconButton onClick={handleRefresh} disabled={refreshing}>
            <RefreshIcon className={refreshing ? 'spin' : ''} />
          </IconButton>
        </Tooltip>
      </Box>

      {/* KPI Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: COLORS.primary, mr: 2 }}>
                  <AccountBalance />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" variant="body2">
                    Total Transactions
                  </Typography>
                  <Typography variant="h5">
                    {totalTransactions.toLocaleString()}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: COLORS.high, mr: 2 }}>
                  <WarningIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" variant="body2">
                    High Risk
                  </Typography>
                  <Typography variant="h5" color="error">
                    {highRiskCount.toLocaleString()}
                  </Typography>
                </Box>
              </Box>
              <Typography variant="caption" color="text.secondary">
                {highRiskPercentage.toFixed(1)}% of total
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: COLORS.medium, mr: 2 }}>
                  <SpeedIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" variant="body2">
                    Avg Risk Score
                  </Typography>
                  <Typography variant="h5">
                    {(avgRiskScore * 100).toFixed(1)}%
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: COLORS.info, mr: 2 }}>
                  <AssessmentIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" variant="body2">
                    Risk Distribution
                  </Typography>
                  <Typography variant="body2">
                    H: {highRiskCount} | M: {mediumRiskCount} | L: {lowRiskCount}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Risk Distribution Pie Chart */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%', minHeight: 350 }}>
            <Typography variant="h6" gutterBottom>
              Risk Distribution
            </Typography>
            {riskDistributionData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={riskDistributionData}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    dataKey="value"
                  >
                    {riskDistributionData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={index === 0 ? COLORS.high : index === 1 ? COLORS.medium : COLORS.low} 
                      />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 250 }}>
                <Typography color="text.secondary">No risk distribution data</Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Risk Trend Chart */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%', minHeight: 350 }}>
            <Typography variant="h6" gutterBottom>
              Risk Trend (Last 30 Days)
            </Typography>
            {timeSeriesRisk.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={timeSeriesRisk}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Line type="monotone" dataKey="total" stroke={COLORS.primary} name="Total Transactions" />
                  <Line type="monotone" dataKey="high_risk" stroke={COLORS.high} name="High Risk" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 250 }}>
                <Typography color="text.secondary">No trend data available</Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Risk by Transaction Type */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Risk by Transaction Type
            </Typography>
            {riskByTypeData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={riskByTypeData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="type" />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Bar dataKey="count" fill={COLORS.primary} name="Total Transactions" />
                  <Bar dataKey="highRisk" fill={COLORS.high} name="High Risk" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
                <Typography color="text.secondary">No transaction type data</Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Risk by Location */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Risk by Location
            </Typography>
            {riskByLocationData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={riskByLocationData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="location" />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Bar dataKey="count" fill={COLORS.primary} name="Total Transactions" />
                  <Bar dataKey="highRisk" fill={COLORS.high} name="High Risk" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
                <Typography color="text.secondary">No location data</Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Recent Alerts Table */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Recent High Risk Alerts
        </Typography>
        {recentAlerts.length > 0 ? (
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Transaction ID</TableCell>
                  <TableCell align="right">Amount</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Location</TableCell>
                  <TableCell align="right">Risk Score</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {recentAlerts.map((alert, index) => (
                  <TableRow key={index}>
                    <TableCell>{alert.transaction_id}</TableCell>
                    <TableCell align="right">${alert.amount?.toLocaleString()}</TableCell>
                    <TableCell>{alert.transaction_type}</TableCell>
                    <TableCell>{alert.location}</TableCell>
                    <TableCell align="right">
                      <Chip
                        size="small"
                        label={`${(alert.risk_score * 100).toFixed(0)}%`}
                        color={alert.risk_score > 0.7 ? 'error' : 'warning'}
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Typography color="text.secondary" align="center" sx={{ py: 4 }}>
            No recent alerts
          </Typography>
        )}
      </Paper>

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

export default RiskAnalytics;