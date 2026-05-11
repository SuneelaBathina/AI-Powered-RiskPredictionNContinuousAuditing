import React, { useState, useEffect } from 'react';
import {
  Typography,
  Box,
  Paper,
  CircularProgress,
  Alert,
  useTheme,
  Grid,
  Card,
  CardContent,
  Chip,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Assessment as AssessmentIcon,
  Security as SecurityIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  DateRange as DateRangeIcon
} from '@mui/icons-material';
import {
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
  ResponsiveContainer,
  LineChart,
  Line
} from 'recharts';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

function Reports() {
  const theme = useTheme();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [reportType, setReportType] = useState('summary');
  
  // Report data states
  const [auditReport, setAuditReport] = useState(null);
  const [riskMetrics, setRiskMetrics] = useState(null);
  const [auditFindings, setAuditFindings] = useState([]);
  const [anomalies, setAnomalies] = useState([]);

  const fetchAllReports = async () => {
    setRefreshing(true);
    setError(null);

    try {
      console.log('Fetching reports from backend API...');

      // Fetch all data in parallel
      const [auditResponse, riskResponse, findingsResponse, anomaliesResponse] = await Promise.all([
        axios.get(`${API_BASE}/api/audit-report`, {
          params: { type: reportType }
        }),
        axios.get(`${API_BASE}/api/risk-metrics`),
        axios.get(`${API_BASE}/api/audit/findings`),
        axios.get(`${API_BASE}/api/anomaly-detection`)
      ]);

      console.log('Audit Report:', auditResponse.data);
      console.log('Risk Metrics:', riskResponse.data);
      console.log('Findings:', findingsResponse.data);
      console.log('Anomalies:', anomaliesResponse.data);

      setAuditReport(auditResponse.data);
      setRiskMetrics(riskResponse.data);
      setAuditFindings(findingsResponse.data || []);
      setAnomalies(anomaliesResponse.data.anomalies || []);
      setLastUpdated(new Date());

    } catch (error) {
      console.error('Error fetching reports:', error);
      setError(`Failed to load reports: ${error.message}`);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchAllReports();
  }, [reportType]);

  const handleRefresh = () => {
    fetchAllReports();
  };

  const handleExportJSON = () => {
    const exportData = {
      auditReport,
      riskMetrics,
      auditFindings,
      anomalies,
      exportedAt: new Date().toISOString()
    };
    
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    const filename = `audit_report_${new Date().toISOString().split('T')[0]}.json`;
    
    const link = document.createElement('a');
    link.setAttribute('href', dataUri);
    link.setAttribute('download', filename);
    link.click();
  };

  const getSeverityColor = (severity) => {
    switch(severity?.toLowerCase()) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading reports from backend...</Typography>
      </Box>
    );
  }

  if (error) {
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
          {error}
        </Alert>
      </Box>
    );
  }

  // Extract data from responses
  const report = auditReport?.report || {};
  const executiveSummary = report.executive_summary || {};
  const keyFindings = report.key_findings || [];
  const recommendations = report.recommendations || [];
  const insights = auditReport?.insights || [];

  // Risk metrics data
  const riskByType = riskMetrics?.risk_by_type || {};
  const riskByLocation = riskMetrics?.risk_by_location || {};
  const timeSeriesRisk = riskMetrics?.time_series_risk || [];
  const recentAlerts = riskMetrics?.recent_alerts || [];

  // Calculate summary stats
  const totalTransactions = riskMetrics?.total_transactions || executiveSummary.total_transactions || 0;
  const highRiskCount = riskMetrics?.high_risk_count || executiveSummary.high_risk_transactions || 0;
  const mediumRiskCount = riskMetrics?.medium_risk_count || executiveSummary.medium_risk_transactions || 0;
  const lowRiskCount = riskMetrics?.low_risk_count || executiveSummary.low_risk_transactions || 0;
  const avgRiskScore = riskMetrics?.avg_risk_score || report.risk_metrics?.average_risk_score || 0;
  const auditFindingsCount = auditFindings.length || executiveSummary.total_audit_findings || 0;

  // Prepare pie chart data
  const pieData = [
    { name: 'High Risk', value: highRiskCount },
    { name: 'Medium Risk', value: mediumRiskCount },
    { name: 'Low Risk', value: lowRiskCount }
  ].filter(item => item.value > 0);

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            Audit Reports & Analytics
          </Typography>
          {lastUpdated && (
            <Typography variant="caption" color="text.secondary">
              Last updated: {lastUpdated.toLocaleString()}
            </Typography>
          )}
        </Box>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>Report Type</InputLabel>
            <Select
              value={reportType}
              label="Report Type"
              onChange={(e) => setReportType(e.target.value)}
            >
              <MenuItem value="summary">Summary Report</MenuItem>
              <MenuItem value="detailed">Detailed Report</MenuItem>
              <MenuItem value="executive">Executive Summary</MenuItem>
            </Select>
          </FormControl>
          <Tooltip title="Refresh">
            <IconButton onClick={handleRefresh} disabled={refreshing}>
              <RefreshIcon className={refreshing ? 'spin' : ''} />
            </IconButton>
          </Tooltip>
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            onClick={handleExportJSON}
          >
            Export JSON
          </Button>
        </Box>
      </Box>

      {/* Executive Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Transactions
              </Typography>
              <Typography variant="h4">
                {totalTransactions.toLocaleString()}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                From risk metrics API
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                High Risk
              </Typography>
              <Typography variant="h4" color="error">
                {highRiskCount.toLocaleString()}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                {((highRiskCount / totalTransactions) * 100 || 0).toFixed(1)}% of total
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Audit Findings
              </Typography>
              <Typography variant="h4" color="warning.main">
                {auditFindingsCount.toLocaleString()}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                {auditFindings.length} active findings
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Avg Risk Score
              </Typography>
              <Typography variant="h4" color="success.main">
                {(avgRiskScore * 100).toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="textSecondary">
                From ML model
              </Typography>
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
            {pieData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
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
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 250 }}>
                <Typography color="text.secondary">No risk distribution data</Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Risk by Location Chart */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%', minHeight: 350 }}>
            <Typography variant="h6" gutterBottom>
              Risk by Location
            </Typography>
            {Object.keys(riskByLocation).length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart
                  data={Object.entries(riskByLocation).map(([key, value]) => ({
                    location: key,
                    'High Risk': value.high_risk || 0,
                    'Total': value.count || 0
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
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 250 }}>
                <Typography color="text.secondary">No location data</Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Risk Trend Chart */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Risk Trend (Last 30 Days)
            </Typography>
            {timeSeriesRisk.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={timeSeriesRisk}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <RechartsTooltip />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="total" stroke="#8884d8" name="Total Transactions" />
                  <Line yAxisId="right" type="monotone" dataKey="high_risk" stroke="#ff4d4f" name="High Risk" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
                <Typography color="text.secondary">No trend data available</Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Key Findings Section */}
      <Accordion defaultExpanded sx={{ mb: 2 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">
            Key Audit Findings ({keyFindings.length || auditFindings.length})
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <List>
            {(keyFindings.length > 0 ? keyFindings : auditFindings).map((finding, index) => (
              <React.Fragment key={finding.id || index}>
                <ListItem alignItems="flex-start">
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mb: 1 }}>
                        <Chip
                          size="small"
                          label={finding.severity || 'MEDIUM'}
                          color={getSeverityColor(finding.severity)}
                        />
                        <Chip
                          size="small"
                          label={finding.type || 'FINDING'}
                          variant="outlined"
                        />
                        <Typography variant="subtitle2">
                          {finding.description || finding.type}
                        </Typography>
                      </Box>
                    }
                    secondary={
                      <Box>
                        {finding.recommendation && (
                          <Typography variant="body2" color="primary" gutterBottom>
                            Recommendation: {finding.recommendation}
                          </Typography>
                        )}
                        {finding.transaction_id && (
                          <Typography variant="caption" display="block" color="text.secondary">
                            Transaction: {finding.transaction_id} | 
                            Amount: ${finding.amount?.toLocaleString() || '0'}
                          </Typography>
                        )}
                        {finding.timestamp && (
                          <Typography variant="caption" display="block" color="text.secondary">
                            Date: {new Date(finding.timestamp).toLocaleString()}
                          </Typography>
                        )}
                      </Box>
                    }
                  />
                </ListItem>
                {index < (keyFindings.length || auditFindings.length) - 1 && <Divider />}
              </React.Fragment>
            ))}
            {keyFindings.length === 0 && auditFindings.length === 0 && (
              <ListItem>
                <ListItemText primary="No audit findings found" />
              </ListItem>
            )}
          </List>
        </AccordionDetails>
      </Accordion>

      {/* Recommendations Section */}
      {recommendations.length > 0 && (
        <Accordion sx={{ mb: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">Recommendations</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <List>
              {recommendations.map((rec, index) => (
                <ListItem key={index}>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <CheckCircleIcon color="success" fontSize="small" />
                        <Typography variant="body2">{rec}</Typography>
                      </Box>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </AccordionDetails>
        </Accordion>
      )}

      {/* Anomalies Section */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">
            Detected Anomalies ({anomalies.length})
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Transaction ID</TableCell>
                  <TableCell align="right">Amount</TableCell>
                  <TableCell align="right">Z-Score</TableCell>
                  <TableCell>Reason</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {anomalies.map((anomaly, index) => (
                  <TableRow key={anomaly.transaction_id || index}>
                    <TableCell>{anomaly.transaction_id}</TableCell>
                    <TableCell align="right">${anomaly.amount?.toLocaleString()}</TableCell>
                    <TableCell align="right">
                      <Chip
                        size="small"
                        label={anomaly.z_score?.toFixed(2)}
                        color={anomaly.z_score > 4 ? 'error' : 'warning'}
                      />
                    </TableCell>
                    <TableCell>{anomaly.reason}</TableCell>
                  </TableRow>
                ))}
                {anomalies.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={4} align="center">
                      No anomalies detected
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </AccordionDetails>
      </Accordion>

      {/* AI Insights */}
      {insights.length > 0 && (
        <Accordion sx={{ mt: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">AI-Powered Insights</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <List>
              {insights.map((insight, index) => (
                <ListItem key={index}>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <AssessmentIcon color="primary" fontSize="small" />
                        <Typography variant="body2">
                          {insight.content || insight}
                        </Typography>
                      </Box>
                    }
                    secondary={insight.metadata?.source && (
                      <Typography variant="caption" color="text.secondary">
                        Source: {insight.metadata.source} | 
                        Relevance: {(insight.relevance_score * 100).toFixed(0)}%
                      </Typography>
                    )}
                  />
                </ListItem>
              ))}
            </List>
          </AccordionDetails>
        </Accordion>
      )}

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

export default Reports;