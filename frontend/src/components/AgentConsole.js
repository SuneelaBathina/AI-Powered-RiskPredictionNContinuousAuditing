import React, { useState, useEffect } from 'react';
import {
  Box, Grid, Paper, Typography, AppBar, Toolbar, IconButton,
  Badge, Card, CardContent, LinearProgress, Chip, Table, TableBody,
  TableCell, TableContainer, TableHead, TableRow, Button, Alert, Snackbar,
  Tab, Tabs, Avatar, Divider, Switch, FormControlLabel, TextField,
  Dialog, DialogTitle, DialogContent, DialogActions, CircularProgress,
  MenuItem, Select, FormControl, InputLabel, InputAdornment
} from '@mui/material';
import {
  Notifications as NotificationsIcon,
  Dashboard as DashboardIcon,
  TrendingUp as TrendingUpIcon, Warning as WarningIcon,
  Memory as MemoryIcon, Assessment as AssessmentIcon,
  Refresh as RefreshIcon, PlayArrow as PlayIcon, Stop as StopIcon,
  Timeline as TimelineIcon, Security as SecurityIcon,
  Visibility as VisibilityIcon, Search as SearchIcon,
  CloudUpload as CloudUploadIcon, Download as DownloadIcon
} from '@mui/icons-material';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts';

const AgentConsole = () => {
  // State Management
  const [activeTab, setActiveTab] = useState(0);
  const [agents, setAgents] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [transactions, setTransactions] = useState([]);
  const [auditLogs, setAuditLogs] = useState([]);
  const [riskMetrics, setRiskMetrics] = useState({
    overallRisk: 0, highRiskCount: 0, mediumRiskCount: 0,
    lowRiskCount: 0, totalTransactions: 0, totalAmount: 0, fraudAttempts: 0
  });
  const [realTimeData, setRealTimeData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [searchTerm, setSearchTerm] = useState('');
  const [filterRisk, setFilterRisk] = useState('all');
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [simulationRunning, setSimulationRunning] = useState(true);

  // Initialize Agents Data
  const initializeAgents = () => {
    const initialAgents = [
      { 
        id: 1, 
        name: 'Risk Detector AI', 
        status: 'running', 
        tasks_processed: 15420, 
        cpu_usage: 25.5, 
        memory_usage: 30.2, 
        last_active: new Date().toISOString(), 
        version: '2.1.0', 
        uptime: '12h 34m',
        description: 'Detects high-risk transactions using ML models'
      },
      { 
        id: 2, 
        name: 'Fraud Scanner', 
        status: 'running', 
        tasks_processed: 8921, 
        cpu_usage: 35.2, 
        memory_usage: 28.7, 
        last_active: new Date().toISOString(), 
        version: '2.1.0', 
        uptime: '8h 22m',
        description: 'Scans for fraudulent patterns and anomalies'
      },
      { 
        id: 3, 
        name: 'Compliance Monitor', 
        status: 'running', 
        tasks_processed: 5234, 
        cpu_usage: 18.3, 
        memory_usage: 22.1, 
        last_active: new Date().toISOString(), 
        version: '2.0.5', 
        uptime: '24h 15m',
        description: 'Ensures regulatory compliance'
      },
      { 
        id: 4, 
        name: 'Transaction Analyzer', 
        status: 'idle', 
        tasks_processed: 3456, 
        cpu_usage: 5.2, 
        memory_usage: 15.8, 
        last_active: new Date().toISOString(), 
        version: '2.1.0', 
        uptime: '4h 45m',
        description: 'Analyzes transaction patterns in real-time'
      },
      { 
        id: 5, 
        name: 'Anomaly Hunter', 
        status: 'running', 
        tasks_processed: 12789, 
        cpu_usage: 42.1, 
        memory_usage: 35.4, 
        last_active: new Date().toISOString(), 
        version: '2.2.0', 
        uptime: '6h 12m',
        description: 'Detects unusual behavior and outliers'
      }
    ];
    setAgents(initialAgents);
  };

  // Generate Mock Transactions
  const generateMockTransactions = () => {
    const types = ['Payment', 'Transfer', 'Withdrawal', 'Deposit', 'Refund'];
    const locations = ['New York', 'London', 'Tokyo', 'Singapore', 'Dubai', 'Mumbai'];
    const devices = ['Desktop', 'Mobile', 'Tablet'];
    const statuses = ['approved', 'reviewing', 'flagged'];
    
    const transactionsList = [];
    for (let i = 0; i < 100; i++) {
      const amount = Math.random() * 50000;
      const riskScore = Math.random() * 100;
      transactionsList.push({
        id: i + 1,
        transaction_id: `TX-${String(i + 1).padStart(8, '0')}`,
        amount: amount,
        timestamp: new Date(Date.now() - Math.random() * 30 * 24 * 3600000).toISOString(),
        account_id: `ACC-${Math.floor(Math.random() * 9000) + 1000}`,
        transaction_type: types[Math.floor(Math.random() * types.length)],
        location: locations[Math.floor(Math.random() * locations.length)],
        device: devices[Math.floor(Math.random() * devices.length)],
        risk_score: riskScore,
        is_fraudulent: riskScore > 85,
        status: riskScore > 70 ? 'flagged' : riskScore > 40 ? 'reviewing' : 'approved'
      });
    }
    return transactionsList;
  };

  // Generate Mock Alerts
  const generateMockAlerts = () => {
    const alertTypes = ['High Risk Transaction', 'Suspicious Pattern', 'Unusual Location', 'Rapid Transactions'];
    const severities = ['critical', 'high', 'medium', 'low'];
    const messages = [
      'Transaction amount exceeds normal pattern',
      'Multiple failed authentication attempts',
      'Unusual geographic location detected',
      'Rapid succession of transactions'
    ];
    
    const alertsList = [];
    for (let i = 0; i < 8; i++) {
      alertsList.push({
        id: i + 1,
        alert_type: alertTypes[Math.floor(Math.random() * alertTypes.length)],
        severity: severities[Math.floor(Math.random() * severities.length)],
        message: messages[Math.floor(Math.random() * messages.length)],
        timestamp: new Date(Date.now() - Math.random() * 48 * 3600000).toISOString(),
        resolved: false,
        transaction_id: `TX-${Math.floor(Math.random() * 1000)}`
      });
    }
    return alertsList;
  };

  // Generate Audit Logs
  const generateAuditLogs = () => {
    const agents = ['Risk Detector AI', 'Fraud Scanner', 'Compliance Monitor', 'Transaction Analyzer', 'Anomaly Hunter'];
    const actions = ['ANALYZE Transaction', 'FLAG Risk', 'UPDATE Model', 'GENERATE Report', 'TRIGGER Alert', 'RESOLVE Issue'];
    const riskLevels = ['low', 'medium', 'high', 'critical'];
    
    const logs = [];
    for (let i = 0; i < 50; i++) {
      logs.push({
        id: i + 1,
        agent_name: agents[Math.floor(Math.random() * agents.length)],
        action: actions[Math.floor(Math.random() * actions.length)],
        timestamp: new Date(Date.now() - i * 3600000).toISOString(),
        details: `Processed transaction with risk score ${(Math.random() * 100).toFixed(1)}%`,
        risk_level: riskLevels[Math.floor(Math.random() * riskLevels.length)]
      });
    }
    return logs;
  };

  // Calculate Risk Metrics
  const calculateRiskMetrics = (transactionsList) => {
    const total = transactionsList.length;
    const highRisk = transactionsList.filter(t => t.risk_score > 70).length;
    const mediumRisk = transactionsList.filter(t => t.risk_score > 40 && t.risk_score <= 70).length;
    const lowRisk = transactionsList.filter(t => t.risk_score <= 40).length;
    const totalAmount = transactionsList.reduce((sum, t) => sum + t.amount, 0);
    const fraudAttempts = transactionsList.filter(t => t.is_fraudulent).length;
    const overallRisk = (highRisk * 100 + mediumRisk * 50) / total;
    
    return {
      overallRisk: overallRisk,
      highRiskCount: highRisk,
      mediumRiskCount: mediumRisk,
      lowRiskCount: lowRisk,
      totalTransactions: total,
      totalAmount: totalAmount,
      fraudAttempts: fraudAttempts
    };
  };

  // Initialize Data
  useEffect(() => {
    initializeAgents();
    const mockTransactionsData = generateMockTransactions();
    setTransactions(mockTransactionsData);
    setAlerts(generateMockAlerts());
    setAuditLogs(generateAuditLogs());
    setRiskMetrics(calculateRiskMetrics(mockTransactionsData));
    
    // Initialize real-time data
    const initialRealTime = [];
    for (let i = 0; i < 20; i++) {
      initialRealTime.push({
        timestamp: new Date(Date.now() - (20 - i) * 60000).toLocaleTimeString(),
        risk: Math.random() * 60 + 20
      });
    }
    setRealTimeData(initialRealTime);
    setLastUpdate(new Date());
  }, []);

  // Simulate real-time updates
  useEffect(() => {
    if (!simulationRunning) return;
    
    const interval = setInterval(() => {
      // Simulate new transaction
      const types = ['Payment', 'Transfer', 'Withdrawal', 'Deposit'];
      const amount = Math.random() * 50000;
      const riskScore = Math.random() * 100;
      
      const newTransaction = {
        id: transactions.length + 1,
        transaction_id: `TX-NEW-${String(Date.now()).slice(-8)}`,
        amount: amount,
        timestamp: new Date().toISOString(),
        account_id: `ACC-${Math.floor(Math.random() * 9000) + 1000}`,
        transaction_type: types[Math.floor(Math.random() * types.length)],
        location: ['New York', 'London', 'Tokyo', 'Singapore'][Math.floor(Math.random() * 4)],
        device: ['Desktop', 'Mobile', 'Tablet'][Math.floor(Math.random() * 3)],
        risk_score: riskScore,
        is_fraudulent: riskScore > 85,
        status: riskScore > 70 ? 'flagged' : riskScore > 40 ? 'reviewing' : 'approved'
      };
      
      setTransactions(prev => [newTransaction, ...prev].slice(0, 200));
      setRiskMetrics(prev => {
        const newHighRisk = newTransaction.risk_score > 70 ? prev.highRiskCount + 1 : prev.highRiskCount;
        const newMediumRisk = newTransaction.risk_score > 40 && newTransaction.risk_score <= 70 ? prev.mediumRiskCount + 1 : prev.mediumRiskCount;
        const newLowRisk = newTransaction.risk_score <= 40 ? prev.lowRiskCount + 1 : prev.lowRiskCount;
        const newOverallRisk = (newHighRisk * 100 + newMediumRisk * 50) / (prev.totalTransactions + 1);
        
        return {
          ...prev,
          overallRisk: newOverallRisk,
          highRiskCount: newHighRisk,
          mediumRiskCount: newMediumRisk,
          lowRiskCount: newLowRisk,
          totalTransactions: prev.totalTransactions + 1,
          totalAmount: prev.totalAmount + newTransaction.amount,
          fraudAttempts: newTransaction.is_fraudulent ? prev.fraudAttempts + 1 : prev.fraudAttempts
        };
      });
      
      // Update real-time chart
      setRealTimeData(prev => [...prev, {
        timestamp: new Date().toLocaleTimeString(),
        risk: riskScore
      }].slice(-30));
      
      // Generate alert for high-risk
      if (riskScore > 85) {
        const newAlert = {
          id: alerts.length + 1,
          alert_type: 'High Risk Transaction',
          severity: 'critical',
          message: `High risk transaction detected: $${amount.toFixed(2)}`,
          timestamp: new Date().toISOString(),
          resolved: false,
          transaction_id: newTransaction.transaction_id
        };
        setAlerts(prev => [newAlert, ...prev]);
        showNotification(`⚠️ CRITICAL: High risk transaction detected!`, 'error');
      } else if (riskScore > 70) {
        const newAlert = {
          id: alerts.length + 1,
          alert_type: 'Suspicious Activity',
          severity: 'high',
          message: `Suspicious transaction from ${newTransaction.location}`,
          timestamp: new Date().toISOString(),
          resolved: false,
          transaction_id: newTransaction.transaction_id
        };
        setAlerts(prev => [newAlert, ...prev]);
        showNotification(`⚠️ Alert: Suspicious transaction detected`, 'warning');
      }
      
      // Update agent tasks counter
      setAgents(prev => prev.map(agent => ({
        ...agent,
        tasks_processed: agent.tasks_processed + (agent.status === 'running' ? Math.floor(Math.random() * 5) : 0),
        cpu_usage: Math.min(100, Math.max(0, agent.cpu_usage + (Math.random() - 0.5) * 5)),
        memory_usage: Math.min(100, Math.max(0, agent.memory_usage + (Math.random() - 0.5) * 2)),
        last_active: new Date().toISOString()
      })));
      
      setLastUpdate(new Date());
    }, 15000); // Update every 15 seconds
    
    return () => clearInterval(interval);
  }, [simulationRunning, transactions.length]);

  // Handle Agent Action
  const handleAgentAction = (agentId, action, agentName) => {
    setAgents(prev => prev.map(agent =>
      agent.id === agentId
        ? { 
            ...agent, 
            status: action === 'start' ? 'running' : action === 'stop' ? 'stopped' : 'idle',
            last_active: new Date().toISOString()
          }
        : agent
    ));
    
    // Add audit log
    const newLog = {
      id: auditLogs.length + 1,
      agent_name: agentName,
      action: `${action.toUpperCase()} Agent`,
      timestamp: new Date().toISOString(),
      details: `Agent ${agentName} was ${action}ed by user`,
      risk_level: 'low'
    };
    setAuditLogs(prev => [newLog, ...prev].slice(0, 200));
    
    showNotification(`${agentName} ${action}ed successfully`, 'success');
  };

  // Resolve Alert
  const resolveAlert = (alertId) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
    
    const newLog = {
      id: auditLogs.length + 1,
      agent_name: 'System',
      action: 'RESOLVE Alert',
      timestamp: new Date().toISOString(),
      details: `Alert ${alertId} was resolved`,
      risk_level: 'low'
    };
    setAuditLogs(prev => [newLog, ...prev].slice(0, 200));
    
    showNotification('Alert resolved', 'success');
  };

  // CSV Upload Handler (Demo)
  const handleFileSelect = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUploadCSV = () => {
    if (!selectedFile) {
      showNotification('Please select a CSV file', 'warning');
      return;
    }
    
    setUploading(true);
    setTimeout(() => {
      showNotification(`Demo: Successfully processed ${selectedFile.name}`, 'success');
      setUploadDialogOpen(false);
      setSelectedFile(null);
      setUploading(false);
      
      // Add audit log
      const newLog = {
        id: auditLogs.length + 1,
        agent_name: 'System',
        action: 'UPLOAD CSV',
        timestamp: new Date().toISOString(),
        details: `Uploaded file: ${selectedFile.name}`,
        risk_level: 'low'
      };
      setAuditLogs(prev => [newLog, ...prev].slice(0, 200));
    }, 1500);
  };

  // Export Transactions
  const exportTransactions = () => {
    const csvData = transactions.map(tx => ({
      'Transaction ID': tx.transaction_id,
      'Amount': tx.amount,
      'Type': tx.transaction_type,
      'Account': tx.account_id,
      'Risk Score': tx.risk_score,
      'Status': tx.status,
      'Timestamp': new Date(tx.timestamp).toLocaleString()
    }));
    
    const headers = Object.keys(csvData[0]).join(',');
    const rows = csvData.map(row => Object.values(row).join(',')).join('\n');
    const csv = headers + '\n' + rows;
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transactions_${new Date().toISOString().slice(0, 19)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showNotification('Transactions exported successfully', 'success');
  };

  // Helper Functions
  const showNotification = (message, severity = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const refreshData = () => {
    setLoading(true);
    setTimeout(() => {
      const newTransactions = generateMockTransactions();
      setTransactions(newTransactions);
      setRiskMetrics(calculateRiskMetrics(newTransactions));
      setAlerts(generateMockAlerts());
      showNotification('Data refreshed successfully', 'success');
      setLoading(false);
      setLastUpdate(new Date());
    }, 500);
  };

  const getRiskColor = (score) => {
    if (score > 70) return '#f44336';
    if (score > 40) return '#ff9800';
    return '#4caf50';
  };

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return '#d32f2f';
      case 'high': return '#f44336';
      case 'medium': return '#ff9800';
      case 'low': return '#4caf50';
      default: return '#9e9e9e';
    }
  };

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'running': return '#4caf50';
      case 'idle': return '#ff9800';
      case 'stopped': return '#f44336';
      default: return '#9e9e9e';
    }
  };

  const viewTransactionDetails = (transaction) => {
    setSelectedTransaction(transaction);
    setDetailsDialogOpen(true);
  };

  // Filtered transactions
  const filteredTransactions = transactions.filter(tx => {
    const matchesSearch = tx.transaction_id?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          tx.account_id?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRisk = filterRisk === 'all' ||
                       (filterRisk === 'high' && tx.risk_score > 70) ||
                       (filterRisk === 'medium' && tx.risk_score > 40 && tx.risk_score <= 70) ||
                       (filterRisk === 'low' && tx.risk_score <= 40);
    return matchesSearch && matchesRisk;
  });

  // Risk distribution data
  const riskDistributionData = [
    { name: 'High Risk', value: riskMetrics.highRiskCount, color: '#f44336' },
    { name: 'Medium Risk', value: riskMetrics.mediumRiskCount, color: '#ff9800' },
    { name: 'Low Risk', value: riskMetrics.lowRiskCount, color: '#4caf50' }
  ];

  const TabPanel = ({ children, value, index }) => (
    <div role="tabpanel" hidden={value !== index}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh', bgcolor: '#f5f5f5' }}>
      {/* App Bar */}
      <AppBar position="static" sx={{ bgcolor: '#1a237e' }}>
        <Toolbar>
          <SecurityIcon sx={{ mr: 2 }} />
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            AI-Powered Risk Prediction & Continuous Auditing Console
          </Typography>
          
          <Button 
            variant="contained" 
            startIcon={<CloudUploadIcon />}
            onClick={() => setUploadDialogOpen(true)}
            sx={{ mr: 2, bgcolor: '#4caf50', '&:hover': { bgcolor: '#388e3c' } }}
          >
            Upload CSV
          </Button>
          
          <Button 
            variant="outlined" 
            startIcon={<DownloadIcon />}
            onClick={exportTransactions}
            sx={{ mr: 2, color: 'white', borderColor: 'white' }}
          >
            Export
          </Button>
          
          <Chip 
            label={`Live: ${lastUpdate.toLocaleTimeString()}`}
            size="small"
            sx={{ mr: 2, bgcolor: 'rgba(76,175,80,0.2)', color: 'white' }}
          />
          
          <FormControlLabel
            control={
              <Switch 
                checked={simulationRunning} 
                onChange={(e) => setSimulationRunning(e.target.checked)} 
              />
            }
            label="Live Updates"
            sx={{ mr: 2, color: 'white' }}
          />
          
          <IconButton color="inherit" onClick={refreshData}>
            <RefreshIcon />
          </IconButton>
          <IconButton color="inherit">
            <Badge badgeContent={alerts.filter(a => !a.resolved).length} color="error">
              <NotificationsIcon />
            </Badge>
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', p: 3 }}>
        {/* KPI Cards */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Box>
                    <Typography color="textSecondary" gutterBottom>Overall Risk Score</Typography>
                    <Typography variant="h3" sx={{ color: getRiskColor(riskMetrics.overallRisk), fontWeight: 'bold' }}>
                      {riskMetrics.overallRisk?.toFixed(1) || 0}%
                    </Typography>
                  </Box>
                  <TrendingUpIcon sx={{ fontSize: 50, color: getRiskColor(riskMetrics.overallRisk) }} />
                </Box>
                <LinearProgress variant="determinate" value={riskMetrics.overallRisk || 0} sx={{ mt: 2, height: 10, borderRadius: 5 }} />
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Box>
                    <Typography color="textSecondary" gutterBottom>Active Alerts</Typography>
                    <Typography variant="h3">{alerts.filter(a => !a.resolved).length}</Typography>
                  </Box>
                  <WarningIcon sx={{ fontSize: 50, color: '#ff9800' }} />
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Box>
                    <Typography color="textSecondary" gutterBottom>Active Agents</Typography>
                    <Typography variant="h3">{agents.filter(a => a.status === 'running').length}/{agents.length}</Typography>
                  </Box>
                  <MemoryIcon sx={{ fontSize: 50, color: '#888' }} />
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Box>
                    <Typography color="textSecondary" gutterBottom>Total Volume</Typography>
                    <Typography variant="h3">${(riskMetrics.totalAmount / 1000).toFixed(0)}K</Typography>
                  </Box>
                  <AssessmentIcon sx={{ fontSize: 50, color: '#888' }} />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Tabs */}
        <Paper sx={{ width: '100%' }}>
          <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)} sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tab icon={<DashboardIcon />} label="Dashboard" />
            <Tab icon={<SecurityIcon />} label={`Agents (${agents.length})`} />
            <Tab icon={<WarningIcon />} label={`Alerts (${alerts.filter(a => !a.resolved).length})`} />
            <Tab icon={<TimelineIcon />} label="Audit Trail" />
            <Tab icon={<AssessmentIcon />} label="Transactions" />
          </Tabs>

          {/* Dashboard Tab */}
          <TabPanel value={activeTab} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Real-Time Risk Trend</Typography>
                    <ResponsiveContainer width="100%" height={350}>
                      <AreaChart data={realTimeData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip />
                        <Area type="monotone" dataKey="risk" stroke="#8884d8" fill="#8884d8" fillOpacity={0.3} name="Risk Score" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Risk Distribution</Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie 
                          data={riskDistributionData} 
                          cx="50%" 
                          cy="50%" 
                          labelLine={true} 
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`} 
                          outerRadius={100} 
                          dataKey="value"
                        >
                          {riskDistributionData.map((entry, index) => (
                            <Cell key={index} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Agents Tab - FIXED AND WORKING */}
          <TabPanel value={activeTab} index={1}>
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
                <CircularProgress />
                <Typography sx={{ ml: 2 }}>Loading agents...</Typography>
              </Box>
            ) : (
              <Grid container spacing={3}>
                {agents.map((agent) => (
                  <Grid item xs={12} md={6} lg={4} key={agent.id}>
                    <Card sx={{ '&:hover': { boxShadow: 6, transition: '0.3s' } }}>
                      <CardContent>
                        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                          <Box display="flex" alignItems="center">
                            <Avatar sx={{ bgcolor: getStatusColor(agent.status), width: 56, height: 56, mr: 2 }}>
                              <SecurityIcon />
                            </Avatar>
                            <Box>
                              <Typography variant="h6">{agent.name}</Typography>
                              <Chip 
                                label={agent.status?.toUpperCase()} 
                                size="small" 
                                sx={{ 
                                  bgcolor: getStatusColor(agent.status), 
                                  color: 'white',
                                  fontWeight: 'bold',
                                  mt: 0.5
                                }} 
                              />
                            </Box>
                          </Box>
                          <Box>
                            <IconButton 
                              size="small" 
                              onClick={() => handleAgentAction(agent.id, 'start', agent.name)} 
                              disabled={agent.status === 'running'}
                              sx={{ 
                                bgcolor: agent.status === 'running' ? 'grey.300' : '#4caf50',
                                color: 'white',
                                mr: 1,
                                '&:hover': { bgcolor: '#388e3c' },
                                '&.Mui-disabled': { bgcolor: '#ccc' }
                              }}
                            >
                              <PlayIcon />
                            </IconButton>
                            <IconButton 
                              size="small" 
                              onClick={() => handleAgentAction(agent.id, 'stop', agent.name)} 
                              disabled={agent.status !== 'running'}
                              sx={{ 
                                bgcolor: agent.status !== 'running' ? 'grey.300' : '#f44336',
                                color: 'white',
                                '&:hover': { bgcolor: '#d32f2f' },
                                '&.Mui-disabled': { bgcolor: '#ccc' }
                              }}
                            >
                              <StopIcon />
                            </IconButton>
                          </Box>
                        </Box>
                        
                        <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                          {agent.description}
                        </Typography>
                        
                        <Divider sx={{ my: 2 }} />
                        
                        <Grid container spacing={2}>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">Tasks Processed</Typography>
                            <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                              {agent.tasks_processed?.toLocaleString() || 0}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">Uptime</Typography>
                            <Typography variant="body2" sx={{ fontWeight: 'bold' }}>{agent.uptime || 'N/A'}</Typography>
                          </Grid>
                          <Grid item xs={12}>
                            <Typography variant="body2" color="textSecondary" gutterBottom>CPU Usage</Typography>
                            <LinearProgress 
                              variant="determinate" 
                              value={agent.cpu_usage || 0} 
                              sx={{ 
                                height: 8, 
                                borderRadius: 4,
                                bgcolor: '#e0e0e0',
                                '& .MuiLinearProgress-bar': {
                                  bgcolor: agent.cpu_usage > 70 ? '#f44336' : agent.cpu_usage > 40 ? '#ff9800' : '#4caf50'
                                }
                              }} 
                            />
                            <Typography variant="caption">{(agent.cpu_usage || 0).toFixed(1)}%</Typography>
                          </Grid>
                          <Grid item xs={12}>
                            <Typography variant="body2" color="textSecondary" gutterBottom>Memory Usage</Typography>
                            <LinearProgress 
                              variant="determinate" 
                              value={agent.memory_usage || 0} 
                              sx={{ 
                                height: 8, 
                                borderRadius: 4,
                                bgcolor: '#e0e0e0',
                                '& .MuiLinearProgress-bar': {
                                  bgcolor: agent.memory_usage > 70 ? '#f44336' : agent.memory_usage > 40 ? '#ff9800' : '#4caf50'
                                }
                              }} 
                            />
                            <Typography variant="caption">{(agent.memory_usage || 0).toFixed(1)}%</Typography>
                          </Grid>
                          <Grid item xs={12}>
                            <Divider sx={{ my: 1 }} />
                            <Typography variant="caption" color="textSecondary">
                              Version: {agent.version || '1.0'} | 
                              Last active: {agent.last_active ? new Date(agent.last_active).toLocaleTimeString() : 'Never'}
                            </Typography>
                          </Grid>
                        </Grid>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            )}
          </TabPanel>

          {/* Alerts Tab */}
          <TabPanel value={activeTab} index={2}>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow sx={{ bgcolor: '#fafafa' }}>
                    <TableCell>Severity</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Message</TableCell>
                    <TableCell>Time</TableCell>
                    <TableCell>Transaction</TableCell>
                    <TableCell>Action</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {alerts.filter(a => !a.resolved).map((alert) => (
                    <TableRow key={alert.id}>
                      <TableCell>
                        <Chip 
                          label={alert.severity?.toUpperCase()} 
                          size="small" 
                          sx={{ 
                            bgcolor: getSeverityColor(alert.severity), 
                            color: 'white',
                            fontWeight: 'bold'
                          }} 
                        />
                      </TableCell>
                      <TableCell>{alert.alert_type}</TableCell>
                      <TableCell>{alert.message}</TableCell>
                      <TableCell>{new Date(alert.timestamp).toLocaleString()}</TableCell>
                      <TableCell>
                        <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                          {alert.transaction_id}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Button 
                          size="small" 
                          variant="contained" 
                          onClick={() => resolveAlert(alert.id)}
                          sx={{ bgcolor: '#4caf50', '&:hover': { bgcolor: '#388e3c' } }}
                        >
                          Resolve
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                  {alerts.filter(a => !a.resolved).length === 0 && (
                    <TableRow>
                      <TableCell colSpan={6} align="center">
                        <Typography sx={{ py: 4, color: '#4caf50' }}>
                          ✓ No active alerts. All systems operational.
                        </Typography>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>

          {/* Audit Trail Tab */}
          <TabPanel value={activeTab} index={3}>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow sx={{ bgcolor: '#fafafa' }}>
                    <TableCell>Time</TableCell>
                    <TableCell>Agent</TableCell>
                    <TableCell>Action</TableCell>
                    <TableCell>Details</TableCell>
                    <TableCell>Risk Level</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {auditLogs.slice().reverse().slice(0, 100).map((log) => (
                    <TableRow key={log.id}>
                      <TableCell>{new Date(log.timestamp).toLocaleString()}</TableCell>
                      <TableCell>{log.agent_name}</TableCell>
                      <TableCell>
                        <Chip 
                          label={log.action} 
                          size="small" 
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell>{log.details}</TableCell>
                      <TableCell>
                        <Chip 
                          label={log.risk_level} 
                          size="small" 
                          sx={{ 
                            bgcolor: getSeverityColor(log.risk_level), 
                            color: 'white',
                            minWidth: 60
                          }} 
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>

          {/* Transactions Tab */}
          <TabPanel value={activeTab} index={4}>
            <Box sx={{ mb: 2, display: 'flex', gap: 2 }}>
              <TextField 
                size="small" 
                placeholder="Search by ID or Account" 
                value={searchTerm} 
                onChange={(e) => setSearchTerm(e.target.value)} 
                sx={{ flexGrow: 1 }} 
                InputProps={{ 
                  startAdornment: <InputAdornment position="start"><SearchIcon /></InputAdornment> 
                }} 
              />
              <FormControl size="small" sx={{ minWidth: 150 }}>
                <InputLabel>Risk Level</InputLabel>
                <Select value={filterRisk} onChange={(e) => setFilterRisk(e.target.value)} label="Risk Level">
                  <MenuItem value="all">All Risks</MenuItem>
                  <MenuItem value="high">High Risk (&gt;70%)</MenuItem>
                  <MenuItem value="medium">Medium Risk (40-70%)</MenuItem>
                  <MenuItem value="low">Low Risk (&lt;40%)</MenuItem>
                </Select>
              </FormControl>
            </Box>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow sx={{ bgcolor: '#fafafa' }}>
                    <TableCell>ID</TableCell>
                    <TableCell>Amount</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Account</TableCell>
                    <TableCell>Risk Score</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Time</TableCell>
                    <TableCell>Action</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredTransactions.slice(0, 50).map((tx) => (
                    <TableRow key={tx.id} sx={{ '&:hover': { bgcolor: '#f5f5f5' } }}>
                      <TableCell>
                        <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                          {tx.transaction_id?.slice(0, 12)}...
                        </Typography>
                      </TableCell>
                      <TableCell sx={{ fontWeight: 'bold' }}>
                        ${tx.amount?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </TableCell>
                      <TableCell>{tx.transaction_type}</TableCell>
                      <TableCell>{tx.account_id}</TableCell>
                      <TableCell>
                        <Chip 
                          label={`${tx.risk_score?.toFixed(1)}%`} 
                          size="small" 
                          sx={{ 
                            bgcolor: getRiskColor(tx.risk_score), 
                            color: 'white', 
                            minWidth: 60,
                            fontWeight: 'bold'
                          }} 
                        />
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={tx.status} 
                          size="small" 
                          sx={{ 
                            bgcolor: tx.status === 'flagged' ? '#f44336' : tx.status === 'approved' ? '#4caf50' : '#ff9800', 
                            color: 'white',
                            textTransform: 'capitalize'
                          }} 
                        />
                      </TableCell>
                      <TableCell>{new Date(tx.timestamp).toLocaleString()}</TableCell>
                      <TableCell>
                        <IconButton 
                          size="small" 
                          onClick={() => viewTransactionDetails(tx)}
                          sx={{ color: '#1a237e' }}
                        >
                          <VisibilityIcon />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>
        </Paper>
      </Box>

      {/* Transaction Details Dialog */}
      <Dialog open={detailsDialogOpen} onClose={() => setDetailsDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle sx={{ bgcolor: '#1a237e', color: 'white' }}>
          Transaction Details
        </DialogTitle>
        <DialogContent>
          {selectedTransaction && (
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="textSecondary">Transaction ID</Typography>
                <Typography variant="body1" sx={{ fontFamily: 'monospace', fontWeight: 'bold' }}>
                  {selectedTransaction.transaction_id}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="textSecondary">Amount</Typography>
                <Typography variant="h5" sx={{ color: getRiskColor(selectedTransaction.risk_score), fontWeight: 'bold' }}>
                  ${selectedTransaction.amount?.toLocaleString()}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="textSecondary">Transaction Type</Typography>
                <Typography variant="body1">{selectedTransaction.transaction_type}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="textSecondary">Account ID</Typography>
                <Typography variant="body1">{selectedTransaction.account_id}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="textSecondary">Risk Score</Typography>
                <Chip 
                  label={`${selectedTransaction.risk_score?.toFixed(1)}%`} 
                  sx={{ 
                    bgcolor: getRiskColor(selectedTransaction.risk_score), 
                    color: 'white',
                    fontSize: '1rem',
                    fontWeight: 'bold'
                  }} 
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="textSecondary">Status</Typography>
                <Chip 
                  label={selectedTransaction.status} 
                  sx={{ 
                    bgcolor: selectedTransaction.status === 'flagged' ? '#f44336' : 
                             selectedTransaction.status === 'approved' ? '#4caf50' : '#ff9800', 
                    color: 'white'
                  }} 
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="textSecondary">Location</Typography>
                <Typography variant="body1">{selectedTransaction.location || 'N/A'}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="textSecondary">Device</Typography>
                <Typography variant="body1">{selectedTransaction.device || 'N/A'}</Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="body2" color="textSecondary">Timestamp</Typography>
                <Typography variant="body1">{new Date(selectedTransaction.timestamp).toLocaleString()}</Typography>
              </Grid>
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsDialogOpen(false)} variant="contained">
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* CSV Upload Dialog */}
      <Dialog open={uploadDialogOpen} onClose={() => setUploadDialogOpen(false)}>
        <DialogTitle>Upload CSV File</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
            Upload a CSV file with transaction data.<br/>
            <strong>Supported columns:</strong> transaction_id, amount, timestamp, account_id, transaction_type
          </Typography>
          <input 
            type="file" 
            accept=".csv" 
            onChange={handleFileSelect} 
            style={{ width: '100%', marginTop: '10px' }} 
          />
          {selectedFile && (
            <Typography variant="caption" sx={{ mt: 2, display: 'block', color: '#4caf50' }}>
              ✓ Selected: {selectedFile.name}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setUploadDialogOpen(false);
            setSelectedFile(null);
          }}>Cancel</Button>
          <Button 
            variant="contained" 
            onClick={handleUploadCSV} 
            disabled={!selectedFile || uploading}
            sx={{ bgcolor: '#4caf50', '&:hover': { bgcolor: '#388e3c' } }}
          >
            {uploading ? <CircularProgress size={24} /> : 'Upload'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for Notifications */}
      <Snackbar 
        open={snackbar.open} 
        autoHideDuration={6000} 
        onClose={() => setSnackbar({ ...snackbar, open: false })} 
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          severity={snackbar.severity} 
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          variant="filled"
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default AgentConsole;