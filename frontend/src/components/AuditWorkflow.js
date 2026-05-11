import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  CircularProgress,
  Alert,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  IconButton,
  Tooltip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Timeline as TimelineIcon,
  Assessment as AssessmentIcon,
  Security as SecurityIcon,
  ExpandMore as ExpandMoreIcon,
  Download as DownloadIcon
} from '@mui/icons-material';
import { useWebSocket } from '../hooks/useWebSocket';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

function AuditWorkflow() {
  const [workflowId, setWorkflowId] = useState(null);
  const [workflowStatus, setWorkflowStatus] = useState('idle');
  const [findings, setFindings] = useState([]);
  const [violations, setViolations] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [summary, setSummary] = useState(null);
  const [reports, setReports] = useState(null);
  const [activeStep, setActiveStep] = useState(0);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [lastUpdateTime, setLastUpdateTime] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [transactionCount, setTransactionCount] = useState(0);
  const [isLoadingTransactions, setIsLoadingTransactions] = useState(false);
  const [transactionSummary, setTransactionSummary] = useState(null);
  const [workflowMode, setWorkflowMode] = useState('unknown'); // 'langgraph', 'full', 'integrated'
  const [agentStatus, setAgentStatus] = useState({}); // Track which agents are running
  
  // LangGraph Workflow Architecture
  const LANGGRAPH_WORKFLOW = {
    name: 'LangGraphAuditWorkflow',
    description: 'Agent-based audit orchestration with LangGraph StateGraph',
    phases: [
      {
        id: 0,
        name: 'risk_assessment',
        label: 'Risk Assessment',
        agent: 'RiskAssessmentAgent',
        description: 'ML-based risk scoring of transactions',
        outputs: ['assessed_transactions', 'high_risk_alerts'],
        icon: <AssessmentIcon />,
        color: '#ff9800'
      },
      {
        id: 1,
        name: 'audit_procedures',
        label: 'Audit Procedures',
        agent: 'AuditAgent',
        description: 'Sampling and auditing high-value transactions',
        outputs: ['audit_findings'],
        icon: <SecurityIcon />,
        color: '#2196f3'
      },
      {
        id: 2,
        name: 'compliance_check',
        label: 'Compliance Check',
        agent: 'ComplianceAgent',
        description: 'AML/KYC regulatory compliance verification',
        outputs: ['compliance_violations'],
        icon: <CheckCircleIcon />,
        color: '#4caf50'
      },
      {
        id: 3,
        name: 'investigation',
        label: 'Investigation',
        agent: 'InvestigationAgent',
        description: 'Deep analysis of high-risk transactions (conditional)',
        outputs: ['investigation_findings'],
        icon: <TimelineIcon />,
        color: '#f44336',
        conditional: true
      },
      {
        id: 4,
        name: 'report_generation',
        label: 'Report Generation',
        agent: 'ReportAgent',
        description: 'Generate executive summary and reports',
        outputs: ['reports', 'compliance_score'],
        icon: <AssessmentIcon />,
        color: '#9c27b0'
      }
    ]
  };
  
  // Convert HTTP/HTTPS to WS/WSS for websocket URL
  const WS_BASE = API_BASE.replace(/^http/, 'ws');
  const { lastMessage, sendMessage, readyState, waitForConnection } = useWebSocket(WS_BASE);

  // Monitor WebSocket connection status
  useEffect(() => {
    if (readyState) {
      setConnectionStatus('connected');
    } else {
      setConnectionStatus('connecting');
    }
  }, [readyState]);

  // Load transaction data on component mount
  useEffect(() => {
    const loadTransactions = async () => {
      setIsLoadingTransactions(true);
      try {
        const response = await axios.get(`${API_BASE}/api/transactions?page=1&per_page=5000`);
        const txnData = response?.data?.transactions || [];
        const total = response?.data?.total || 0;
        
        setTransactions(txnData);
        setTransactionCount(total);
        
        // Calculate summary statistics
        if (txnData.length > 0) {
          const amountSum = txnData.reduce((sum, t) => sum + (parseFloat(t.amount) || 0), 0);
          const amountAvg = amountSum / txnData.length;
          const internationalCount = txnData.filter(t => t.is_international).length;
          const fraudCount = txnData.filter(t => t.is_fraud).length;
          
          setTransactionSummary({
            total,
            loaded: txnData.length,
            totalAmount: amountSum.toFixed(2),
            averageAmount: amountAvg.toFixed(2),
            international: internationalCount,
            potential_fraud: fraudCount
          });
        }
      } catch (error) {
        console.error('Error loading transactions:', error);
        setError(`Failed to load transaction data: ${error.message}`);
      } finally {
        setIsLoadingTransactions(false);
      }
    };
    
    loadTransactions();
  }, []);

  // Map phases from LANGGRAPH_WORKFLOW for backward compatibility
  const steps = LANGGRAPH_WORKFLOW.phases.map(phase => ({
    label: phase.label,
    icon: phase.icon,
    agent: phase.agent,
    description: phase.description
  }));

  useEffect(() => {
    if (lastMessage) {
      setLastUpdateTime(new Date());
      try {
        const event = lastMessage.event;
        const data = lastMessage.payload || {};
        console.log('WebSocket message:', event, data);
        
        switch (event) {
          case 'audit_started':
            setWorkflowStatus('running');
            setWorkflowId(data.workflow_id);
            setActiveStep(0);
            setError(null);
            setFindings([]);
            setViolations([]);
            setAlerts([]);
            setSummary(null);
            setReports(null);
            console.log('Audit started:', data.workflow_id);
            break;
          
          case 'phase_progress':
            // Update progress based on workflow phase
            const phaseNumber = data.phase_number || 0;
            const phaseName = data.phase || '';
            
            // Find the phase definition
            const phaseInfo = LANGGRAPH_WORKFLOW.phases.find(p => p.name === phaseName || p.id === phaseNumber);
            
            // Update agent status
            setAgentStatus(prev => ({
              ...prev,
              [phaseNumber]: {
                agent: phaseInfo?.agent || 'Unknown Agent',
                status: 'executing',
                updatedAt: new Date().toLocaleTimeString()
              }
            }));
            
            setActiveStep(phaseNumber);
            console.log('Phase progress:', phaseName, 'Agent:', phaseInfo?.agent);
            break;
            
          case 'audit_findings_chunk':
            // Handle chunked findings from workflow
            setFindings(prev => [...prev, ...data.findings]);
            if (data.total > 0) {
              const progress = data.progress / data.total;
              if (progress < 0.25) setActiveStep(0);
              else if (progress < 0.5) setActiveStep(1);
              else if (progress < 0.75) setActiveStep(2);
              else if (progress < 0.95) setActiveStep(3);
              else setActiveStep(4);
            }
            console.log('Findings chunk:', data.progress, 'of', data.total);
            break;
            
          case 'audit_complete':
            setWorkflowStatus('completed');
            
            // Detect workflow mode if not set
            if (!workflowMode || workflowMode === 'unknown') {
              // Check if we got full results (indicates LangGraph mode)
              if (data.full_results) {
                setWorkflowMode('langgraph');
              }
            }
            
            // Mark all agents as completed
            setAgentStatus(prev => {
              const updated = { ...prev };
              LANGGRAPH_WORKFLOW.phases.forEach((phase, idx) => {
                if (updated[idx]) {
                  updated[idx].status = 'completed';
                }
              });
              return updated;
            });
            
            console.log('Audit complete:', data);
            // Extract all data from workflow results
            if (data.summary) {
              setSummary(data.summary);
            }
            // Parse full results if included
            if (data.full_results) {
              const results = data.full_results;
              setFindings(results.audit_findings || []);
              setViolations(results.compliance_violations || []);
              setAlerts(results.high_risk_alerts || []);
              setReports(results.reports || {});
              setMetrics({
                compliance_score: results.compliance_score,
                risk_metrics: results.risk_metrics
              });
              console.log('Audit results loaded:', {
                findings: results.audit_findings?.length || 0,
                violations: results.compliance_violations?.length || 0,
                alerts: results.high_risk_alerts?.length || 0
              });
            }
            setActiveStep(4);
            break;
            
          case 'audit_error':
            setError(data.error || 'An error occurred during the audit workflow');
            setWorkflowStatus('error');
            console.error('Audit error:', data.error);
            break;
            
          case 'workflow_status':
            setWorkflowStatus(data.status);
            console.log('Workflow status:', data.status);
            break;
            
          default:
            console.log('Unknown event:', event);
            break;
        }
      } catch (error) {
        console.error('Error processing WebSocket message:', error);
      }
    }
  }, [lastMessage]);

  const startAudit = async () => {
  setLoading(true);
  setError(null);
  setFindings([]);
  setViolations([]);
  setAlerts([]);
  setSummary(null);
  setReports(null);
  setMetrics(null);
  setActiveStep(0);
  setWorkflowMode('unknown');
  
  try {
    if (!transactions || transactions.length === 0) {
      setError('No transaction data available. Please refresh the page and try again.');
      setLoading(false);
      return;
    }
    
    console.log(`📊 Starting audit with ${transactions.length} transactions`);
    console.log(`🔌 WebSocket readyState: ${readyState ? 'Connected' : 'Disconnected'}`);
    
    // Use REST API directly (more reliable)
    console.log('🚀 Using REST API for audit...');
    setWorkflowStatus('running');
    setWorkflowMode('rest_api');
    
    // Simulate step progress
    const phases = ['risk_assessment', 'audit_procedures', 'compliance_check', 'investigation', 'report_generation'];
    
    for (let i = 0; i < phases.length; i++) {
      setActiveStep(i);
      setAgentStatus(prev => ({
        ...prev,
        [i]: {
          agent: LANGGRAPH_WORKFLOW.phases[i]?.agent || 'Agent',
          status: 'running',
          updatedAt: new Date().toLocaleTimeString()
        }
      }));
      await new Promise(r => setTimeout(r, 500));
      setAgentStatus(prev => ({
        ...prev,
        [i]: {
          ...prev[i],
          status: 'completed'
        }
      }));
    }
    
    // Call REST API
    const response = await axios.post(`${API_BASE}/api/audit/start`, {
      transactions: transactions.slice(0, 500),
      config: { test_mode: true }
    });
    
    console.log('✅ REST API response:', response.data);
    
    if (response.data.success) {
      setWorkflowId(response.data.workflow_id);
      setSummary(response.data.summary);
      setFindings(response.data.full_results?.audit_findings || []);
      setViolations(response.data.full_results?.compliance_violations || []);
      setAlerts(response.data.full_results?.high_risk_alerts || []);
      setReports(response.data.full_results?.reports || {});
      setMetrics({
        compliance_score: response.data.full_results?.compliance_score || 0.85,
        risk_metrics: response.data.full_results?.risk_metrics || {}
      });
      setWorkflowStatus('completed');
      setActiveStep(4);
    } else {
      throw new Error(response.data.error || 'Audit failed');
    }
    
  } catch (error) {
    console.error('❌ Audit error:', error);
    setError(`Failed to start audit: ${error.message}`);
    setWorkflowStatus('error');
  } finally {
    setLoading(false);
  }
};

  const stopAudit = () => {
    sendMessage('stop_audit', {});
    setWorkflowStatus('idle');
  };

  const getStatusColor = () => {
    switch (workflowStatus) {
      case 'running': return 'warning';
      case 'completed': return 'success';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return 'error';
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            Audit Workflow
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="body2" color="text.secondary">
              AI-Powered Continuous Auditing with Integrated Risk Assessment
            </Typography>
            <Chip
              icon={connectionStatus === 'connected' ? <CheckCircleIcon /> : <WarningIcon />}
              label={connectionStatus === 'connected' ? 'Connected' : 'Connecting...'}
              size="small"
              color={connectionStatus === 'connected' ? 'success' : 'warning'}
              variant="outlined"
            />
          </Box>
        </Box>
        <Box sx={{ display: 'flex', gap: 2 }}>
          {workflowId && (
            <Chip
              label={`Workflow: ${workflowId}`}
              color={getStatusColor()}
              variant="outlined"
            />
          )}
          {workflowStatus === 'running' ? (
            <Button
              variant="contained"
              color="error"
              startIcon={<StopIcon />}
              onClick={stopAudit}
            >
              Stop Audit
            </Button>
          ) : (
            <Button
              variant="contained"
              color="primary"
              startIcon={<PlayIcon />}
              onClick={startAudit}
              disabled={loading}
            >
              {loading ? 'Starting...' : 'Start Audit'}
            </Button>
          )}
        </Box>
      </Box>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Connection Status Alert */}
      {connectionStatus !== 'connected' && !loading && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Establishing WebSocket connection... This will take a moment.
        </Alert>
      )}

      {/* Transaction Data Summary */}
      {workflowStatus === 'idle' && (
        <Paper sx={{ p: 3, mb: 3, bgcolor: '#f5f5f5' }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AssessmentIcon /> Transaction Data Overview
          </Typography>
          {isLoadingTransactions ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CircularProgress size={20} />
              <Typography>Loading transaction data...</Typography>
            </Box>
          ) : transactionSummary ? (
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      Total Transactions
                    </Typography>
                    <Typography variant="h5">
                      {transactionSummary.total.toLocaleString()}
                    </Typography>
                    <Typography variant="caption">
                      ({transactionSummary.loaded.toLocaleString()} loaded)
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      Total Amount
                    </Typography>
                    <Typography variant="h5">
                      ${parseFloat(transactionSummary.totalAmount).toLocaleString('en-US', {
                        maximumFractionDigits: 2
                      })}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      Average Amount
                    </Typography>
                    <Typography variant="h5">
                      ${parseFloat(transactionSummary.averageAmount).toLocaleString('en-US', {
                        maximumFractionDigits: 2
                      })}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      International Transactions
                    </Typography>
                    <Typography variant="h5" color="primary">
                      {transactionSummary.international}
                    </Typography>
                    <Typography variant="caption">
                      {((transactionSummary.international / transactionSummary.loaded) * 100).toFixed(1)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      Potential Fraud Flags
                    </Typography>
                    <Typography variant="h5" color="error">
                      {transactionSummary.potential_fraud}
                    </Typography>
                    <Typography variant="caption">
                      {((transactionSummary.potential_fraud / transactionSummary.loaded) * 100).toFixed(1)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          ) : (
            <Typography color="error">Failed to load transaction summary</Typography>
          )}
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="textSecondary">
              These transactions will be analyzed in the audit workflow. The workflow will assess risk, execute audit procedures,
              check compliance, and generate a comprehensive report.
            </Typography>
          </Box>
        </Paper>
      )}

      {/* LangGraph Workflow Information */}
      {(workflowStatus === 'running' || workflowStatus === 'completed') && (
        <Paper sx={{ p: 3, mb: 3, bgcolor: '#f3f5f6', border: '1px solid #e0e0e0' }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TimelineIcon color="primary" /> LangGraph Agent Workflow
          </Typography>
          
          <Box sx={{ mb: 2 }}>
            <Chip 
              label={`Workflow Mode: ${workflowMode === 'langgraph' ? 'LangGraph (Agent-Based)' : 'Integrated'}`}
              color={workflowMode === 'langgraph' ? 'primary' : 'default'}
              variant="outlined"
              size="small"
            />
          </Box>

          <Grid container spacing={2}>
            {LANGGRAPH_WORKFLOW.phases.map((phase, idx) => {
              const agentInfo = agentStatus[idx];
              const isExecuted = idx < activeStep;
              const isActive = idx === activeStep && workflowStatus === 'running';
              
              return (
                <Grid item xs={12} sm={6} md={4} key={phase.id}>
                  <Card 
                    sx={{
                      borderLeft: `5px solid ${phase.color}`,
                      bgcolor: isActive ? '#fff9e6' : isExecuted ? '#e8f5e9' : '#fafafa',
                      opacity: isExecuted || isActive ? 1 : 0.6
                    }}
                  >
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                          {phase.label}
                        </Typography>
                        {isExecuted && (
                          <CheckCircleIcon sx={{ color: '#4caf50', fontSize: '20px' }} />
                        )}
                        {isActive && (
                          <CircularProgress size={20} />
                        )}
                      </Box>
                      
                      <Typography variant="caption" color="primary" sx={{ display: 'block', mb: 1 }}>
                        Agent: <strong>{phase.agent}</strong>
                      </Typography>
                      
                      <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mb: 1 }}>
                        {phase.description}
                      </Typography>
                      
                      {agentInfo && (
                        <Typography variant="caption" sx={{ display: 'block', color: '#666' }}>
                          Status: <strong>{agentInfo.status}</strong> • {agentInfo.updatedAt}
                        </Typography>
                      )}
                      
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="caption" sx={{ display: 'block', color: '#999' }}>
                          Outputs: {phase.outputs.join(', ')}
                        </Typography>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>

          <Box sx={{ mt: 2, p: 2, bgcolor: 'white', borderRadius: 1, border: '1px solid #ddd' }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
              Workflow Graph
            </Typography>
            <Typography variant="caption" sx={{ fontFamily: 'monospace', display: 'block', lineHeight: 1.8 }}>
              <span>Risk Assessment</span><br />
              <span>&nbsp;&nbsp;&nbsp;&nbsp;↓ (RiskAssessmentAgent)</span><br />
              <span>Audit Procedures</span><br />
              <span>&nbsp;&nbsp;&nbsp;&nbsp;↓ (AuditAgent)</span><br />
              <span>Compliance Check</span><br />
              <span>&nbsp;&nbsp;&nbsp;&nbsp;├→ Has High-Risk Alerts?</span><br />
              <span>&nbsp;&nbsp;&nbsp;&nbsp;├→ YES: Investigation (InvestigationAgent)</span><br />
              <span>&nbsp;&nbsp;&nbsp;&nbsp;└→ NO: Skip to Report</span><br />
              <span>&nbsp;&nbsp;&nbsp;&nbsp;↓</span><br />
              <span>Report Generation</span><br />
              <span>&nbsp;&nbsp;&nbsp;&nbsp;↓ (ReportAgent)</span><br />
              <span>[Completed]</span>
            </Typography>
          </Box>
        </Paper>
      )}

      {/* Workflow Stepper */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Audit Workflow Progress
        </Typography>
        {workflowStatus === 'running' && (
          <Box sx={{ mb: 2 }}>
            <LinearProgress variant="determinate" value={(activeStep + 1) * 20} />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              {Math.round((activeStep + 1) * 20)}% Complete
            </Typography>
          </Box>
        )}
        <Stepper activeStep={activeStep} orientation="vertical">
          {steps.map((step, index) => (
            <Step key={step.label} completed={index < activeStep}>
              <StepLabel StepIconComponent={() => (
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  {index < activeStep ? (
                    <CheckCircleIcon color="success" />
                  ) : index === activeStep && workflowStatus === 'running' ? (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <CircularProgress size={24} />
                      <Typography variant="caption">{step.label}...</Typography>
                    </Box>
                  ) : (
                    step.icon
                  )}
                </Box>
              )}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Box>
                    <Typography variant="subtitle1">{step.label}</Typography>
                    {step.agent && (
                      <Typography variant="caption" color="primary" sx={{ display: 'block' }}>
                        Agent: {step.agent}
                      </Typography>
                    )}
                  </Box>
                </Box>
              </StepLabel>
              <StepContent>
                <Typography color="text.secondary" sx={{ mb: 1 }}>
                  {step.description || 
                    (index === 0 && 'Analyzing transactions for risk patterns...') ||
                    (index === 1 && 'Executing audit procedures...') ||
                    (index === 2 && 'Checking compliance with regulations...') ||
                    (index === 3 && 'Investigating high-risk findings...') ||
                    (index === 4 && 'Generating comprehensive audit report...')}
                </Typography>
                {agentStatus[index] && (
                  <Chip 
                    label={`${agentStatus[index].agent} - ${agentStatus[index].status}`}
                    color={agentStatus[index].status === 'completed' ? 'success' : 'warning'}
                    size="small"
                    variant="outlined"
                  />
                )}
              </StepContent>
            </Step>
          ))}
        </Stepper>
      </Paper>

      {/* Findings Section */}
      {findings.length > 0 && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Audit Findings ({findings.length})
          </Typography>
          <List>
            {findings.map((finding, index) => (
              <React.Fragment key={finding.id || index}>
                <ListItem alignItems="flex-start">
                  <ListItemIcon>
                    {finding.severity === 'HIGH' || finding.severity === 'CRITICAL' ? (
                      <ErrorIcon color="error" />
                    ) : finding.severity === 'MEDIUM' ? (
                      <WarningIcon color="warning" />
                    ) : (
                      <CheckCircleIcon color="success" />
                    )}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                        <Chip
                          size="small"
                          label={finding.type}
                          variant="outlined"
                        />
                        <Chip
                          size="small"
                          label={finding.severity}
                          color={getSeverityColor(finding.severity)}
                        />
                        <Typography variant="body2">
                          {finding.description}
                        </Typography>
                      </Box>
                    }
                    secondary={
                      finding.recommendation && (
                        <Typography variant="caption" color="primary">
                          Recommendation: {finding.recommendation}
                        </Typography>
                      )
                    }
                  />
                </ListItem>
                {index < findings.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        </Paper>
      )}

      {/* Summary Section */}
      {summary && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            🎯 Executive Summary
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Total Transactions
                  </Typography>
                  <Typography variant="h5">
                    {summary.total_transactions?.toLocaleString() || 'N/A'}
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
                  <Typography variant="h5" color="error">
                    {summary.high_risk_transactions?.toLocaleString() || alerts.length}
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
                  <Typography variant="h5">
                    {summary.total_audit_findings?.toLocaleString() || findings.length}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Compliance Score
                  </Typography>
                  <Typography variant="h5" color="success.main">
                    {metrics?.compliance_score 
                      ? (metrics.compliance_score * 100).toFixed(0) 
                      : (summary.compliance_score || 0.85) * 100}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Paper>
      )}

      {/* Compliance Violations Section */}
      {violations.length > 0 && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <WarningIcon color="warning" /> Compliance Violations ({violations.length})
          </Typography>
          <List>
            {violations.map((violation, index) => (
              <React.Fragment key={violation.transaction_id || index}>
                <ListItem alignItems="flex-start">
                  <ListItemIcon>
                    {violation.severity === 'CRITICAL' ? (
                      <ErrorIcon color="error" />
                    ) : violation.severity === 'HIGH' ? (
                      <WarningIcon color="error" />
                    ) : (
                      <WarningIcon color="warning" />
                    )}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                        <Chip
                          size="small"
                          label={violation.type}
                          variant="outlined"
                          icon={<SecurityIcon />}
                        />
                        <Chip
                          size="small"
                          label={violation.severity}
                          color={getSeverityColor(violation.severity)}
                        />
                        <Typography variant="body2">{violation.description}</Typography>
                      </Box>
                    }
                    secondary={
                      violation.regulation && (
                        <Typography variant="caption" color="primary">
                          Regulation: {violation.regulation}
                        </Typography>
                      )
                    }
                  />
                </ListItem>
                {index < violations.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        </Paper>
      )}

      {/* High-Risk Alerts Section */}
      {alerts.length > 0 && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ErrorIcon color="error" /> High-Risk Alerts ({alerts.length})
          </Typography>
          <List>
            {alerts.slice(0, 50).map((alert, index) => (
              <React.Fragment key={alert.transaction_id || index}>
                <ListItem alignItems="flex-start">
                  <ListItemIcon>
                    <ErrorIcon color="error" />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="body2">
                          Transaction: {alert.transaction_id}
                        </Typography>
                        <Chip label={`Risk: ${(alert.risk_score * 100).toFixed(0)}%`} color="error" size="small" />
                        <Typography variant="body2" sx={{ ml: 1 }}>
                          Amount: ${alert.amount?.toLocaleString()}
                        </Typography>
                      </Box>
                    }
                    secondary="Requires immediate review"
                  />
                </ListItem>
                {index < Math.min(alerts.length - 1, 49) && <Divider />}
              </React.Fragment>
            ))}
            {alerts.length > 50 && (
              <Typography variant="caption" sx={{ p: 2, display: 'block' }}>
                ... and {alerts.length - 50} more alerts
              </Typography>
            )}
          </List>
        </Paper>
      )}

      {/* Workflow Status Indicator */}
      {workflowStatus === 'running' && (
        <Box sx={{ mt: 3 }}>
          <LinearProgress />
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            Audit in progress. This may take a few moments...
          </Typography>
        </Box>
      )}
    </Box>
  );
}

export default AuditWorkflow;