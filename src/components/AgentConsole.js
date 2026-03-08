import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  Chip,
  IconButton,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Badge,
  Tooltip,
  Button
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Memory as MemoryIcon,
  Psychology as PsychologyIcon,
  Security as SecurityIcon,
  Gavel as GavelIcon,
  Search as SearchIcon,
  Description as DescriptionIcon,
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Timeline as TimelineIcon
} from '@mui/icons-material';
import ForceGraph2D from 'react-force-graph-2d';
import { useWebSocket } from '../hooks/useWebSocket';

function AgentConsole() {
  const [agents, setAgents] = useState({
    risk: { status: 'idle', tasks: 0, memory: [] },
    audit: { status: 'idle', tasks: 0, memory: [] },
    compliance: { status: 'idle', tasks: 0, memory: [] },
    investigation: { status: 'idle', tasks: 0, memory: [] },
    report: { status: 'idle', tasks: 0, memory: [] }
  });
  
  const [agentGraph, setAgentGraph] = useState({ nodes: [], links: [] });
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  
  const { lastMessage, sendMessage } = useWebSocket('ws://localhost:5000');

  useEffect(() => {
    if (lastMessage) {
      try {
        const data = JSON.parse(lastMessage.data);
        if (data.type === 'AGENT_UPDATE') {
          updateAgentState(data.agent, data.state);
        } else if (data.type === 'AGENT_GRAPH_UPDATE') {
          setAgentGraph(data.graph);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    }
  }, [lastMessage]);

  const updateAgentState = (agentName, state) => {
    setAgents(prev => ({
      ...prev,
      [agentName]: {
        ...prev[agentName],
        ...state
      }
    }));
  };

  const startWorkflow = () => {
    setIsRunning(true);
    sendMessage(JSON.stringify({ type: 'START_WORKFLOW' }));
  };

  const stopWorkflow = () => {
    setIsRunning(false);
    sendMessage(JSON.stringify({ type: 'STOP_WORKFLOW' }));
  };

  const getAgentIcon = (agentType) => {
    switch (agentType) {
      case 'risk': return <PsychologyIcon />;
      case 'audit': return <SecurityIcon />;
      case 'compliance': return <GavelIcon />;
      case 'investigation': return <SearchIcon />;
      case 'report': return <DescriptionIcon />;
      default: return <MemoryIcon />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'success';
      case 'processing': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const graphData = {
    nodes: [
      { id: 'risk', name: 'Risk Agent', group: 1 },
      { id: 'audit', name: 'Audit Agent', group: 2 },
      { id: 'compliance', name: 'Compliance Agent', group: 3 },
      { id: 'investigation', name: 'Investigation Agent', group: 4 },
      { id: 'report', name: 'Report Agent', group: 5 }
    ],
    links: [
      { source: 'risk', target: 'audit' },
      { source: 'audit', target: 'compliance' },
      { source: 'compliance', target: 'investigation' },
      { source: 'investigation', target: 'report' },
      { source: 'risk', target: 'investigation' },
      { source: 'compliance', target: 'report' }
    ]
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" fontWeight="bold">
          Agent Console
        </Typography>
        <Box>
          <Tooltip title={isRunning ? "Stop Workflow" : "Start Workflow"}>
            <IconButton
              color={isRunning ? "error" : "success"}
              onClick={isRunning ? stopWorkflow : startWorkflow}
              sx={{ mr: 1 }}
            >
              {isRunning ? <StopIcon /> : <PlayArrowIcon />}
            </IconButton>
          </Tooltip>
          <Tooltip title="Refresh">
            <IconButton onClick={() => sendMessage(JSON.stringify({ type: 'REFRESH_AGENTS' }))}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Agent Status Cards */}
        <Grid item xs={12}>
          <Grid container spacing={2}>
            {Object.entries(agents).map(([name, data]) => (
              <Grid item xs={12} sm={6} md={2.4} key={name}>
                <Card
                  sx={{
                    cursor: 'pointer',
                    border: selectedAgent === name ? 2 : 1,
                    borderColor: selectedAgent === name ? 'primary.main' : 'divider',
                    '&:hover': { boxShadow: 3 }
                  }}
                  onClick={() => setSelectedAgent(name)}
                >
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Badge
                        color={getStatusColor(data.status)}
                        variant="dot"
                        anchorOrigin={{
                          vertical: 'top',
                          horizontal: 'right',
                        }}
                      >
                        <Avatar sx={{ bgcolor: 'primary.main' }}>
                          {getAgentIcon(name)}
                        </Avatar>
                      </Badge>
                      <Typography variant="h6" sx={{ ml: 1, textTransform: 'capitalize' }}>
                        {name}
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="textSecondary">
                      Tasks: {data.tasks}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Memory: {data.memory?.length || 0} items
                    </Typography>
                    {data.status === 'processing' && (
                      <LinearProgress sx={{ mt: 1 }} />
                    )}
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Grid>

        {/* Agent Graph Visualization */}
        <Grid item xs={12} md={7}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Agent Communication Graph
            </Typography>
            <Box sx={{ height: 400 }}>
              <ForceGraph2D
                graphData={graphData}
                nodeLabel="name"
                nodeColor={node => {
                  const agent = agents[node.id];
                  return agent?.status === 'active' ? '#4caf50' :
                         agent?.status === 'processing' ? '#ff9800' :
                         agent?.status === 'error' ? '#f44336' : '#1976d2';
                }}
                nodeCanvasObject={(node, ctx, globalScale) => {
                  const label = node.name;
                  const fontSize = 12/globalScale;
                  ctx.font = `${fontSize}px Sans-Serif`;
                  ctx.fillStyle = 'black';
                  ctx.fillText(label, node.x, node.y + 8);
                }}
                linkDirectionalParticles={2}
                linkDirectionalParticleSpeed={0.005}
              />
            </Box>
          </Paper>
        </Grid>

        {/* Selected Agent Details */}
        <Grid item xs={12} md={5}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              {selectedAgent ? `${selectedAgent.charAt(0).toUpperCase() + selectedAgent.slice(1)} Agent Details` : 'Select an Agent'}
            </Typography>
            {selectedAgent && agents[selectedAgent] && (
              <Box>
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <TimelineIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary="Status"
                      secondary={
                        <Chip
                          label={agents[selectedAgent].status}
                          size="small"
                          color={getStatusColor(agents[selectedAgent].status)}
                        />
                      }
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <MemoryIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary="Tasks Completed"
                      secondary={agents[selectedAgent].tasks}
                    />
                  </ListItem>
                </List>

                <Divider sx={{ my: 2 }} />

                <Typography variant="subtitle2" gutterBottom>
                  Recent Memory
                </Typography>
                <Box sx={{ maxHeight: 200, overflow: 'auto' }}>
                  {agents[selectedAgent].memory?.slice(-5).map((item, index) => (
                    <Accordion key={index} size="small">
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography variant="body2">
                          {item.key} - {item.timestamp}
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Typography variant="caption" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                          {JSON.stringify(item.value, null, 2)}
                        </Typography>
                      </AccordionDetails>
                    </Accordion>
                  ))}
                </Box>

                {agents[selectedAgent].status === 'error' && (
                  <Box sx={{ mt: 2, p: 2, bgcolor: '#ffebee', borderRadius: 1 }}>
                    <Typography color="error" variant="body2">
                      Error detected. Check logs for details.
                    </Typography>
                  </Box>
                )}
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Agent Activity Log */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Agent Activity Log
            </Typography>
            <List sx={{ maxHeight: 300, overflow: 'auto' }}>
              {[1, 2, 3, 4, 5].map((item) => (
                <React.Fragment key={item}>
                  <ListItem>
                    <ListItemIcon>
                      {item % 2 === 0 ? <CheckCircleIcon color="success" /> : <TimelineIcon color="info" />}
                    </ListItemIcon>
                    <ListItemText
                      primary={`Risk Agent completed assessment of 50 transactions`}
                      secondary="2 minutes ago"
                    />
                  </ListItem>
                  <Divider variant="inset" component="li" />
                </React.Fragment>
              ))}
            </List>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default AgentConsole;