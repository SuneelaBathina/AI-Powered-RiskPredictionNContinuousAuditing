import React, { useState, useEffect } from 'react';
import {
  ThemeProvider,
  CssBaseline,
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Badge,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Avatar,
  Menu,
  MenuItem,
  Chip
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Assessment as AssessmentIcon,
  Security as SecurityIcon,
  AccountBalance as AccountBalanceIcon,
  Timeline as TimelineIcon,
  Notifications as NotificationsIcon,
  Person as PersonIcon,
  Settings as SettingsIcon,
  Logout as LogoutIcon,
  Warning as WarningIcon
} from '@mui/icons-material';
import { theme } from './styles/theme';
import { useWebSocket } from './hooks/useWebSocket';
import Dashboard from './components/Dashboard';
import RiskAnalytics from './components/RiskAnalytics';
import AuditWorkflow from './components/AuditWorkflow';
import AgentConsole from './components/AgentConsole';
import TransactionsView from './components/TransactionsView';
import Reports from './components/Reports';

function App() {
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [anchorEl, setAnchorEl] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  
  // WebSocket connection for real-time updates
  const { lastMessage, sendMessage } = useWebSocket('ws://localhost:5000');

  useEffect(() => {
    if (lastMessage) {
      try {
        const data = JSON.parse(lastMessage.data);
        if (data.type === 'REAL_TIME_ALERT') {
          setNotifications(prev => [data.data, ...prev].slice(0, 50));
          setUnreadCount(prev => prev + 1);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    }
  }, [lastMessage]);

  const handleNotificationClick = () => {
    setUnreadCount(0);
  };

  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: <DashboardIcon /> },
    { id: 'risk-analytics', label: 'Risk Analytics', icon: <AssessmentIcon /> },
    { id: 'audit-workflow', label: 'Audit Workflow', icon: <SecurityIcon /> },
    { id: 'agent-console', label: 'Agent Console', icon: <TimelineIcon /> },
    { id: 'transactions', label: 'Transactions', icon: <AccountBalanceIcon /> },
    { id: 'reports', label: 'Reports', icon: <AssessmentIcon /> }
  ];

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />;
      case 'risk-analytics':
        return <RiskAnalytics />;
      case 'audit-workflow':
        return <AuditWorkflow />;
      case 'agent-console':
        return <AgentConsole />;
      case 'transactions':
        return <TransactionsView />;
      case 'reports':
        return <Reports />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex' }}>
        <AppBar position="fixed" sx={{ zIndex: theme.zIndex.drawer + 1 }}>
          <Toolbar>
            <IconButton
              color="inherit"
              edge="start"
              onClick={() => setDrawerOpen(!drawerOpen)}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" noWrap sx={{ flexGrow: 1 }}>
              AI-Powered Financial Risk & Audit System
            </Typography>
            <IconButton color="inherit" onClick={handleNotificationClick}>
              <Badge badgeContent={unreadCount} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>
            <IconButton color="inherit" onClick={(e) => setAnchorEl(e.currentTarget)}>
              <Avatar sx={{ width: 32, height: 32 }}>
                <PersonIcon />
              </Avatar>
            </IconButton>
          </Toolbar>
        </AppBar>

        <Drawer
          variant="permanent"
          sx={{
            width: drawerOpen ? 280 : 72,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: drawerOpen ? 280 : 72,
              boxSizing: 'border-box',
              transition: theme.transitions.create('width', {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.enteringScreen,
              }),
              overflowX: 'hidden',
            },
          }}
        >
          <Toolbar />
          <Box sx={{ overflow: 'auto', mt: 2 }}>
            <List>
              {menuItems.map((item) => (
                <ListItem
                  button
                  key={item.id}
                  onClick={() => setCurrentPage(item.id)}
                  selected={currentPage === item.id}
                  sx={{
                    mx: 1,
                    borderRadius: 1,
                    '&.Mui-selected': {
                      backgroundColor: 'primary.light',
                      color: 'primary.contrastText',
                      '&:hover': {
                        backgroundColor: 'primary.main',
                      },
                      '& .MuiListItemIcon-root': {
                        color: 'primary.contrastText',
                      },
                    },
                  }}
                >
                  <ListItemIcon sx={{ minWidth: drawerOpen ? 56 : 'auto', ml: drawerOpen ? 0 : 1 }}>
                    {item.icon}
                  </ListItemIcon>
                  {drawerOpen && <ListItemText primary={item.label} />}
                </ListItem>
              ))}
            </List>
          </Box>
        </Drawer>

        <Box component="main" sx={{ flexGrow: 1, p: 3, bgcolor: 'background.default', minHeight: '100vh' }}>
          <Toolbar />
          {renderPage()}
        </Box>

        {/* User Menu */}
        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={() => setAnchorEl(null)}
        >
          <MenuItem onClick={() => setAnchorEl(null)}>
            <ListItemIcon>
              <PersonIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>Profile</ListItemText>
          </MenuItem>
          <MenuItem onClick={() => setAnchorEl(null)}>
            <ListItemIcon>
              <SettingsIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>Settings</ListItemText>
          </MenuItem>
          <MenuItem onClick={() => setAnchorEl(null)}>
            <ListItemIcon>
              <LogoutIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>Logout</ListItemText>
          </MenuItem>
        </Menu>

        {/* Notifications Drawer */}
        <Drawer
          anchor="right"
          open={unreadCount > 0}
          onClose={() => setUnreadCount(0)}
          sx={{
            '& .MuiDrawer-paper': {
              width: 400,
              p: 2,
            },
          }}
        >
          <Typography variant="h6" gutterBottom>
            Real-time Alerts
          </Typography>
          {notifications.slice(0, 10).map((notification, index) => (
            <Box
              key={index}
              sx={{
                p: 2,
                mb: 1,
                bgcolor: 'background.paper',
                borderRadius: 1,
                border: '1px solid',
                borderColor: 'divider',
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <WarningIcon color="error" sx={{ mr: 1 }} />
                <Typography variant="subtitle2">
                  High Risk Alert
                </Typography>
                <Chip
                  label="CRITICAL"
                  size="small"
                  color="error"
                  sx={{ ml: 'auto' }}
                />
              </Box>
              <Typography variant="body2">
                Transaction ID: {notification.transaction_id}
              </Typography>
              <Typography variant="body2">
                Amount: ${notification.amount?.toFixed(2)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Risk Score: {(notification.risk_score * 100).toFixed(1)}%
              </Typography>
            </Box>
          ))}
        </Drawer>
      </Box>
    </ThemeProvider>
  );
}

export default App;