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

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';


function App() {
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [anchorEl, setAnchorEl] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  
  // Socket.IO connection for real-time updates
  const { lastMessage, sendMessage } = useWebSocket(API_BASE);

  useEffect(() => {
    if (lastMessage) {
      const { event, payload } = lastMessage;
      if (event === 'REAL_TIME_ALERT') {
        setNotifications(prev => [payload, ...prev].slice(0, 50));
        setUnreadCount(prev => prev + 1);
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
        {/* App Bar */}
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
            <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
              AI-Powered Financial Risk & Audit System
            </Typography>
            <IconButton color="inherit" onClick={handleNotificationClick}>
              <Badge badgeContent={unreadCount} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>
            <IconButton
              color="inherit"
              onClick={(event) => setAnchorEl(event.currentTarget)}
            >
              <Avatar sx={{ width: 32, height: 32 }}>
                <PersonIcon />
              </Avatar>
            </IconButton>
            <Menu
              anchorEl={anchorEl}
              open={Boolean(anchorEl)}
              onClose={() => setAnchorEl(null)}
            >
              <MenuItem onClick={() => setAnchorEl(null)}>
                <ListItemIcon>
                  <PersonIcon fontSize="small" />
                </ListItemIcon>
                Profile
              </MenuItem>
              <MenuItem onClick={() => setAnchorEl(null)}>
                <ListItemIcon>
                  <SettingsIcon fontSize="small" />
                </ListItemIcon>
                Settings
              </MenuItem>
              <MenuItem onClick={() => setAnchorEl(null)}>
                <ListItemIcon>
                  <LogoutIcon fontSize="small" />
                </ListItemIcon>
                Logout
              </MenuItem>
            </Menu>
          </Toolbar>
        </AppBar>

        {/* Navigation Drawer */}
        <Drawer
          variant="persistent"
          anchor="left"
          open={drawerOpen}
          sx={{
            width: 240,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: 240,
              boxSizing: 'border-box',
            },
          }}
        >
          <Toolbar />
          <Box sx={{ overflow: 'auto' }}>
            <List>
              {menuItems.map((item) => (
                <ListItem
                  button
                  key={item.id}
                  selected={currentPage === item.id}
                  onClick={() => setCurrentPage(item.id)}
                >
                  <ListItemIcon>
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText primary={item.label} />
                </ListItem>
              ))}
            </List>
          </Box>
        </Drawer>

        {/* Main Content */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            transition: theme.transitions.create('margin', {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.leavingScreen,
            }),
            marginLeft: drawerOpen ? 0 : `-240px`,
          }}
        >
          <Toolbar />
          {renderPage()}
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;