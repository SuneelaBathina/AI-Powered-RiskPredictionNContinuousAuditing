import React, { useEffect, useState } from 'react';
import {
  Typography,
  Box,
  Paper,
  CircularProgress,
  Alert,
  useTheme
} from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

const columns = [
  { field: 'transaction_id', headerName: 'Txn ID', width: 180 },
  { field: 'timestamp', headerName: 'Timestamp', width: 200 },
  { field: 'account_id', headerName: 'Account', width: 130 },
  { field: 'amount', headerName: 'Amount', width: 110, type: 'number' },
  { field: 'transaction_type', headerName: 'Type', width: 140 },
  { field: 'location', headerName: 'Location', width: 120 },
  { field: 'merchant_category', headerName: 'Category', width: 140 },
  { field: 'channel', headerName: 'Channel', width: 120 },
  { field: 'device_type', headerName: 'Device', width: 120 },
  {
    field: 'is_fraud',
    headerName: 'Fraud?',
    width: 100,
    type: 'boolean',
    valueFormatter: (params) => (params.value ? 'Yes' : 'No')
  }
];

const TransactionsView = () => {
  const theme = useTheme();
  const [transactions, setTransactions] = useState([]);
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(25);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchTransactions = async (pageNumber = 0, perPage = 25) => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${API_BASE}/api/transactions`, {
        params: {
          page: pageNumber + 1,
          per_page: perPage
        }
      });

      const data = response?.data || {};
      console.log('Transactions API response:', data);

      if (!Array.isArray(data.transactions)) {
        throw new Error(data.error || 'Invalid transactions response from API');
      }

      setTransactions(
        data.transactions.map((txn) => ({
          ...txn,
          id: txn.transaction_id ?? txn.id
        }))
      );
      setTotal(Number(data.total ?? data.transactions.length));
    } catch (err) {
      console.error('Error fetching transactions', err);
      setError(err.response?.data?.error || err.message || 'Failed to fetch transactions');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTransactions(page, pageSize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, pageSize]);

  return (
    <Box sx={{ height: '100%', width: '100%' }}>
      <Typography
        variant="h4"
        gutterBottom
        sx={{ color: theme.palette.text.primary }}
      >
        Transactions
      </Typography>
      <Typography
        variant="h6"
        gutterBottom
        sx={{ color: theme.palette.text.secondary }}
      >
        
      </Typography>

      <Paper sx={{ height: 560, width: '100%', p: 2, bgcolor: theme.palette.background.paper }}>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <CircularProgress />
          </Box>
        ) : (
          <DataGrid
            rows={transactions}
            columns={columns}
            pagination
            paginationMode="server"
            rowCount={total}
            page={page}
            pageSize={pageSize}
            onPageChange={(newPage) => setPage(newPage)}
            onPageSizeChange={(newPageSize) => setPageSize(newPageSize)}
            rowsPerPageOptions={[10, 25, 50, 100]}
            disableSelectionOnClick
            autoHeight
            sx={{
              border: 0,
              bgcolor: theme.palette.background.default,
              color: theme.palette.text.primary,
              '& .MuiDataGrid-columnHeaders': {
                bgcolor: theme.palette.background.paper,
                color: theme.palette.text.primary,
                borderBottom: `1px solid ${theme.palette.divider}`
              },
              '& .MuiDataGrid-cell': {
                borderBottom: `1px solid ${theme.palette.divider}`
              },
              '& .MuiDataGrid-footerContainer': {
                bgcolor: theme.palette.background.paper,
                borderTop: `1px solid ${theme.palette.divider}`
              }
            }}
          />
        )}
      </Paper>
    </Box>
  );
};

export default TransactionsView;