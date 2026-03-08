import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for financial transactions"""
    
    def __init__(self):
        self.feature_stats = {}
        
    def create_features(self, df):
        """Create features for risk prediction - FIXED VERSION without time-based rolling"""
        df = df.copy()
        logger.info(f"Creating features from {len(df)} transactions")
        
        # Time-based features
        df = self._create_time_features(df)
        
        # Amount-based features
        df = self._create_amount_features(df)
        
        # Transaction pattern features - USING FIXED VERSION
        df = self._create_pattern_features_fixed(df)
        
        # Velocity features
        df = self._create_velocity_features_fixed(df)
        
        # Location-based features
        df = self._create_location_features(df)
        
        # Customer history features
        df = self._create_customer_features_fixed(df)
        
        # Interaction features
        df = self._create_interaction_features(df)
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        logger.info(f"Feature creation complete. Total features: {len(df.columns)}")
        
        return df
    
    def _create_time_features(self, df):
        """Create time-based features"""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Basic time features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            df['year'] = df['timestamp'].dt.year
            df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                                       (df['day_of_week'] < 5)).astype(int)
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
            df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
        
        return df
    
    def _create_amount_features(self, df):
        """Create amount-based features"""
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            df['amount_sqrt'] = np.sqrt(df['amount'])
            df['amount_squared'] = df['amount'] ** 2
            
            # Amount rounding patterns
            df['amount_rounded_100'] = (df['amount'] / 100).round() * 100
            df['amount_rounded_1000'] = (df['amount'] / 1000).round() * 1000
            df['amount_mod_100'] = df['amount'] % 100
            df['amount_mod_1000'] = df['amount'] % 1000
            df['is_round_amount'] = (df['amount'] % 100 == 0).astype(int)
            df['is_whole_number'] = (df['amount'] == df['amount'].astype(int)).astype(int)
            
            # Amount categories
            df['amount_digits'] = df['amount'].astype(str).str.replace('.', '').str.len()
            df['amount_decimal_places'] = df['amount'].astype(str).str.split('.').str[-1].str.len()
            
            # Amount bins
            try:
                df['amount_decile'] = pd.qcut(df['amount'], q=10, labels=False, duplicates='drop')
            except:
                df['amount_decile'] = pd.cut(df['amount'], bins=10, labels=False)
        
        return df
    
    def _create_pattern_features_fixed(self, df):
        """Create transaction pattern features - FIXED VERSION without time-based rolling"""
        required_cols = ['account_id', 'amount', 'timestamp']
        if all(col in df.columns for col in required_cols):
            # Sort by account and time
            df = df.sort_values(['account_id', 'timestamp'])
            
            # Previous transaction features
            df['prev_amount'] = df.groupby('account_id')['amount'].shift(1)
            df['prev_amount_diff'] = df['amount'] - df['prev_amount']
            df['prev_amount_diff_pct'] = (df['prev_amount_diff'] / (df['prev_amount'].abs() + 1)) * 100
            df['prev_amount_ratio'] = df['amount'] / (df['prev_amount'].abs() + 1)
            
            # Time since last transaction
            df['prev_timestamp'] = df.groupby('account_id')['timestamp'].shift(1)
            df['seconds_since_last'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds()
            df['minutes_since_last'] = df['seconds_since_last'] / 60
            df['hours_since_last'] = df['seconds_since_last'] / 3600
            df['days_since_last'] = df['hours_since_last'] / 24
            
            # Time-based flags
            df['is_rapid_txn'] = (df['minutes_since_last'] < 5).astype(int)
            df['is_very_rapid_txn'] = (df['minutes_since_last'] < 1).astype(int)
            
            # FIXED: Use integer windows instead of time-based strings
            windows = [3, 5, 10, 20, 50]  # number of previous transactions
            
            for window in windows:
                # Rolling mean
                df[f'avg_amount_last_{window}'] = df.groupby('account_id')['amount'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling std
                df[f'std_amount_last_{window}'] = df.groupby('account_id')['amount'].transform(
                    lambda x: x.rolling(window=window, min_periods=2).std()
                ).fillna(0)
                
                # Rolling max
                df[f'max_amount_last_{window}'] = df.groupby('account_id')['amount'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
                
                # Rolling min
                df[f'min_amount_last_{window}'] = df.groupby('account_id')['amount'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                
                # Rolling count
                df[f'count_last_{window}'] = df.groupby('account_id')['amount'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).count()
                )
                
                # Rolling sum
                df[f'sum_last_{window}'] = df.groupby('account_id')['amount'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).sum()
                )
            
            # Amount compared to recent averages
            df['amount_vs_avg_10'] = df['amount'] / (df['avg_amount_last_10'].abs() + 1)
            df['amount_vs_avg_20'] = df['amount'] / (df['avg_amount_last_20'].abs() + 1)
            
            # Z-score based on recent history
            df['amount_zscore_10'] = (df['amount'] - df['avg_amount_last_10']) / (df['std_amount_last_10'] + 1)
            df['amount_zscore_20'] = (df['amount'] - df['avg_amount_last_20']) / (df['std_amount_last_20'] + 1)
        
        return df
    
    def _create_velocity_features_fixed(self, df):
        """Create velocity-based features - FIXED VERSION"""
        if all(col in df.columns for col in ['account_id', 'timestamp']):
            # Simple velocity features without time-based grouping
            df['txn_date'] = df['timestamp'].dt.date
            
            # Daily transaction count (using transform to avoid future data leakage)
            daily_counts = df.groupby(['account_id', 'txn_date']).size().reset_index(name='daily_txn_count')
            df = df.merge(daily_counts, on=['account_id', 'txn_date'], how='left')
            
            # Daily amount sum
            daily_sums = df.groupby(['account_id', 'txn_date'])['amount'].sum().reset_index(name='daily_amount_sum')
            df = df.merge(daily_sums, on=['account_id', 'txn_date'], how='left')
            
            # Fill NaN
            df['daily_txn_count'] = df['daily_txn_count'].fillna(1)
            df['daily_amount_sum'] = df['daily_amount_sum'].fillna(df['amount'])
            
            # Amount per transaction
            df['amount_per_txn_daily'] = df['daily_amount_sum'] / df['daily_txn_count']
        
        return df
    
    def _create_location_features(self, df):
        """Create location-based features"""
        if 'location' in df.columns:
            # Location risk scores (simplified)
            high_risk_locations = ['NY', 'CA', 'FL', 'TX', 'NV']
            medium_risk_locations = ['IL', 'PA', 'OH', 'GA', 'NC']
            
            df['location_risk_score'] = df['location'].apply(
                lambda x: 1.0 if x in high_risk_locations 
                else (0.5 if x in medium_risk_locations else 0.2)
            )
            
            df['is_high_risk_location'] = df['location'].isin(high_risk_locations).astype(int)
            df['is_medium_risk_location'] = df['location'].isin(medium_risk_locations).astype(int)
            
            # Location frequency
            location_counts = df['location'].value_counts()
            location_freq = location_counts / len(df)
            location_freq_dict = location_freq.to_dict()
            
            df['location_frequency'] = df['location'].map(location_counts)
            df['location_frequency_ratio'] = df['location'].map(location_freq_dict)
        
        return df
    
    def _create_customer_features_fixed(self, df):
        """Create customer-based features - FIXED VERSION"""
        if 'account_id' in df.columns:
            # Customer lifetime statistics
            if 'amount' in df.columns:
                # Lifetime aggregates (using transform to avoid future leakage)
                df['customer_lifetime_txns'] = df.groupby('account_id')['amount'].transform('count')
                df['customer_lifetime_amount'] = df.groupby('account_id')['amount'].transform('sum')
                df['customer_avg_amount'] = df.groupby('account_id')['amount'].transform('mean')
                df['customer_std_amount'] = df.groupby('account_id')['amount'].transform('std').fillna(0)
                df['customer_max_amount'] = df.groupby('account_id')['amount'].transform('max')
                df['customer_min_amount'] = df.groupby('account_id')['amount'].transform('min')
                
                # Percentiles
                df['customer_amount_p95'] = df.groupby('account_id')['amount'].transform(
                    lambda x: x.quantile(0.95)
                )
                
                # Is this transaction unusual for this customer
                df['is_unusual_for_customer'] = (
                    (df['amount'] > df['customer_amount_p95']) | 
                    (df['amount'] < df['customer_min_amount'] * 0.5)
                ).astype(int)
                
                # Deviation from customer pattern
                df['amount_deviation_from_avg'] = df['amount'] - df['customer_avg_amount']
                df['amount_ratio_to_avg'] = df['amount'] / (df['customer_avg_amount'].abs() + 1)
        
        return df
    
    def _create_interaction_features(self, df):
        """Create interaction features"""
        # Amount and time interactions
        if 'amount' in df.columns:
            if 'is_weekend' in df.columns:
                df['amount_weekend'] = df['amount'] * df['is_weekend']
            
            if 'is_business_hours' in df.columns:
                df['amount_business_hours'] = df['amount'] * df['is_business_hours']
            
            if 'is_high_risk_location' in df.columns:
                df['amount_high_risk_location'] = df['amount'] * df['is_high_risk_location']
        
        return df
    
    def get_feature_names(self):
        """Get list of feature names"""
        return list(self.feature_stats.keys())