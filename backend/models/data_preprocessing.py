import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    SMOTE = None
    SMOTETomek = None
import joblib
import os
import logging
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import json

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing class for financial transaction data.
    Handles feature engineering, scaling, encoding, and data splitting.
    """
    
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.numeric_features = []
        self.categorical_features = []
        self.datetime_features = []
        self.target_column = 'is_fraud'
        self.imputer = None
        self.smote = None
        
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'is_fraud', 
                    test_size: float = 0.2, random_state: int = 42,
                    balance_data: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete data preparation pipeline
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            test_size: Proportion of test set
            random_state: Random seed
            balance_data: Whether to apply SMOTE for class balancing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Starting data preparation pipeline")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Store target column
        self.target_column = target_column
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Create features
        df = self.create_features(df)
        
        # Identify feature types
        self._identify_feature_types(df)
        
        # Separate features and target
        X = df.drop(columns=[target_column], errors='ignore')
        y = df[target_column] if target_column in df.columns else None
        
        if y is None:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y, shuffle=True
        )
        
        # Fit preprocessors on training data
        self._fit_preprocessors(X_train)
        
        # Transform data
        X_train_transformed = self.transform(X_train)
        X_test_transformed = self.transform(X_test)
        
        # Balance training data if requested
        if balance_data and self._is_imbalanced(y_train):
            logger.info("Applying SMOTE for class balancing")
            X_train_transformed, y_train = self._balance_data(
                X_train_transformed, y_train, random_state
            )
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        logger.info(f"Data preparation complete. Train shape: {X_train_transformed.shape}, "
                   f"Test shape: {X_test_transformed.shape}")
        
        return X_train_transformed, X_test_transformed, y_train, y_test
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataframe"""
        logger.info("Handling missing values")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
        
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'UNKNOWN', inplace=True)
        
        # For datetime columns, forward fill
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            if df[col].isnull().any():
                df[col].fillna(method='ffill', inplace=True)
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features for risk prediction
        """
        logger.info("Creating features")
        df = df.copy()
        
        # Time-based features
        df = self._create_time_features(df)
        
        # Amount-based features
        df = self._create_amount_features(df)
        
        # Transaction pattern features
        df = self._create_pattern_features(df)
        
        # Velocity features
        df = self._create_velocity_features(df)
        
        # Location-based features
        df = self._create_location_features(df)
        
        # Customer history features
        df = self._create_customer_features(df)
        
        # Interaction features
        df = self._create_interaction_features(df)
        
        # Aggregate features
        df = self._create_aggregate_features(df)
        
        logger.info(f"Feature creation complete. Total features: {len(df.columns)}")
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
            df['week_of_year'] = df['timestamp'].dt.isocalendar().week
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                                       (df['day_of_week'] < 5)).astype(int)
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Time segments
            df['time_segment'] = pd.cut(df['hour'], 
                                        bins=[0, 6, 12, 18, 24], 
                                        labels=['late_night', 'morning', 'afternoon', 'evening'])
        
        return df
    
    def _create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features"""
        if 'amount' in df.columns:
            # Basic amount transformations
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
            
            # Amount digits and patterns
            df['amount_digits'] = df['amount'].astype(str).str.replace('.', '').str.len()
            df['amount_decimal_places'] = df['amount'].astype(str).str.split('.').str[-1].str.len()
            
            # Amount bins/categories
            df['amount_bin'] = pd.qcut(df['amount'], q=10, labels=False, duplicates='drop')
            
        return df
    
    def _create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
    
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
            df['hours_since_last_txn'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds() / 3600
            df['days_since_last_txn'] = df['hours_since_last_txn'] / 24
            
            # Time-based flags
            df['is_rapid_txn'] = (df['hours_since_last_txn'] < 1).astype(int)  # Less than 1 hour
            df['is_very_rapid_txn'] = (df['hours_since_last_txn'] < 0.1).astype(int)  # Less than 6 minutes
            
            # Simple rolling statistics (based on transaction count, not time)
            windows = [3, 5, 10]
            for window in windows:
                # Average of last N transactions
                df[f'avg_last_{window}'] = df.groupby('account_id')['amount'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Standard deviation
                df[f'std_last_{window}'] = df.groupby('account_id')['amount'].transform(
                    lambda x: x.rolling(window=window, min_periods=2).std()
                ).fillna(0)
                
                # Max of last N
                df[f'max_last_{window}'] = df.groupby('account_id')['amount'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
                
                # Count in last N
                df[f'count_last_{window}'] = df.groupby('account_id')['amount'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).count()
                )
            
            # Compare with recent average
            df['amount_vs_avg_10'] = df['amount'] / (df['avg_last_10'].abs() + 1)
            
            # Z-score based on recent history
            df['amount_zscore_10'] = (df['amount'] - df['avg_last_10']) / (df['std_last_10'] + 1)
            
            # Fill NaN
            for col in df.columns:
                if col.startswith(('avg_', 'std_', 'max_', 'count_')):
                    df[col] = df[col].fillna(0)
        
        return df
    
    def _create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create velocity-based features"""
        if all(col in df.columns for col in ['account_id', 'timestamp', 'amount']):
            # Hourly velocity
            df['txn_hour'] = df['timestamp'].dt.floor('H')
            hourly_counts = df.groupby(['account_id', 'txn_hour']).size().reset_index(name='hourly_txn_count')
            df = df.merge(hourly_counts, on=['account_id', 'txn_hour'], how='left')
            
            hourly_amount = df.groupby(['account_id', 'txn_hour'])['amount'].sum().reset_index(name='hourly_amount_sum')
            df = df.merge(hourly_amount, on=['account_id', 'txn_hour'], how='left')
            
            # Daily velocity
            df['txn_date'] = df['timestamp'].dt.date
            daily_counts = df.groupby(['account_id', 'txn_date']).size().reset_index(name='daily_txn_count')
            df = df.merge(daily_counts, on=['account_id', 'txn_date'], how='left')
            
            daily_amount = df.groupby(['account_id', 'txn_date'])['amount'].sum().reset_index(name='daily_amount_sum')
            df = df.merge(daily_amount, on=['account_id', 'txn_date'], how='left')
            
            # Velocity ratios
            df['amount_per_hour'] = df['hourly_amount_sum'] / (df['hourly_txn_count'] + 1)
            df['amount_per_day'] = df['daily_amount_sum'] / (df['daily_txn_count'] + 1)
            
            # Velocity changes
            df['hourly_velocity_change'] = df.groupby('account_id')['hourly_txn_count'].diff()
            df['daily_velocity_change'] = df.groupby('account_id')['daily_txn_count'].diff()
            
        return df
    
    def _create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based features"""
        if 'location' in df.columns:
            # Location risk (simplified - in production, use actual country risk scores)
            high_risk_locations = ['NY', 'CA', 'FL', 'TX', 'NV']  # Example high-risk states
            medium_risk_locations = ['IL', 'PA', 'OH', 'GA', 'NC']
            
            df['location_risk_score'] = df['location'].apply(
                lambda x: 1.0 if x in high_risk_locations 
                else (0.5 if x in medium_risk_locations else 0.0)
            )
            
            df['is_high_risk_location'] = df['location'].isin(high_risk_locations).astype(int)
            df['is_medium_risk_location'] = df['location'].isin(medium_risk_locations).astype(int)
            
            # Location frequency
            location_counts = df['location'].value_counts().to_dict()
            df['location_frequency'] = df['location'].map(location_counts)
            df['location_frequency_ratio'] = df['location_frequency'] / len(df)
            
        return df
    
    def _create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-based features"""
        if 'account_id' in df.columns:
            # Account age if available
            if 'account_created_date' in df.columns:
                df['account_created_date'] = pd.to_datetime(df['account_created_date'])
                df['account_age_days'] = (df['timestamp'] - df['account_created_date']).dt.days
                df['account_age_months'] = df['account_age_days'] / 30
                df['account_age_years'] = df['account_age_days'] / 365
                df['is_new_account'] = (df['account_age_days'] < 30).astype(int)
                df['is_old_account'] = (df['account_age_days'] > 365).astype(int)
            
            # Customer transaction history
            if 'amount' in df.columns:
                # Lifetime statistics
                df['customer_lifetime_txns'] = df.groupby('account_id')['amount'].transform('count')
                df['customer_lifetime_amount'] = df.groupby('account_id')['amount'].transform('sum')
                df['customer_avg_amount'] = df.groupby('account_id')['amount'].transform('mean')
                df['customer_std_amount'] = df.groupby('account_id')['amount'].transform('std')
                df['customer_max_amount'] = df.groupby('account_id')['amount'].transform('max')
                df['customer_min_amount'] = df.groupby('account_id')['amount'].transform('min')
                
                # Percentiles
                df['customer_amount_p95'] = df.groupby('account_id')['amount'].transform(
                    lambda x: x.quantile(0.95)
                )
                df['customer_amount_p99'] = df.groupby('account_id')['amount'].transform(
                    lambda x: x.quantile(0.99)
                )
                
                # Is this transaction unusual for this customer
                df['is_unusual_for_customer'] = (
                    (df['amount'] > df['customer_amount_p95']) | 
                    (df['amount'] < df['customer_min_amount'] * 0.5)
                ).astype(int)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between existing features"""
        # Amount and time interactions
        if 'amount' in df.columns and 'is_weekend' in df.columns:
            df['amount_weekend'] = df['amount'] * df['is_weekend']
        
        if 'amount' in df.columns and 'is_business_hours' in df.columns:
            df['amount_business_hours'] = df['amount'] * df['is_business_hours']
            df['amount_non_business'] = df['amount'] * (1 - df['is_business_hours'])
        
        # Amount and location interactions
        if 'amount' in df.columns and 'is_high_risk_location' in df.columns:
            df['amount_high_risk_location'] = df['amount'] * df['is_high_risk_location']
        
        # Velocity and amount interactions
        if 'hourly_txn_count' in df.columns and 'amount' in df.columns:
            df['amount_per_txn_hourly'] = df['amount'] / (df['hourly_txn_count'] + 1)
        
        # Risk score interactions
        if 'customer_avg_amount' in df.columns and 'amount' in df.columns:
            df['amount_to_customer_avg'] = df['amount'] / (df['customer_avg_amount'] + 1)
            df['amount_deviation_from_avg'] = df['amount'] - df['customer_avg_amount']
        
        return df
    
    def _create_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregate features across different dimensions"""
        # Merchant category features (if available)
        if 'merchant_category' in df.columns:
            high_risk_categories = ['gambling', 'cryptocurrency', 'money_transfer', 'adult']
            df['is_high_risk_category'] = df['merchant_category'].isin(high_risk_categories).astype(int)
            
            # Category frequency
            category_counts = df['merchant_category'].value_counts().to_dict()
            df['category_frequency'] = df['merchant_category'].map(category_counts)
        
        # Channel features (if available)
        if 'channel' in df.columns:
            high_risk_channels = ['online', 'mobile', 'international']
            df['is_high_risk_channel'] = df['channel'].isin(high_risk_channels).astype(int)
        
        # Device features (if available)
        if 'device_type' in df.columns:
            device_risk = {'MOBILE': 0.7, 'TABLET': 0.5, 'DESKTOP': 0.3}
            df['device_risk_score'] = df['device_type'].map(device_risk).fillna(0.5)
        
        return df
    
    def _identify_feature_types(self, df: pd.DataFrame):
        """Identify numeric, categorical, and datetime features"""
        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_features = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Remove target column from features if present
        if self.target_column in self.numeric_features:
            self.numeric_features.remove(self.target_column)
        if self.target_column in self.categorical_features:
            self.categorical_features.remove(self.target_column)
    
    def _fit_preprocessors(self, X_train: pd.DataFrame):
        """Fit all preprocessors on training data"""
        logger.info("Fitting preprocessors")
        
        # Initialize scaler based on data distribution
        if self._has_outliers(X_train):
            self.scaler = RobustScaler()
            logger.info("Using RobustScaler due to outliers")
        else:
            self.scaler = StandardScaler()
            logger.info("Using StandardScaler")
        
        # Fit scaler on numeric features
        if self.numeric_features:
            numeric_data = X_train[self.numeric_features].fillna(0)
            self.scaler.fit(numeric_data)
        
        # Fit label encoders for categorical features
        for feature in self.categorical_features:
            if feature in X_train.columns:
                self.label_encoders[feature] = LabelEncoder()
                # Handle unseen categories by adding 'UNKNOWN'
                unique_values = X_train[feature].fillna('UNKNOWN').unique().tolist()
                if 'UNKNOWN' not in unique_values:
                    unique_values.append('UNKNOWN')
                self.label_encoders[feature].fit(unique_values)
        
        # Initialize imputer
        self.imputer = SimpleImputer(strategy='median')
        if self.numeric_features:
            self.imputer.fit(X_train[self.numeric_features])
        
        logger.info(f"Preprocessors fitted: {len(self.numeric_features)} numeric, "
                   f"{len(self.categorical_features)} categorical features")
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted preprocessors
        
        Args:
            X: Input features dataframe
            
        Returns:
            Transformed features as numpy array
        """
        transformed_features = []
        
        # Transform numeric features
        if self.numeric_features:
            numeric_data = X[self.numeric_features].fillna(0).copy()
            
            # Apply imputation
            if self.imputer:
                numeric_data = self.imputer.transform(numeric_data)
            
            # Apply scaling
            if self.scaler:
                numeric_scaled = self.scaler.transform(numeric_data)
                transformed_features.append(numeric_scaled)
        
        # Transform categorical features
        for feature in self.categorical_features:
            if feature in X.columns and feature in self.label_encoders:
                # Handle unseen categories
                encoder = self.label_encoders[feature]
                feature_data = X[feature].fillna('UNKNOWN')
                
                # Replace unseen categories with 'UNKNOWN'
                feature_data = feature_data.apply(
                    lambda x: x if x in encoder.classes_ else 'UNKNOWN'
                )
                
                encoded = encoder.transform(feature_data).reshape(-1, 1)
                transformed_features.append(encoded)
        
        # Combine all features
        if transformed_features:
            X_transformed = np.hstack(transformed_features)
        else:
            X_transformed = np.array([])
        
        return X_transformed
    
    def _has_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> bool:
        """Check if dataset has significant outliers"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return False
        
        # Calculate z-scores
        z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
        
        # Check if any column has >5% outliers
        outlier_ratio = (z_scores > threshold).mean().max()
        return outlier_ratio > 0.05
    
    def _is_imbalanced(self, y: pd.Series, threshold: float = 0.3) -> bool:
        """Check if dataset is imbalanced"""
        class_counts = y.value_counts(normalize=True)
        return class_counts.min() < threshold
    
    def _balance_data(self, X: np.ndarray, y: pd.Series, random_state: int = 42) -> Tuple[np.ndarray, pd.Series]:
        """Balance imbalanced dataset using SMOTE"""
        try:
            # Use SMOTE with Tomek links for better results
            self.smote = SMOTETomek(random_state=random_state)
            X_resampled, y_resampled = self.smote.fit_resample(X, y)
            
            logger.info(f"Data balanced. Original shape: {X.shape}, New shape: {X_resampled.shape}")
            return X_resampled, pd.Series(y_resampled)
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Using original data.")
            return X, y
    
    def prepare_single_transaction(self, transaction_data: Dict) -> np.ndarray:
        """
        Prepare a single transaction for prediction
        
        Args:
            transaction_data: Dictionary containing transaction data
            
        Returns:
            Transformed features for prediction
        """
        # Convert to dataframe
        df = pd.DataFrame([transaction_data])
        
        # Create features
        df = self.create_features(df)
        
        # Ensure all required features are present
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
        
        # Select only the features used during training
        if self.feature_columns:
            df = df[self.feature_columns]
        
        # Transform
        X_transformed = self.transform(df)
        
        return X_transformed
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        if self.feature_columns:
            return self.feature_columns
        else:
            return self.numeric_features + self.categorical_features
    
    def get_feature_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics for all features"""
        stats = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q50': float(df[col].quantile(0.5)),
                'q75': float(df[col].quantile(0.75)),
                'missing': int(df[col].isnull().sum())
            }
        
        return stats
    
    def save(self, path: str):
        """
        Save preprocessor to disk
        
        Args:
            path: Path to save preprocessor
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'datetime_features': self.datetime_features,
            'target_column': self.target_column,
            'imputer': self.imputer
        }
        
        joblib.dump(preprocessor_data, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load(self, path: str):
        """
        Load preprocessor from disk
        
        Args:
            path: Path to load preprocessor from
        """
        preprocessor_data = joblib.load(path)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_columns = preprocessor_data['feature_columns']
        self.numeric_features = preprocessor_data['numeric_features']
        self.categorical_features = preprocessor_data['categorical_features']
        self.datetime_features = preprocessor_data['datetime_features']
        self.target_column = preprocessor_data['target_column']
        self