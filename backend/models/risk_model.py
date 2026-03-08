import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class RiskPredictor:
    """XGBoost-based risk prediction model"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.feature_importance = None
        self.metrics = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.model_path = 'models/saved/risk_model.pkl'
        
    def train_from_csv(self, csv_path, target_column='is_fraud'):
        """Train model from CSV file"""
        logger.info(f"Loading data from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            logger.info("Creating synthetic training data...")
            df = self._create_synthetic_training_data()
        
        # Engineer features using feature_engineer
        logger.info("Engineering features...")
        df = self.feature_engineer.create_features(df)
        
        # Prepare data using preprocessor
        logger.info("Preparing data for training...")
        # FIXED: Changed from prepare_features to create_features
        X, y = self._prepare_features(df, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate class weight
        scale_pos_weight = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else 1
        
        # Initialize XGBoost model
        logger.info("Training XGBoost model...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        self.metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_proba))
        }
        
        logger.info(f"Training complete. Metrics: {self.metrics}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_columns or [f"feature_{i}" for i in range(X.shape[1])],
                self.model.feature_importances_
            ))
        
        # Save model
        self.save_model()
        
        return self.model
    
    def _prepare_features(self, df, target_column):
        """Prepare features for training"""
        # Handle target column
        if target_column in df.columns:
            y = df[target_column].values
            X = df.drop(columns=[target_column])
        else:
            logger.warning(f"Target column '{target_column}' not found. Creating synthetic target.")
            y = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
            X = df.copy()
        
        # Select numeric columns only for simplicity
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            logger.warning("No numeric columns found. Using all columns.")
            numeric_cols = X.columns.tolist()
        
        X_numeric = X[numeric_cols].fillna(0)
        
        # Store feature columns
        self.feature_columns = numeric_cols
        
        return X_numeric.values, y
    
    def _create_synthetic_training_data(self):
        """Create synthetic training data"""
        logger.info("Creating synthetic training data")
        np.random.seed(42)
        
        n_samples = 5000
        
        data = {
            'transaction_id': [f'TXN_{str(i).zfill(6)}' for i in range(n_samples)],
            'amount': np.random.exponential(1000, n_samples).round(2),
            'account_age_days': np.random.randint(1, 3650, n_samples),
            'account_balance': np.random.uniform(100, 100000, n_samples).round(2),
            'previous_transactions_24h': np.random.poisson(5, n_samples),
            'avg_transaction_amount_30d': np.random.exponential(800, n_samples).round(2),
            'transaction_type': np.random.choice(['PURCHASE', 'WITHDRAWAL', 'DEPOSIT', 'TRANSFER'], n_samples),
            'location': np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL'], n_samples),
            'device_type': np.random.choice(['MOBILE', 'DESKTOP', 'TABLET'], n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        }
        
        df = pd.DataFrame(data)
        
        # Add some patterns for fraud
        fraud_indices = df[df['is_fraud'] == 1].index
        for idx in fraud_indices[:100]:
            if idx < len(df):
                df.loc[idx, 'amount'] = np.random.uniform(5000, 50000).round(2)
                df.loc[idx, 'previous_transactions_24h'] = np.random.poisson(15)
        
        logger.info(f"Created {len(df)} synthetic transactions with {df['is_fraud'].sum()} fraud cases")
        return df
    
    def predict(self, transaction_data):
        """Predict risk for a single transaction"""
        if self.model is None:
            logger.warning("Model not loaded. Attempting to load...")
            if not self.load_model():
                logger.warning("No saved model found. Using rule-based prediction.")
                return self._rule_based_prediction(transaction_data)
        
        # Convert to dataframe
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data
        
        # Engineer features
        df = self.feature_engineer.create_features(df)
        
        # Extract numeric features
        if self.feature_columns:
            # Use stored feature columns
            X_dict = {}
            for col in self.feature_columns:
                if col in df.columns:
                    X_dict[col] = float(df[col].iloc[0]) if pd.notna(df[col].iloc[0]) else 0
                else:
                    X_dict[col] = 0
            X = np.array([[X_dict[col] for col in self.feature_columns]])
        else:
            # Use all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                X = df[numeric_cols].fillna(0).values
            else:
                X = np.array([[1000, 365, 10000]])  # Default values
        
        # Scale features
        try:
            X_scaled = self.scaler.transform(X)
        except:
            X_scaled = X
        
        # Predict
        risk_score = float(self.model.predict_proba(X_scaled)[0, 1])
        
        return {
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score),
            'confidence': 2 * abs(risk_score - 0.5),
            'prediction': 1 if risk_score > 0.5 else 0
        }
    
    def _rule_based_prediction(self, data):
        """Simple rule-based prediction for fallback"""
        amount = float(data.get('amount', 1000))
        account_age = int(data.get('account_age_days', 365))
        prev_txns = int(data.get('previous_transactions_24h', 5))
        
        risk_score = 0.2
        if amount > 10000:
            risk_score += 0.5
        elif amount > 5000:
            risk_score += 0.3
        
        if account_age < 30:
            risk_score += 0.2
        
        if prev_txns > 15:
            risk_score += 0.3
        
        risk_score = min(risk_score, 0.95)
        
        return {
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score),
            'confidence': 0.7,
            'prediction': 1 if risk_score > 0.5 else 0
        }
    
    def _get_risk_level(self, score):
        """Convert risk score to risk level"""
        if score >= 0.7:
            return 'HIGH'
        elif score >= 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def save_model(self, path=None):
        """Save model to disk"""
        save_path = path or self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path=None):
        """Load model from disk"""
        load_path = path or self.model_path
        
        if os.path.exists(load_path):
            model_data = joblib.load(load_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.metrics = model_data.get('metrics', {})
            self.feature_importance = model_data.get('feature_importance', {})
            logger.info(f"Model loaded from {load_path}")
            return True
        else:
            logger.warning(f"Model file {load_path} not found")
            return False