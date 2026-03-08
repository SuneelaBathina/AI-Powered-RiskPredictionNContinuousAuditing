import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import logging
from typing import Optional, Tuple, List, Dict
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataGenerator:
    """
    Generates synthetic financial transaction data for testing and development.
    Creates realistic transaction patterns with fraud indicators.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Configuration
        self.locations = ['NY', 'CA', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI',
                         'NJ', 'VA', 'WA', 'MA', 'IN', 'TN', 'MO', 'MD', 'WI', 'CO']
        
        self.transaction_types = ['DEPOSIT', 'WITHDRAWAL', 'TRANSFER', 'PAYMENT', 'PURCHASE']
        self.device_types = ['MOBILE', 'DESKTOP', 'TABLET', 'ATM', 'POS']
        self.merchant_categories = [
            'retail', 'grocery', 'restaurant', 'travel', 'entertainment',
            'utilities', 'healthcare', 'education', 'gambling', 'cryptocurrency',
            'money_transfer', 'electronics', 'jewelry', 'automotive', 'professional_services'
        ]
        self.channels = ['online', 'mobile_app', 'branch', 'atm', 'phone']
        self.currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY']
        
        # Risk weights for different categories
        self.high_risk_merchants = ['gambling', 'cryptocurrency', 'money_transfer', 'jewelry']
        self.medium_risk_merchants = ['electronics', 'travel', 'entertainment', 'automotive']
        
        # Create accounts with different profiles
        self.accounts = self._create_account_profiles(n_accounts=500)
    
    def _create_account_profiles(self, n_accounts: int = 500) -> List[Dict]:
        """Create customer accounts with different risk profiles"""
        accounts = []
        
        for i in range(n_accounts):
            account_id = f"ACC_{str(i+1).zfill(6)}"
            
            # Account age (in days)
            account_age = np.random.randint(1, 3650)  # 1 day to 10 years
            
            # Account type
            account_type = np.random.choice(['PERSONAL', 'BUSINESS', 'JOINT'], p=[0.7, 0.2, 0.1])
            
            # Risk profile (some accounts are naturally higher risk)
            if account_type == 'BUSINESS':
                base_risk = np.random.uniform(0.2, 0.8)
            else:
                base_risk = np.random.uniform(0.1, 0.5)
            
            # Location risk factor
            location = np.random.choice(self.locations)
            location_risk = 0.3 if location in ['NY', 'CA', 'FL', 'TX'] else 0.1
            
            accounts.append({
                'account_id': account_id,
                'account_type': account_type,
                'account_age_days': account_age,
                'base_risk_profile': base_risk,
                'location': location,
                'location_risk': location_risk,
                'avg_monthly_txns': np.random.poisson(lam=20),
                'avg_transaction_amount': np.random.exponential(scale=500),
                'account_balance': np.random.uniform(100, 100000),
                'credit_score': np.random.randint(300, 850),
                'is_high_risk_customer': np.random.choice([True, False], p=[0.1, 0.9])
            })
        
        return accounts
    
    def generate_transactions(self, n_transactions: int = 10000, 
                            fraud_rate: float = 0.05,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Generate synthetic transaction data
        
        Args:
            n_transactions: Number of transactions to generate
            fraud_rate: Proportion of fraudulent transactions (0.05 = 5%)
            start_date: Start date for transactions
            end_date: End date for transactions
            
        Returns:
            DataFrame with synthetic transaction data
        """
        logger.info(f"Generating {n_transactions} synthetic transactions with {fraud_rate*100}% fraud rate")
        
        if start_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
        elif end_date is None:
            end_date = datetime.now()
        
        transactions = []
        
        for i in range(n_transactions):
            # Select random account
            account = random.choice(self.accounts)
            
            # Generate timestamp (weighted towards recent dates)
            days_range = (end_date - start_date).days
            # More transactions in recent days (exponential weighting)
            days_ago = int(np.random.exponential(scale=days_range/3))
            days_ago = min(days_ago, days_range)
            timestamp = end_date - timedelta(days=days_ago)
            
            # Add random hour (with business hour bias)
            hour = self._generate_hour()
            timestamp = timestamp.replace(hour=hour, minute=random.randint(0, 59), 
                                        second=random.randint(0, 59))
            
            # Generate transaction amount (with patterns)
            amount = self._generate_amount(account)
            
            # Determine if transaction is fraudulent
            is_fraud = self._determine_fraud(account, amount, timestamp, fraud_rate)
            
            # Generate transaction details
            transaction = {
                'transaction_id': f"TXN_{str(i+1).zfill(8)}",
                'timestamp': timestamp,
                'account_id': account['account_id'],
                'account_type': account['account_type'],
                'account_age_days': account['account_age_days'],
                'account_balance': self._update_balance(account, amount, is_fraud),
                'amount': amount,
                'transaction_type': self._generate_transaction_type(account, timestamp),
                'location': account['location'],
                'merchant_category': self._generate_merchant_category(amount, is_fraud),
                'channel': self._generate_channel(timestamp, amount),
                'device_type': self._generate_device_type(amount, is_fraud),
                'currency': np.random.choice(['USD', 'EUR', 'GBP'], p=[0.8, 0.15, 0.05]),
                'is_international': np.random.choice([True, False], p=[0.1, 0.9]),
                'ip_address': f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
                'is_fraud': int(is_fraud),
                'fraud_type': self._get_fraud_type(is_fraud, amount, timestamp) if is_fraud else None
            }
            
            transactions.append(transaction)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1} transactions...")
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Log statistics
        self._log_statistics(df)
        
        return df
    
    def _generate_hour(self) -> int:
        """Generate hour with business hour bias"""
        # 70% chance of business hours (9-17), 30% chance of other hours
        if np.random.random() < 0.7:
            return np.random.randint(9, 18)
        else:
            return np.random.randint(0, 24)
    
    def _generate_amount(self, account: Dict) -> float:
        """Generate transaction amount based on account profile"""
        base_amount = account['avg_transaction_amount']
        
        # Add some randomness
        if account['account_type'] == 'BUSINESS':
            # Business accounts have larger transactions
            multiplier = np.random.lognormal(mean=1, sigma=0.5)
            amount = base_amount * multiplier
        else:
            # Personal accounts have more varied amounts
            if np.random.random() < 0.3:
                # Small transactions
                amount = np.random.exponential(scale=50)
            elif np.random.random() < 0.6:
                # Medium transactions
                amount = np.random.exponential(scale=base_amount)
            else:
                # Large transactions
                amount = np.random.exponential(scale=base_amount * 3)
        
        # Round to 2 decimal places
        return round(amount, 2)
    
    def _generate_transaction_type(self, account: Dict, timestamp: datetime) -> str:
        """Generate transaction type based on patterns"""
        if account['account_type'] == 'BUSINESS':
            weights = [0.2, 0.1, 0.3, 0.3, 0.1]  # More transfers/payments for business
        else:
            weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # More deposits/withdrawals for personal
        
        # Weekend effect
        if timestamp.weekday() >= 5:  # Weekend
            weights = [0.2, 0.2, 0.1, 0.4, 0.1]  # More purchases on weekend
        
        return np.random.choice(self.transaction_types, p=weights)
    
    def _generate_merchant_category(self, amount: float, is_fraud: bool) -> str:
        """Generate merchant category based on amount and fraud status"""
        if is_fraud:
            # Fraudulent transactions more likely in high-risk categories
            weights = [0.05] * len(self.merchant_categories)
            for i, cat in enumerate(self.merchant_categories):
                if cat in self.high_risk_merchants:
                    weights[i] = 0.15
                elif cat in self.medium_risk_merchants:
                    weights[i] = 0.1
            weights = [w/sum(weights) for w in weights]
        else:
            # Normal distribution for legitimate transactions
            if amount > 1000:
                # Large purchases more likely in certain categories
                weights = [0.1, 0.05, 0.15, 0.2, 0.1, 0.02, 0.02, 0.02, 0.01, 0.01, 0.02, 0.1, 0.1, 0.05, 0.05]
            else:
                weights = [0.2, 0.15, 0.2, 0.05, 0.1, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01, 0.03, 0.02, 0.02, 0.03]
            weights = [w/sum(weights) for w in weights]
        
        return np.random.choice(self.merchant_categories, p=weights)
    
    def _generate_channel(self, timestamp: datetime, amount: float) -> str:
        """Generate channel based on time and amount"""
        hour = timestamp.hour
        
        # Mobile app more common at night and weekends
        if timestamp.weekday() >= 5 or hour < 8 or hour > 20:
            mobile_prob = 0.5
        else:
            mobile_prob = 0.3
        
        # Large transactions more likely in branch
        if amount > 5000:
            branch_prob = 0.4
        else:
            branch_prob = 0.1
        
        weights = {
            'online': 0.3,
            'mobile_app': mobile_prob,
            'branch': branch_prob,
            'atm': 0.2,
            'phone': 0.1
        }
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return np.random.choice(list(weights.keys()), p=list(weights.values()))
    
    def _generate_device_type(self, amount: float, is_fraud: bool) -> str:
        """Generate device type"""
        if is_fraud:
            # Fraud more common on mobile
            weights = [0.6, 0.2, 0.1, 0.05, 0.05]
        else:
            if amount > 1000:
                # Large transactions more common on desktop
                weights = [0.2, 0.5, 0.1, 0.1, 0.1]
            else:
                weights = [0.4, 0.3, 0.1, 0.1, 0.1]
        
        return np.random.choice(self.device_types, p=weights)
    
    def _determine_fraud(self, account: Dict, amount: float, 
                        timestamp: datetime, base_fraud_rate: float) -> bool:
        """Determine if transaction should be marked as fraudulent"""
        
        # Base probability from account profile
        fraud_prob = account['base_risk_profile'] * base_fraud_rate * 2
        
        # Adjust based on amount
        avg_amount = account['avg_transaction_amount']
        if amount > avg_amount * 3:
            fraud_prob *= 2
        elif amount > avg_amount * 5:
            fraud_prob *= 3
        
        # Adjust based on time
        hour = timestamp.hour
        if hour < 5 or hour > 23:  # Late night
            fraud_prob *= 1.5
        
        # Adjust based on day
        if timestamp.weekday() >= 5:  # Weekend
            fraud_prob *= 1.2
        
        # Cap probability
        fraud_prob = min(fraud_prob, 0.95)
        
        return np.random.random() < fraud_prob
    
    def _get_fraud_type(self, is_fraud: bool, amount: float, timestamp: datetime) -> Optional[str]:
        """Get fraud type for fraudulent transactions"""
        if not is_fraud:
            return None
        
        fraud_types = []
        
        # Amount-based fraud
        if amount > 10000:
            fraud_types.append('STRUCTURING')
        elif amount > 5000:
            fraud_types.append('SUSPICIOUS_AMOUNT')
        
        # Time-based fraud
        hour = timestamp.hour
        if hour < 5 or hour > 23:
            fraud_types.append('UNUSUAL_HOUR')
        
        # Weekend fraud
        if timestamp.weekday() >= 5:
            fraud_types.append('WEEKEND_ACTIVITY')
        
        if not fraud_types:
            fraud_types.append('OTHER_SUSPICIOUS')
        
        return np.random.choice(fraud_types)
    
    def _update_balance(self, account: Dict, amount: float, is_fraud: bool) -> float:
        """Update account balance (simulated)"""
        # For fraud, balance might not actually change
        if is_fraud and np.random.random() < 0.3:
            return account['account_balance']
        
        # Simulate balance change
        if np.random.random() < 0.5:  # 50% chance of deposit
            new_balance = account['account_balance'] + amount
        else:  # 50% chance of withdrawal
            new_balance = max(0, account['account_balance'] - amount)
        
        return round(new_balance, 2)
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataframe"""
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                                   (df['day_of_week'] < 5)).astype(int)
        
        # Amount features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_category'] = pd.cut(df['amount'], 
                                       bins=[0, 100, 500, 1000, 5000, 100000],
                                       labels=['micro', 'small', 'medium', 'large', 'xlarge'])
        
        # Risk indicators (simplified)
        df['high_risk_merchant'] = df['merchant_category'].isin(self.high_risk_merchants).astype(int)
        df['medium_risk_merchant'] = df['merchant_category'].isin(self.medium_risk_merchants).astype(int)
        
        # Fraud flags (for training)
        df['is_fraud'] = df['is_fraud'].astype(int)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def _log_statistics(self, df: pd.DataFrame):
        """Log statistics about generated data"""
        logger.info("=" * 50)
        logger.info("Generated Data Statistics:")
        logger.info(f"Total transactions: {len(df)}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
        logger.info(f"Average amount: ${df['amount'].mean():.2f}")
        logger.info(f"Median amount: ${df['amount'].median():.2f}")
        logger.info(f"Max amount: ${df['amount'].max():.2f}")
        logger.info(f"Min amount: ${df['amount'].min():.2f}")
        logger.info("\nTransaction types:")
        logger.info(df['transaction_type'].value_counts())
        logger.info("\nFraud by transaction type:")
        logger.info(df.groupby('transaction_type')['is_fraud'].mean())
        logger.info("=" * 50)


def generate_synthetic_data(n_transactions: int = 10000, 
                          fraud_rate: float = 0.05,
                          output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to generate synthetic financial data
    
    Args:
        n_transactions: Number of transactions to generate
        fraud_rate: Proportion of fraudulent transactions
        output_file: Optional file path to save the data
        
    Returns:
        DataFrame with synthetic transaction data
    """
    generator = FinancialDataGenerator(random_seed=42)
    df = generator.generate_transactions(
        n_transactions=n_transactions,
        fraud_rate=fraud_rate
    )
    
    if output_file:
        df.to_csv(output_file, index=False)
        logger.info(f"Data saved to {output_file}")
    
    return df


def generate_training_test_data(n_train: int = 10000, 
                               n_test: int = 2000,
                               fraud_rate: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate separate training and test datasets
    
    Args:
        n_train: Number of training transactions
        n_test: Number of test transactions
        fraud_rate: Proportion of fraudulent transactions
        
    Returns:
        Tuple of (train_df, test_df)
    """
    generator = FinancialDataGenerator(random_seed=42)
    
    # Generate training data
    train_df = generator.generate_transactions(
        n_transactions=n_train,
        fraud_rate=fraud_rate
    )
    
    # Generate test data with different random seed
    generator.random_seed = 99
    test_df = generator.generate_transactions(
        n_transactions=n_test,
        fraud_rate=fraud_rate
    )
    
    return train_df, test_df


def generate_scenario_based_data(scenario: str = 'normal') -> pd.DataFrame:
    """
    Generate data for specific scenarios
    
    Args:
        scenario: 'normal', 'high_fraud', 'holiday', 'crisis'
        
    Returns:
        DataFrame with scenario-based data
    """
    generator = FinancialDataGenerator(random_seed=42)
    
    if scenario == 'high_fraud':
        # Generate data with high fraud rate
        df = generator.generate_transactions(n_transactions=5000, fraud_rate=0.15)
        
    elif scenario == 'holiday':
        # Generate data focused on holiday season
        end_date = datetime(2023, 12, 31)
        start_date = datetime(2023, 11, 15)
        df = generator.generate_transactions(
            n_transactions=5000,
            fraud_rate=0.08,
            start_date=start_date,
            end_date=end_date
        )
        
    elif scenario == 'crisis':
        # Generate data with economic crisis patterns
        df = generator.generate_transactions(n_transactions=5000, fraud_rate=0.12)
        # Increase withdrawal frequency
        mask = np.random.random(len(df)) < 0.3
        df.loc[mask, 'transaction_type'] = 'WITHDRAWAL'
        
    else:  # normal
        df = generator.generate_transactions(n_transactions=5000, fraud_rate=0.05)
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Generating synthetic financial data...")
    
    # Generate 10000 transactions with 5% fraud rate
    df = generate_synthetic_data(
        n_transactions=10000,
        fraud_rate=0.05,
        output_file='data/financial_transactions.csv'
    )
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData info:")
    print(df.info())