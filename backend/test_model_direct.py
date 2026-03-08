import logging
from models.risk_model import RiskPredictor
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_direct():
    """Test the model directly without Flask"""
    
    print("\n" + "="*60)
    print("Testing Risk Predictor Directly")
    print("="*60)
    
    # Initialize predictor
    predictor = RiskPredictor()
    
    # Check if model exists
    model_path = 'models/saved/risk_model.pkl'
    if os.path.exists(model_path):
        print("Loading existing model...")
        if predictor.load_model():
            print("✓ Model loaded successfully")
    else:
        print("No existing model found. Training new model...")
        
        # Check if data exists
        data_path = 'data/financial_transactions.csv'
        if not os.path.exists(data_path):
            print("Creating sample data...")
            # Create sample data
            import pandas as pd
            import numpy as np
            
            n_samples = 5000
            data = {
                'amount': np.random.exponential(1000, n_samples),
                'account_age_days': np.random.randint(1, 3650, n_samples),
                'account_balance': np.random.uniform(100, 100000, n_samples),
                'previous_transactions_24h': np.random.poisson(5, n_samples),
                'avg_transaction_amount_30d': np.random.exponential(800, n_samples),
                'transaction_type': np.random.choice(['PURCHASE', 'WITHDRAWAL', 'DEPOSIT'], n_samples),
                'location': np.random.choice(['NY', 'CA', 'TX'], n_samples),
                'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
            }
            df = pd.DataFrame(data)
            df.to_csv(data_path, index=False)
            print(f"Created {n_samples} sample transactions")
        
        # Train model
        predictor.train_from_csv(data_path)
    
    # Test prediction
    print("\nTesting prediction:")
    test_data = {
        'amount': 15000,
        'account_age_days': 30,
        'account_balance': 5000,
        'previous_transactions_24h': 15,
        'avg_transaction_amount_30d': 500,
        'transaction_type': 'WITHDRAWAL',
        'location': 'CA'
    }
    
    result = predictor.predict(test_data)
    print(f"Risk Score: {result['risk_score']:.3f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)
    
    return predictor

if __name__ == "__main__":
    test_direct()