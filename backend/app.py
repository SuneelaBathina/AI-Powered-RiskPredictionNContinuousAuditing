from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from config import Config
import logging
import os
import sys
from datetime import datetime

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.risk_model import RiskPredictor
from api.routes import register_routes
from api.websocket import register_socket_handlers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize components
risk_predictor = None
vector_store = None
audit_workflow = None

def create_sample_data():
    """Create sample transaction data for training"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    logger.info("Creating sample transaction data...")
    
    np.random.seed(42)
    n_samples = 10000
    
    # Generate timestamps (last 365 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    timestamps = [start_date + timedelta(
        days=np.random.randint(0, 365),
        hours=np.random.randint(0, 24),
        minutes=np.random.randint(0, 60)
    ) for _ in range(n_samples)]
    
    # Generate account ages (1 day to 10 years)
    account_ages = np.random.randint(1, 3650, n_samples)
    
    # Generate transaction amounts with realistic distribution
    amounts = np.random.lognormal(mean=6, sigma=1, size=n_samples).round(2)
    
    # Generate account balances
    balances = np.random.uniform(100, 100000, n_samples).round(2)
    
    # Generate previous transactions (0-50 in last 24h)
    prev_txns = np.random.poisson(lam=5, size=n_samples)
    
    # Generate average transaction amounts
    avg_amounts = np.random.lognormal(mean=5.5, sigma=0.8, size=n_samples).round(2)
    
    # Generate categorical features
    transaction_types = np.random.choice(
        ['PURCHASE', 'WITHDRAWAL', 'DEPOSIT', 'TRANSFER', 'PAYMENT'], 
        n_samples, 
        p=[0.4, 0.2, 0.15, 0.15, 0.1]
    )
    
    locations = np.random.choice(
        ['NY', 'CA', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'], 
        n_samples
    )
    
    device_types = np.random.choice(
        ['MOBILE', 'DESKTOP', 'TABLET', 'ATM'], 
        n_samples,
        p=[0.4, 0.3, 0.2, 0.1]
    )
    
    channels = np.random.choice(
        ['online', 'mobile', 'branch', 'atm'], 
        n_samples,
        p=[0.3, 0.3, 0.2, 0.2]
    )
    
    hours = np.random.randint(0, 24, n_samples)
    days_of_week = np.random.randint(0, 7, n_samples)
    is_weekend = (days_of_week >= 5).astype(int)
    
    # Generate fraud labels based on realistic patterns
    fraud_prob = np.zeros(n_samples)
    
    # High amount increases fraud probability
    fraud_prob += (amounts > 5000) * 0.3
    fraud_prob += (amounts > 10000) * 0.3
    
    # Unusual hours increase fraud probability
    fraud_prob += ((hours < 5) | (hours > 23)) * 0.2
    
    # High velocity increases fraud probability
    fraud_prob += (prev_txns > 15) * 0.2
    
    # New accounts are higher risk
    fraud_prob += (account_ages < 30) * 0.15
    
    # Certain transaction types are higher risk
    fraud_prob += (transaction_types == 'WITHDRAWAL') * 0.1
    fraud_prob += (transaction_types == 'TRANSFER') * 0.1
    
    # Location risk (simplified)
    high_risk_locations = ['CA', 'FL', 'TX']
    fraud_prob += np.isin(locations, high_risk_locations) * 0.1
    
    # Cap probability
    fraud_prob = np.clip(fraud_prob, 0, 0.95)
    
    # Generate fraud labels
    is_fraud = (np.random.random(n_samples) < fraud_prob).astype(int)
    
    # Create DataFrame
    data = {
        'transaction_id': [f'TXN_{str(i).zfill(6)}' for i in range(n_samples)],
        'timestamp': timestamps,
        'account_id': [f'ACC_{str(np.random.randint(1, 1001)).zfill(4)}' for _ in range(n_samples)],
        'amount': amounts,
        'account_age_days': account_ages,
        'account_balance': balances,
        'previous_transactions_24h': prev_txns,
        'avg_transaction_amount_30d': avg_amounts,
        'transaction_type': transaction_types,
        'location': locations,
        'device_type': device_types,
        'channel': channels,
        'hour': hours,
        'day_of_week': days_of_week,
        'is_weekend': is_weekend,
        'is_fraud': is_fraud
    }
    
    df = pd.DataFrame(data)
    
    logger.info(f"Created sample dataset with {len(df)} transactions")
    logger.info(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    logger.info(f"Average amount: ${df['amount'].mean():.2f}")
    logger.info(f"Max amount: ${df['amount'].max():.2f}")
    
    return df

def train_model():
    """Train the risk prediction model"""
    global risk_predictor
    
    logger.info("=" * 60)
    logger.info("Training Risk Prediction Model")
    logger.info("=" * 60)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)
    
    # Check if data exists
    data_path = 'data/financial_transactions.csv'
    if not os.path.exists(data_path):
        logger.info("No existing data found. Creating sample data...")
        df = create_sample_data()
        df.to_csv(data_path, index=False)
        logger.info(f"Sample data saved to {data_path}")
    else:
        logger.info(f"Loading existing data from {data_path}")
        import pandas as pd
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} transactions")
    
    # Initialize and train model
    try:
        risk_predictor = RiskPredictor()
        
        # Check if we have a pre-trained model
        model_path = 'models/saved/risk_model.pkl'
        if os.path.exists(model_path):
            logger.info("Loading existing model...")
            if risk_predictor.load_model():
                logger.info("✓ Model loaded successfully")
                
                # Test the model
                test_model()
                return True
        
        # Train new model
        logger.info("Training new XGBoost model...")
        risk_predictor.train_from_csv(data_path)
        
        # Save the model
        risk_predictor.save_model()
        logger.info("✓ Model training complete and saved")
        
        # Test the model
        test_model()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error training model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Create a simple fallback model
        logger.info("Creating fallback rule-based model...")
        risk_predictor = FallbackRiskPredictor()
        return False

def test_model():
    """Test the model with a sample transaction"""
    global risk_predictor
    
    if not risk_predictor:
        logger.warning("No model to test")
        return
    
    try:
        test_cases = [
            {
                'name': 'Normal Transaction',
                'data': {
                    'amount': 500,
                    'account_age_days': 1000,
                    'account_balance': 25000,
                    'previous_transactions_24h': 2,
                    'avg_transaction_amount_30d': 450,
                    'transaction_type': 'PURCHASE',
                    'location': 'NY',
                    'hour': 14,
                    'is_weekend': 0
                }
            },
            {
                'name': 'Suspicious Transaction',
                'data': {
                    'amount': 25000,
                    'account_age_days': 5,
                    'account_balance': 5000,
                    'previous_transactions_24h': 20,
                    'avg_transaction_amount_30d': 300,
                    'transaction_type': 'WITHDRAWAL',
                    'location': 'CA',
                    'hour': 3,
                    'is_weekend': 1
                }
            }
        ]
        
        logger.info("Testing model with sample transactions:")
        for test in test_cases:
            result = risk_predictor.predict(test['data'])
            logger.info(f"  {test['name']}: Risk Score={result['risk_score']:.3f}, "
                       f"Level={result['risk_level']}, Confidence={result['confidence']:.3f}")
        
        logger.info("✓ Model test successful")
        
    except Exception as e:
        logger.error(f"✗ Model test failed: {e}")

class FallbackRiskPredictor:
    """Simple fallback predictor when XGBoost fails"""
    
    def __init__(self):
        logger.info("Initializing fallback risk predictor")
    
    def predict(self, data):
        """Simple rule-based prediction"""
        amount = float(data.get('amount', 1000))
        account_age = int(data.get('account_age_days', 365))
        prev_txns = int(data.get('previous_transactions_24h', 5))
        hour = int(data.get('hour', 12))
        
        # Calculate risk score
        risk_score = 0.2  # Base score
        
        if amount > 10000:
            risk_score += 0.5
        elif amount > 5000:
            risk_score += 0.3
        
        if account_age < 30:
            risk_score += 0.2
        
        if prev_txns > 15:
            risk_score += 0.3
        
        if hour < 6 or hour > 23:
            risk_score += 0.2
        
        risk_score = min(risk_score, 0.95)
        
        if risk_score >= 0.7:
            risk_level = 'HIGH'
        elif risk_score >= 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'confidence': 0.7,
            'prediction': 1 if risk_score > 0.5 else 0
        }
    
    def predict_batch(self, df):
        results = []
        for _, row in df.iterrows():
            results.append(self.predict(row.to_dict()))
        return results
    
    def load_model(self):
        return True
    
    def save_model(self):
        pass

# Initialize the system on startup
logger.info("=" * 60)
logger.info("Starting AI-Powered Risk & Audit System")
logger.info("=" * 60)

# Train the model
train_model()

# Import and register routes
try:
    from rag.vector_store import VectorStore
    from graph.workflow import AuditWorkflow
    from aws_integration.bedrock_client import MockBedrockClient
    
    # Initialize vector store
    logger.info("Initializing vector store...")
    vector_store = VectorStore()
    
    # Initialize audit workflow
    logger.info("Initializing audit workflow...")
    bedrock_client = MockBedrockClient()
    audit_workflow = AuditWorkflow(risk_predictor, vector_store, bedrock_client)
    logger.info("✓ Audit workflow initialized")
    
except Exception as e:
    logger.error(f"✗ Error initializing other components: {e}")
    vector_store = None
    audit_workflow = None

# Register routes
try:
    register_routes(app, risk_predictor, audit_workflow, vector_store)
    logger.info("✓ Routes registered successfully")
except Exception as e:
    logger.error(f"✗ Error registering routes: {e}")

# Register socket handlers
try:
    register_socket_handlers(socketio, audit_workflow)
    logger.info("✓ Socket handlers registered successfully")
except Exception as e:
    logger.error(f"✗ Error registering socket handlers: {e}")

# @app.route('/')
# def home():
#     return jsonify({
#         'name': 'AI-Powered Financial Risk & Audit System',
#         'version': '1.0.0',
#         'status': 'running',
#         'model_status': 'trained' if risk_predictor else 'not trained',
#         'timestamp': datetime.now().isoformat()
#     })
# @app.route('/')
# def home():
#     """Root endpoint"""
#     return jsonify({
#         'name': 'AI-Powered Financial Risk & Audit System',
#         'version': '1.0.0',
#         'status': 'running',
#         'timestamp': datetime.now().isoformat(),
#         'endpoints': [
#             '/',
#             '/api/health',
#             '/api/test',
#             '/api/routes',
#             '/api/status',
#             '/api/transactions',
#             '/api/predict-risk',
#             '/api/test-predict'
#         ]
#     })

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model': {
            'initialized': risk_predictor is not None,
            'type': type(risk_predictor).__name__ if risk_predictor else None
        },
        'vector_store': vector_store is not None,
        'audit_workflow': audit_workflow is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Server starting on http://localhost:5000")
    logger.info("=" * 60)
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Starting Flask server on http://localhost:5000")
    logger.info("Available endpoints:")
    logger.info("  - http://localhost:5000/")
    logger.info("  - http://localhost:5000/api/test")
    logger.info("  - http://localhost:5000/api/health")
    logger.info("  - http://localhost:5000/api/routes")
    logger.info("=" * 50)
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)