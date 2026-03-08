import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # AWS
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    BEDROCK_MODEL_ID = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-v2')
    
    # Database
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # Model paths
    MODEL_PATH = 'models/saved/risk_model.pkl'
    SCALER_PATH = 'models/saved/scaler.pkl'
    ENCODER_PATH = 'models/saved/encoders.pkl'
    
    # RAG Configuration
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    VECTOR_STORE_PATH = 'rag/vector_store'
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Agent Configuration
    MAX_AGENT_ITERATIONS = 10
    AGENT_TIMEOUT = 30  # seconds
    ENABLE_PARALLEL_AGENTS = True
    
    # Risk thresholds
    HIGH_RISK_THRESHOLD = 0.7
    MEDIUM_RISK_THRESHOLD = 0.3
    AML_THRESHOLD = 10000  # $10,000 for CTR