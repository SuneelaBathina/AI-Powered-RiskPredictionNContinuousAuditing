from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import re

class Transaction(BaseModel):
    """Transaction data model"""
    transaction_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    account_id: str
    amount: float = Field(..., gt=0, description="Transaction amount must be positive")
    transaction_type: str
    location: str
    merchant_category: Optional[str] = None
    channel: Optional[str] = None
    device_type: Optional[str] = None
    currency: str = "USD"
    account_balance: Optional[float] = None
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        if v > 1000000:
            raise ValueError('Amount exceeds maximum limit')
        return round(v, 2)
    
    @validator('transaction_id')
    def validate_transaction_id(cls, v):
        if v and not re.match(r'^[A-Z0-9_]+$', v):
            raise ValueError('Transaction ID must be alphanumeric and underscores only')
        return v

class RiskPredictionResponse(BaseModel):
    """Risk prediction response model"""
    transaction_id: str
    risk_score: float
    risk_level: str
    confidence: float
    timestamp: datetime
    
    @validator('risk_level')
    def validate_risk_level(cls, v):
        if v not in ['LOW', 'MEDIUM', 'HIGH']:
            raise ValueError('Risk level must be LOW, MEDIUM, or HIGH')
        return v

class AuditFinding(BaseModel):
    """Audit finding model"""
    id: str
    type: str
    severity: str
    transaction_id: str
    amount: float
    description: str
    recommendation: Optional[str] = None
    timestamp: datetime

class RiskMetrics(BaseModel):
    """Aggregate risk metrics"""
    total_transactions: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_risk_score: float
    high_risk_percentage: float
    risk_by_type: Dict[str, Any]
    risk_by_location: Dict[str, Any]
    time_series_risk: List[Dict[str, Any]]
    recent_alerts: List[Dict[str, Any]]

class AuditReport(BaseModel):
    """Audit report model"""
    report: Dict[str, Any]
    generated_at: datetime
    type: str

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    timestamp: datetime = Field(default_factory=datetime.now)
    status_code: int = 400

class SuccessResponse(BaseModel):
    """Success response model"""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)