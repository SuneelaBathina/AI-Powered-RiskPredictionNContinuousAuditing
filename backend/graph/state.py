from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime
import operator

class AuditState(TypedDict):
    """State schema for the audit workflow"""
    # Input
    transactions: List[Dict[str, Any]]
    audit_period: Dict[str, str]
    config: Dict[str, Any]
    
    # Processed data
    assessed_transactions: Annotated[List[Dict[str, Any]], operator.add]
    high_risk_alerts: Annotated[List[Dict[str, Any]], operator.add]
    risk_metrics: Dict[str, Any]
    
    # Audit findings
    audit_findings: Annotated[List[Dict[str, Any]], operator.add]
    audit_summary: Dict[str, Any]
    audit_recommendations: str
    
    # Compliance
    compliance_violations: Annotated[List[Dict[str, Any]], operator.add]
    regulatory_reports: Annotated[List[Dict[str, Any]], operator.add]
    compliance_score: float
    compliance_report: str
    
    # Investigations
    investigations: Annotated[List[Dict[str, Any]], operator.add]
    investigation_reports: Annotated[List[Dict[str, Any]], operator.add]
    
    # Final reports
    reports: Dict[str, Any]
    
    # Workflow control
    current_phase: str
    timestamp: str
    errors: List[str]
    workflow_id: str


class AuditConfig(TypedDict):
    """Configuration for audit workflow"""
    enable_parallel_processing: bool
    risk_threshold_high: float
    risk_threshold_medium: float
    ctr_threshold: float
    sar_threshold: float
    max_investigations: int
    auto_escalate: bool
    notification_emails: List[str]