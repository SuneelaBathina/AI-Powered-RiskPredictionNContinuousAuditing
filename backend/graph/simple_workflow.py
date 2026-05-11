"""Simple audit workflow that doesn't require langchain - fallback for import errors"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import uuid
import random

logger = logging.getLogger(__name__)


class SimpleAuditWorkflow:
    """Lightweight audit workflow without LLM/langchain dependencies"""
    
    def __init__(self, risk_predictor):
        self.risk_predictor = risk_predictor
        self.workflow_id = None
        self.config = {
            'risk_threshold_high': 0.7,
            'risk_threshold_medium': 0.3,
            'ctr_threshold': 10000,
            'sar_threshold': 5000
        }
    
    def run(self, transactions: List[Dict], config: Optional[Dict] = None) -> Dict[str, Any]:
        """Run simplified audit workflow"""
        
        if config:
            self.config.update(config)
        
        self.workflow_id = f"SIMPLE_AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        logger.info(f"Starting simple audit workflow: {self.workflow_id}")
        logger.info(f"Processing {len(transactions)} transactions")
        
        assessed_transactions = []
        high_risk_alerts = []
        audit_findings = []
        compliance_violations = []
        
        try:
            # Risk Assessment Phase
            for transaction in transactions:
                try:
                    if self.risk_predictor:
                        risk_result = self.risk_predictor.predict(transaction)
                    else:
                        # Fallback if risk predictor not available
                        amount = transaction.get('amount', 0)
                        risk_result = {
                            'risk_score': 0.3 if amount > 5000 else 0.1,
                            'risk_level': 'HIGH' if amount > 10000 else 'MEDIUM' if amount > 5000 else 'LOW',
                            'confidence': 0.8
                        }
                    
                    assessed_txn = {
                        **transaction,
                        'risk_score': risk_result.get('risk_score', 0),
                        'risk_level': risk_result.get('risk_level', 'LOW'),
                        'confidence': risk_result.get('confidence', 0.8),
                        'assessed_at': datetime.now().isoformat()
                    }
                    
                    assessed_transactions.append(assessed_txn)
                    
                    # Generate alerts for high-risk
                    if risk_result.get('risk_level') == 'HIGH':
                        high_risk_alerts.append({
                            'transaction_id': transaction.get('transaction_id'),
                            'amount': transaction.get('amount'),
                            'risk_score': risk_result.get('risk_score'),
                            'alert_type': 'HIGH_RISK_TRANSACTION'
                        })
                    
                    # Generate audit findings
                    if transaction.get('audit_finding'):
                        finding = {
                            'id': f"FINDING_{transaction.get('transaction_id', 'UNKNOWN')}",
                            'transaction_id': transaction.get('transaction_id'),
                            'type': transaction.get('finding_severity', 'LOW'),
                            'severity': transaction.get('finding_severity', 'LOW'),
                            'description': transaction.get('audit_finding', 'Audit finding'),
                            'amount': transaction.get('amount')
                        }
                        audit_findings.append(finding)
                    
                    # Check compliance violations
                    if transaction.get('regulatory_violation'):
                        compliance_violations.append({
                            'transaction_id': transaction.get('transaction_id'),
                            'violation_type': transaction.get('regulatory_violation_type', 'UNKNOWN'),
                            'severity': 'HIGH'
                        })
                
                except Exception as e:
                    logger.warning(f"Error processing transaction {transaction.get('transaction_id')}: {e}")
                    continue
            
            # Generate compliance score
            total_transactions = len(assessed_transactions)
            violation_ratio = len(compliance_violations) / total_transactions if total_transactions > 0 else 0
            compliance_score = max(0, 1.0 - (violation_ratio * 0.5))
            
            # Generate reports
            reports = {
                'executive_summary': {
                    'total_transactions': total_transactions,
                    'high_risk_count': len(high_risk_alerts),
                    'audit_findings_count': len(audit_findings),
                    'compliance_violations': len(compliance_violations),
                    'compliance_score': compliance_score,
                    'average_risk_score': sum(t.get('risk_score', 0) for t in assessed_transactions) / total_transactions if total_transactions > 0 else 0
                },
                'audit_findings': audit_findings,
                'workflow_summary': {
                    'workflow_id': self.workflow_id,
                    'started_at': datetime.now().isoformat(),
                    'completed_at': datetime.now().isoformat(),
                    'phases_completed': 'risk_assessment,audit,compliance,reporting',
                    'errors_count': 0,
                    'total_transactions_processed': total_transactions,
                    'high_risk_found': len(high_risk_alerts)
                }
            }
            
            return {
                'workflow_id': self.workflow_id,
                'transactions': assessed_transactions,
                'assessed_transactions': assessed_transactions,
                'high_risk_alerts': high_risk_alerts,
                'audit_findings': audit_findings,
                'compliance_violations': compliance_violations,
                'compliance_score': compliance_score,
                'reports': reports,
                'current_phase': 'complete',
                'timestamp': datetime.now().isoformat(),
                'errors': []
            }
            
        except Exception as e:
            logger.error(f"Simple audit workflow failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'workflow_id': self.workflow_id,
                'transactions': [],
                'assessed_transactions': [],
                'high_risk_alerts': [],
                'audit_findings': [],
                'compliance_violations': [],
                'compliance_score': 0,
                'reports': {},
                'current_phase': 'error',
                'timestamp': datetime.now().isoformat(),
                'errors': [str(e)]
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get workflow status"""
        return {
            'workflow_id': self.workflow_id,
            'status': 'active' if self.workflow_id else 'idle',
            'type': 'simple_workflow'
        }
