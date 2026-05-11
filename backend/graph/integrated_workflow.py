"""
Integrated Audit Workflow - Orchestrates audit operations for comprehensive auditing
without requiring LangGraph or heavy LLM dependencies
"""

import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class IntegratedAuditWorkflow:
    """Orchestrates audit agents to process transactions and generate findings"""
    
    def __init__(self, risk_predictor, vector_store=None, bedrock_client=None):
        self.risk_predictor = risk_predictor
        self.vector_store = vector_store
        self.bedrock_client = bedrock_client
        self.current_workflow_id: Optional[str] = None
        self.current_state = {}
        self.status = "idle"
        
        logger.info("IntegratedAuditWorkflow initialized")
    
    def run(self, transactions: List[Dict], progress_callback=None) -> Dict[str, Any]:
        """
        Execute the complete audit workflow:
        1. Risk Assessment
        2. Audit Procedures
        3. Compliance Checking
        4. Report Generation
        
        Args:
            transactions: List of transaction dictionaries
            progress_callback: Optional callback function(phase_number, phase_name) to track progress
        """
        workflow_id = f"AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.current_workflow_id = workflow_id
        self.status = "processing"
        
        logger.info(f"Starting audit workflow: {workflow_id} with {len(transactions)} transactions")
        
        try:
            # Initialize state
            state = {
                'workflow_id': workflow_id,
                'transactions': transactions,
                'assessed_transactions': [],
                'high_risk_alerts': [],
                'audit_findings': [],
                'compliance_violations': [],
                'reports': {},
                'timestamp': datetime.now().isoformat(),
                'current_phase': 'initialization'
            }
            
            # Phase 1: Risk Assessment
            logger.info(f"[{workflow_id}] Phase 1: Risk Assessment")
            if progress_callback:
                progress_callback(0, 'risk_assessment')
            state = self._phase_risk_assessment(state)
            
            # Phase 2: Audit Procedures
            logger.info(f"[{workflow_id}] Phase 2: Audit Procedures")
            if progress_callback:
                progress_callback(1, 'audit_procedures')
            state = self._phase_audit_procedures(state)
            
            # Phase 3: Compliance Check
            logger.info(f"[{workflow_id}] Phase 3: Compliance Checks")
            if progress_callback:
                progress_callback(2, 'compliance_check')
            state = self._phase_compliance_check(state)
            
            # Phase 4: Investigation (optional, for high-risk transactions)
            if state.get('high_risk_alerts'):
                logger.info(f"[{workflow_id}] Phase 4: Investigation")
                if progress_callback:
                    progress_callback(3, 'investigation')
                state = self._phase_investigation(state)
            
            # Phase 5: Report Generation
            logger.info(f"[{workflow_id}] Phase 5: Report Generation")
            if progress_callback:
                progress_callback(4, 'report_generation')
            state = self._phase_report_generation(state)
            
            # Calculate compliance score
            state['compliance_score'] = self._calculate_compliance_score(state)
            state['current_phase'] = 'complete'
            self.status = "completed"
            
            logger.info(f"[{workflow_id}] Workflow completed successfully")
            logger.info(f"[{workflow_id}] Summary: {len(state.get('audit_findings', []))} findings, "
                       f"{len(state.get('high_risk_alerts', []))} alerts, "
                       f"compliance_score: {state.get('compliance_score', 0):.2f}")
            
            return state
            
        except Exception as e:
            logger.error(f"[{workflow_id}] Workflow failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.status = "failed"
            
            return {
                'workflow_id': workflow_id,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _phase_risk_assessment(self, state: Dict) -> Dict:
        """Phase 1: Assess transaction risks using ML model"""
        state['current_phase'] = 'risk_assessment'
        transactions = state.get('transactions', [])
        assessed_transactions = []
        high_risk_alerts = []
        
        logger.info(f"Assessing {len(transactions)} transactions for risk")
        
        for transaction in transactions:
            try:
                # Get ML-based risk score
                risk_result = self.risk_predictor.predict(transaction)
                
                assessed_txn = {
                    **transaction,
                    'risk_score': risk_result.get('risk_score', 0),
                    'risk_level': risk_result.get('risk_level', 'LOW'),
                    'confidence': risk_result.get('confidence', 0.5),
                    'assessed_at': state.get('timestamp'),
                    'risk_explanation': self._generate_risk_explanation(transaction, risk_result)
                }
                
                assessed_transactions.append(assessed_txn)
                
                # Generate alert for high-risk transactions
                if risk_result.get('risk_level') == 'HIGH':
                    high_risk_alerts.append({
                        'transaction_id': transaction.get('transaction_id'),
                        'risk_score': risk_result.get('risk_score'),
                        'amount': transaction.get('amount'),
                        'alert_level': 'HIGH',
                        'requires_immediate_review': True,
                        'detected_at': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.warning(f"Error assessing transaction {transaction.get('transaction_id')}: {e}")
                assessed_transactions.append({
                    **transaction,
                    'risk_score': 0.5,
                    'risk_level': 'MEDIUM',
                    'confidence': 0.3,
                    'error': str(e)
                })
        
        state['assessed_transactions'] = assessed_transactions
        state['high_risk_alerts'] = high_risk_alerts
        state['risk_metrics'] = self._calculate_risk_metrics(assessed_transactions)
        
        logger.info(f"Risk assessment complete: {len(high_risk_alerts)} high-risk transactions found")
        
        return state
    
    def _phase_audit_procedures(self, state: Dict) -> Dict:
        """Phase 2: Execute audit procedures on transactions"""
        state['current_phase'] = 'audit_procedures'
        assessed_transactions = state.get('assessed_transactions', [])
        high_risk_alerts = state.get('high_risk_alerts', [])
        
        audit_findings = []
        audit_procedures = self._get_audit_procedures()
        
        logger.info(f"Executing {len(audit_procedures)} audit procedures")
        
        # Create audit sample (include all high-risk, sample others)
        audit_sample = [t for t in assessed_transactions if t.get('risk_level') in ['HIGH', 'MEDIUM']]
        
        # Add random sample of low-risk
        low_risk = [t for t in assessed_transactions if t.get('risk_level') == 'LOW']
        if low_risk:
            import random
            sample_size = min(10, len(low_risk) // 10)  # 10% sample
            audit_sample.extend(random.sample(low_risk, sample_size))
        
        for procedure in audit_procedures:
            procedure_findings = self._execute_audit_procedure(procedure, audit_sample, state)
            audit_findings.extend(procedure_findings)
        
        state['audit_findings'] = audit_findings
        state['audit_sample_size'] = len(audit_sample)
        
        logger.info(f"Audit procedures complete: {len(audit_findings)} findings")
        
        return state
    
    def _phase_compliance_check(self, state: Dict) -> Dict:
        """Phase 3: Check regulatory compliance"""
        state['current_phase'] = 'compliance_check'
        transactions = state.get('assessed_transactions', [])
        audit_findings = state.get('audit_findings', [])
        
        violations = []
        regulations = self._get_regulations()
        
        logger.info("Checking compliance with regulations")
        
        for transaction in transactions:
            amount = transaction.get('amount', 0)
            
            # AML Compliance Check
            if amount > regulations['aml']['ctr_threshold']:
                violations.append({
                    'transaction_id': transaction.get('transaction_id'),
                    'type': 'AML_CTR_REQUIRED',
                    'severity': 'HIGH',
                    'description': f'Currency Transaction Report (CTR) required for amount ${amount}',
                    'regulation': 'AML',
                    'detected_at': datetime.now().isoformat()
                })
            
            # SAR Compliance Check
            if amount > regulations['aml']['sar_threshold'] and transaction.get('risk_level') == 'HIGH':
                violations.append({
                    'transaction_id': transaction.get('transaction_id'),
                    'type': 'AML_SAR_REQUIRED',
                    'severity': 'CRITICAL',
                    'description': f'Suspicious Activity Report (SAR) may be required for high-risk transaction of ${amount}',
                    'regulation': 'AML',
                    'detected_at': datetime.now().isoformat()
                })
            
            # KYC Compliance Check (enhanced for international)
            if transaction.get('location') == 'International' and amount > 5000:
                violations.append({
                    'transaction_id': transaction.get('transaction_id'),
                    'type': 'KYC_ENHANCED_DUE_DILIGENCE',
                    'severity': 'MEDIUM',
                    'description': 'Enhanced Due Diligence required for international transaction over $5000',
                    'regulation': 'KYC',
                    'detected_at': datetime.now().isoformat()
                })
        
        state['compliance_violations'] = violations
        state['compliance_metrics'] = {
            'total_reviewed': len(transactions),
            'violations_found': len(violations),
            'violation_types': list(set(v['type'] for v in violations))
        }
        
        logger.info(f"Compliance check complete: {len(violations)} violations found")
        
        return state
    
    def _phase_investigation(self, state: Dict) -> Dict:
        """Phase 4: Investigate high-risk transactions in detail"""
        state['current_phase'] = 'investigation'
        high_risk_alerts = state.get('high_risk_alerts', [])
        assessed_transactions = state.get('assessed_transactions', [])
        
        investigation_findings = []
        
        logger.info(f"Investigating {len(high_risk_alerts)} high-risk transactions")
        
        for alert in high_risk_alerts[:10]:  # Limit to top 10
            transaction_id = alert.get('transaction_id')
            transaction = next((t for t in assessed_transactions if t.get('transaction_id') == transaction_id), None)
            
            if transaction:
                investigation = self._investigate_transaction(transaction, alert, state)
                investigation_findings.append(investigation)
        
        state['investigation_findings'] = investigation_findings
        
        logger.info(f"Investigation complete: {len(investigation_findings)} investigations")
        
        return state
    
    def _phase_report_generation(self, state: Dict) -> Dict:
        """Phase 5: Generate comprehensive audit reports"""
        state['current_phase'] = 'report_generation'
        
        reports = {}
        
        # Executive Summary
        reports['executive_summary'] = self._generate_executive_summary(state)
        
        # Audit Findings Report
        reports['audit_findings'] = self._generate_audit_findings_report(state)
        
        # Compliance Report
        reports['compliance_report'] = self._generate_compliance_report(state)
        
        # Risk Report
        reports['risk_report'] = self._generate_risk_report(state)
        
        # Workflow Summary
        reports['workflow_summary'] = {
            'workflow_id': state.get('workflow_id'),
            'execution_time': self._calculate_execution_time(state),
            'transactions_processed': len(state.get('transactions', [])),
            'findings_count': len(state.get('audit_findings', [])),
            'violations_count': len(state.get('compliance_violations', [])),
            'alerts_generated': len(state.get('high_risk_alerts', [])),
            'status': 'completed',
            'generated_at': datetime.now().isoformat()
        }
        
        state['reports'] = reports
        
        logger.info("Report generation complete")
        
        return state
    
    # ========== Helper Methods ==========
    
    def _generate_risk_explanation(self, transaction: Dict, risk_result: Dict) -> str:
        """Generate explanation for transaction risk"""
        amount = transaction.get('amount', 0)
        txn_type = transaction.get('transaction_type', 'unknown')
        location = transaction.get('location', 'unknown')
        risk_level = risk_result.get('risk_level', 'UNKNOWN')
        
        explanations = {
            'HIGH': f"High-risk transaction detected: {txn_type} of ${amount} in {location}. "
                   f"Risk score: {risk_result.get('risk_score', 0):.2f}. Requires immediate review.",
            'MEDIUM': f"Medium-risk transaction: {txn_type} of ${amount} in {location}. "
                     f"Risk score: {risk_result.get('risk_score', 0):.2f}. Review recommended.",
            'LOW': f"Low-risk transaction: {txn_type} of ${amount} in {location}. "
                  f"Risk score: {risk_result.get('risk_score', 0):.2f}. Standard monitoring."
        }
        
        return explanations.get(risk_level, "Risk assessment complete")
    
    def _calculate_risk_metrics(self, assessed_transactions: List[Dict]) -> Dict:
        """Calculate risk metrics from assessed transactions"""
        if not assessed_transactions:
            return {'total': 0, 'high': 0, 'medium': 0, 'low': 0, 'avg_score': 0}
        
        high = len([t for t in assessed_transactions if t.get('risk_level') == 'HIGH'])
        medium = len([t for t in assessed_transactions if t.get('risk_level') == 'MEDIUM'])
        low = len([t for t in assessed_transactions if t.get('risk_level') == 'LOW'])
        
        avg_score = sum(t.get('risk_score', 0) for t in assessed_transactions) / len(assessed_transactions)
        
        return {
            'total': len(assessed_transactions),
            'high': high,
            'medium': medium,
            'low': low,
            'avg_score': round(avg_score, 3),
            'high_percentage': round(high / len(assessed_transactions) * 100, 2)
        }
    
    def _get_audit_procedures(self) -> List[Dict]:
        """Get standard audit procedures"""
        return [
            {
                'name': 'High Value Transaction Review',
                'threshold': 10000,
                'description': 'Review all transactions above threshold for proper documentation'
            },
            {
                'name': 'Unusual Pattern Detection',
                'description': 'Identify transactions with unusual patterns or frequencies'
            },
            {
                'name': 'Compliance Verification',
                'description': 'Verify compliance with AML and KYC regulations'
            },
            {
                'name': 'Segregation of Duties',
                'description': 'Check for proper segregation in transaction approval'
            }
        ]
    
    def _execute_audit_procedure(self, procedure: Dict, audit_sample: List[Dict], state: Dict) -> List[Dict]:
        """Execute a single audit procedure"""
        findings = []
        procedure_name = procedure.get('name')
        threshold = procedure.get('threshold', 0)
        
        for transaction in audit_sample:
            amount = transaction.get('amount', 0)
            
            # High Value Transaction Review
            if procedure_name == 'High Value Transaction Review' and amount > threshold:
                findings.append({
                    'transaction_id': transaction.get('transaction_id'),
                    'procedure': procedure_name,
                    'severity': 'MEDIUM',
                    'finding': f'High-value transaction of ${amount} requires documentation review',
                    'amount': amount,
                    'identified_at': datetime.now().isoformat()
                })
            
            # Unusual Pattern Detection
            elif procedure_name == 'Unusual Pattern Detection':
                if transaction.get('risk_score', 0) > 0.6:
                    findings.append({
                        'transaction_id': transaction.get('transaction_id'),
                        'procedure': procedure_name,
                        'severity': 'MEDIUM',
                        'finding': f'Unusual transaction pattern detected',
                        'risk_score': transaction.get('risk_score'),
                        'identified_at': datetime.now().isoformat()
                    })
        
        return findings
    
    def _get_regulations(self) -> Dict:
        """Get regulatory thresholds and requirements"""
        return {
            'aml': {
                'ctr_threshold': 10000,
                'sar_threshold': 5000,
                'reporting_deadline_days': 15
            },
            'kyc': {
                'requires_verification': True,
                'international_threshold': 5000
            },
            'data_privacy': {
                'data_retention_days': 2555  # 7 years
            }
        }
    
    def _investigate_transaction(self, transaction: Dict, alert: Dict, state: Dict) -> Dict:
        """Investigate a high-risk transaction in detail"""
        return {
            'transaction_id': transaction.get('transaction_id'),
            'investigation_type': 'high_risk_review',
            'risk_score': alert.get('risk_score'),
            'investigation_items': [
                'Verify customer identity and KYC documents',
                'Review transaction purpose and legitimacy',
                'Check against sanctions and watch lists',
                'Analyze transaction frequency and patterns',
                'Verify destination and source of funds'
            ],
            'preliminary_finding': 'Transaction flagged for further investigation',
            'recommended_action': 'Escalate to compliance team for review',
            'investigation_date': datetime.now().isoformat()
        }
    
    def _generate_executive_summary(self, state: Dict) -> Dict:
        """Generate executive summary"""
        transactions = state.get('transactions', [])
        findings = state.get('audit_findings', [])
        violations = state.get('compliance_violations', [])
        alerts = state.get('high_risk_alerts', [])
        
        return {
            'audit_period': f"{datetime.now() - timedelta(days=30)} to {datetime.now()}",
            'total_transactions': len(transactions),
            'transactions_audited': state.get('audit_sample_size', 0),
            'high_risk_count': len(alerts),
            'audit_findings_count': len(findings),
            'compliance_violations': len(violations),
            'compliance_score': state.get('compliance_score', 0),
            'overall_risk_level': self._assess_overall_risk(state),
            'key_recommendations': [
                'Review and escalate all high-risk transactions',
                'File required compliance reports (CTR, SAR)',
                'Implement enhanced due diligence for flagged accounts',
                'Review transaction monitoring controls',
                'Update audit procedures based on findings'
            ],
            'generated_date': datetime.now().isoformat()
        }
    
    def _generate_audit_findings_report(self, state: Dict) -> Dict:
        """Generate audit findings report"""
        return {
            'report_type': 'audit_findings',
            'total_findings': len(state.get('audit_findings', [])),
            'findings_by_severity': self._count_by_severity(state.get('audit_findings', [])),
            'findings': state.get('audit_findings', [])[:100],  # Limit to first 100
            'generated_date': datetime.now().isoformat()
        }
    
    def _generate_compliance_report(self, state: Dict) -> Dict:
        """Generate compliance report"""
        violations = state.get('compliance_violations', [])
        
        return {
            'report_type': 'compliance',
            'total_violations': len(violations),
            'violations_by_type': self._count_by_type(violations),
            'violations_by_severity': self._count_by_severity(violations),
            'violations': violations[:50],  # Limit to first 50
            'compliance_metrics': state.get('compliance_metrics', {}),
            'generated_date': datetime.now().isoformat()
        }
    
    def _generate_risk_report(self, state: Dict) -> Dict:
        """Generate risk report"""
        return {
            'report_type': 'risk_assessment',
            'risk_metrics': state.get('risk_metrics', {}),
            'high_risk_alerts': state.get('high_risk_alerts', [])[:50],
            'generated_date': datetime.now().isoformat()
        }
    
    def _count_by_severity(self, items: List[Dict]) -> Dict:
        """Count items by severity"""
        severity_map = {}
        for item in items:
            severity = item.get('severity', 'UNKNOWN')
            severity_map[severity] = severity_map.get(severity, 0) + 1
        return severity_map
    
    def _count_by_type(self, items: List[Dict]) -> Dict:
        """Count items by type"""
        type_map = {}
        for item in items:
            item_type = item.get('type', 'UNKNOWN')
            type_map[item_type] = type_map.get(item_type, 0) + 1
        return type_map
    
    def _calculate_compliance_score(self, state: Dict) -> float:
        """Calculate overall compliance score (0-1)"""
        violations = state.get('compliance_violations', [])
        
        if not violations:
            return 1.0
        
        # Score based on violation severity
        critical_count = len([v for v in violations if v.get('severity') == 'CRITICAL'])
        high_count = len([v for v in violations if v.get('severity') == 'HIGH'])
        medium_count = len([v for v in violations if v.get('severity') == 'MEDIUM'])
        
        # Deduct points based on violations
        score = 1.0
        score -= critical_count * 0.2
        score -= high_count * 0.1
        score -= medium_count * 0.02
        
        return max(0.0, min(1.0, score))
    
    def _assess_overall_risk(self, state: Dict) -> str:
        """Assess overall risk level"""
        risk_metrics = state.get('risk_metrics', {})
        
        high_percentage = risk_metrics.get('high_percentage', 0)
        
        if high_percentage > 20:
            return 'CRITICAL'
        elif high_percentage > 10:
            return 'HIGH'
        elif high_percentage > 5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_execution_time(self, state: Dict) -> str:
        """Calculate workflow execution time"""
        timestamp = state.get('timestamp')
        if timestamp:
            try:
                start_time = datetime.fromisoformat(timestamp)
                execution_time = (datetime.now() - start_time).total_seconds()
                return f"{execution_time:.2f} seconds"
            except:
                pass
        return "unknown"
    
    def get_status(self) -> Dict:
        """Get workflow status"""
        return {
            'workflow_id': self.current_workflow_id,
            'status': self.status,
            'current_phase': self.current_state.get('current_phase', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
