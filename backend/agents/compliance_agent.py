from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
from .base_agent import BaseAgent

class ComplianceAgent(BaseAgent):
    """Agent responsible for regulatory compliance checks"""
    
    def __init__(self, bedrock_client, vector_store):
        super().__init__("ComplianceAgent", bedrock_client, vector_store)
        self.regulations = self._load_regulations()
        
    def _load_regulations(self) -> Dict:
        """Load regulatory requirements"""
        return {
            'aml': {
                'ctr_threshold': 10000,
                'sar_threshold': 5000,
                'reporting_deadline_days': 15
            },
            'kyc': {
                'requires_verification': True,
                'enhanced_due_diligence': ['PEP', 'high_risk_jurisdiction']
            },
            'data_privacy': {
                'data_retention_days': 2555,  # 7 years
                'requires_consent': True
            }
        }
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_activity("Starting compliance checks")
        
        transactions = state.get('assessed_transactions', [])
        audit_findings = state.get('audit_findings', [])
        
        compliance_violations = []
        regulatory_reports = []
        
        for transaction in transactions:
            # Check AML compliance
            aml_violations = self._check_aml_compliance(transaction)
            compliance_violations.extend(aml_violations)
            
            # Generate CTR if needed
            if transaction.get('amount', 0) > self.regulations['aml']['ctr_threshold']:
                ctr_report = self._generate_ctr(transaction)
                regulatory_reports.append(ctr_report)
        
        # Check for SAR requirements based on audit findings
        for finding in audit_findings:
            if self._requires_sar(finding):
                sar_report = self._generate_sar(finding, transactions)
                regulatory_reports.append(sar_report)
        
        # Get latest regulatory updates from knowledge base
        reg_updates = self.query_knowledge_base("recent regulatory changes in banking")
        
        # Generate compliance report using LLM
        compliance_report_prompt = f"""
        Generate a compliance report based on:
        
        Violations: {json.dumps(compliance_violations[:5])}
        Regulatory Reports: {json.dumps(regulatory_reports[:3])}
        Regulatory Updates: {reg_updates}
        
        Include:
        1. Overall compliance status
        2. Critical violations requiring immediate action
        3. Regulatory reporting status
        4. Recommendations for compliance improvement
        """
        
        compliance_report = self.invoke_llm(compliance_report_prompt)
        
        # Update state
        state['compliance_violations'] = compliance_violations
        state['regulatory_reports'] = regulatory_reports
        state['compliance_report'] = compliance_report
        state['compliance_score'] = self._calculate_compliance_score(compliance_violations, len(transactions))
        
        self.update_memory('compliance_score', state['compliance_score'])
        
        self.log_activity(f"Compliance checks complete. Score: {state['compliance_score']}")
        
        return state
    
    def _check_aml_compliance(self, transaction: Dict) -> List[Dict]:
        """Check AML compliance for transaction"""
        violations = []
        
        # Check for CTR requirement
        if transaction.get('amount', 0) > self.regulations['aml']['ctr_threshold']:
            if not transaction.get('ctr_filed', False):
                violations.append({
                    'type': 'CTR_REQUIRED',
                    'severity': 'HIGH',
                    'transaction_id': transaction.get('transaction_id'),
                    'description': 'Currency Transaction Report required',
                    'deadline_days': self.regulations['aml']['reporting_deadline_days']
                })
        
        # Check for suspicious activity
        if transaction.get('risk_level') == 'HIGH' and transaction.get('amount', 0) > self.regulations['aml']['sar_threshold']:
            violations.append({
                'type': 'POTENTIAL_SAR',
                'severity': 'HIGH',
                'transaction_id': transaction.get('transaction_id'),
                'description': 'Potential Suspicious Activity Report required',
                'risk_score': transaction.get('risk_score')
            })
        
        return violations
    
    def _requires_sar(self, finding: Dict) -> bool:
        """Determine if finding requires SAR"""
        return (
            finding.get('severity') == 'HIGH' and
            finding.get('type') in ['HIGH_VALUE', 'HIGH_VELOCITY', 'UNUSUAL_PATTERN']
        )
    
    def _generate_ctr(self, transaction: Dict) -> Dict:
        """Generate Currency Transaction Report"""
        return {
            'report_type': 'CTR',
            'transaction_id': transaction.get('transaction_id'),
            'amount': transaction.get('amount'),
            'date': transaction.get('timestamp'),
            'customer_info': {
                'account_id': transaction.get('account_id'),
                'location': transaction.get('location')
            },
            'filing_deadline': (datetime.now() + timedelta(days=15)).isoformat(),
            'status': 'PENDING'
        }
    
    def _generate_sar(self, finding: Dict, transactions: List[Dict]) -> Dict:
        """Generate Suspicious Activity Report"""
        related_transactions = [
            t for t in transactions[-10:]  # Last 10 transactions
            if t.get('account_id') == finding.get('account_id')
        ]
        
        return {
            'report_type': 'SAR',
            'finding_id': finding.get('id'),
            'description': finding.get('description'),
            'related_transactions': len(related_transactions),
            'total_amount': sum(t.get('amount', 0) for t in related_transactions),
            'filing_deadline': (datetime.now() + timedelta(days=30)).isoformat(),
            'priority': 'HIGH'
        }
    
    def _calculate_compliance_score(self, violations: List, total_transactions: int) -> float:
        """Calculate overall compliance score"""
        if total_transactions == 0:
            return 1.0
        
        # Weight violations by severity
        severity_weights = {'HIGH': 1.0, 'MEDIUM': 0.5, 'LOW': 0.2}
        violation_score = sum(severity_weights.get(v.get('severity'), 0) for v in violations)
        
        # Normalize score
        max_possible_score = total_transactions * 1.0
        compliance_score = max(0, 1 - (violation_score / max_possible_score))
        
        return compliance_score