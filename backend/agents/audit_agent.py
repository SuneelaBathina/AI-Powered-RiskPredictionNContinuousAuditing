from typing import Dict, Any, List
import json
from datetime import datetime
from .base_agent import BaseAgent

class AuditAgent(BaseAgent):
    """Agent responsible for performing audit procedures"""
    
    def __init__(self, bedrock_client, vector_store):
        super().__init__("AuditAgent", bedrock_client, vector_store)
        self.audit_procedures = self._initialize_audit_procedures()
        
    def _initialize_audit_procedures(self) -> List[Dict]:
        """Initialize standard audit procedures"""
        return [
            {
                'name': 'High Value Transaction Review',
                'threshold': 10000,
                'procedure': 'Review all transactions above threshold for proper documentation'
            },
            {
                'name': 'Unusual Pattern Detection',
                'procedure': 'Identify transactions with unusual patterns or frequencies'
            },
            {
                'name': 'Compliance Check',
                'procedure': 'Verify compliance with AML and KYC regulations'
            },
            {
                'name': 'Segregation of Duties',
                'procedure': 'Check for proper segregation in transaction approval'
            },
            {
                'name': 'Reconciliation Review',
                'procedure': 'Verify transaction reconciliation with statements'
            }
        ]
        
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_activity("Starting audit procedures")
        
        assessed_transactions = state.get('assessed_transactions', [])
        high_risk_alerts = state.get('high_risk_alerts', [])
        
        if not assessed_transactions:
            self.log_activity("No transactions to audit", "warning")
            return state
        
        # Select audit sample
        audit_sample = self._select_audit_sample(assessed_transactions, high_risk_alerts)
        
        # Perform audit procedures
        audit_findings = []
        for transaction in audit_sample:
            findings = self._audit_transaction(transaction)
            audit_findings.extend(findings)
        
        # Generate audit summary
        audit_summary = self._generate_audit_summary(audit_findings)
        
        # Get regulatory context from knowledge base
        regulatory_context = self.query_knowledge_base(
            "current AML regulations and compliance requirements"
        )
        
        # Generate recommendations using LLM
        recommendations_prompt = f"""
        Based on these audit findings and regulatory context, provide recommendations:
        
        Audit Findings: {json.dumps(audit_findings[:5])}
        Regulatory Context: {regulatory_context}
        
        Provide:
        1. Priority actions needed
        2. Process improvements
        3. Compliance gaps to address
        """
        
        recommendations = self.invoke_llm(recommendations_prompt)
        
        # Update state
        state['audit_findings'] = audit_findings
        state['audit_summary'] = audit_summary
        state['audit_recommendations'] = recommendations
        state['audit_timestamp'] = datetime.now().isoformat()
        
        self.update_memory('last_audit_findings', len(audit_findings))
        
        self.log_activity(f"Audit complete. Found {len(audit_findings)} findings")
        
        return state
    
    def _select_audit_sample(self, transactions: List[Dict], alerts: List[Dict]) -> List[Dict]:
        """Select transactions for audit sampling"""
        sample = []
        
        # Include all high-risk transactions
        high_risk_txns = [t for t in transactions if t.get('risk_level') == 'HIGH']
        sample.extend(high_risk_txns)
        
        # Stratified random sampling of medium and low risk
        medium_risk = [t for t in transactions if t.get('risk_level') == 'MEDIUM']
        low_risk = [t for t in transactions if t.get('risk_level') == 'LOW']
        
        # Sample 20% of medium risk and 5% of low risk
        import random
        sample.extend(random.sample(medium_risk, min(len(medium_risk), int(len(medium_risk) * 0.2))))
        sample.extend(random.sample(low_risk, min(len(low_risk), int(len(low_risk) * 0.05))))
        
        return sample
    
    def _audit_transaction(self, transaction: Dict) -> List[Dict]:
        """Perform audit procedures on a single transaction"""
        findings = []
        
        # Check for high value
        if transaction.get('amount', 0) > 10000:
            findings.append({
                'type': 'HIGH_VALUE',
                'severity': 'MEDIUM',
                'transaction_id': transaction.get('transaction_id'),
                'description': f"High value transaction of ${transaction['amount']}",
                'procedure': 'Verify documentation and approval'
            })
        
        # Check for unusual timing
        if transaction.get('is_weekend') and transaction.get('amount', 0) > 5000:
            findings.append({
                'type': 'UNUSUAL_TIMING',
                'severity': 'LOW',
                'transaction_id': transaction.get('transaction_id'),
                'description': "Large transaction on weekend",
                'procedure': 'Review for unusual patterns'
            })
        
        # Check for velocity issues
        if transaction.get('hourly_velocity', 0) > 10:
            findings.append({
                'type': 'HIGH_VELOCITY',
                'severity': 'MEDIUM',
                'transaction_id': transaction.get('transaction_id'),
                'description': f"High transaction velocity: {transaction['hourly_velocity']} per hour",
                'procedure': 'Investigate for potential fraud'
            })
        
        return findings
    
    def _generate_audit_summary(self, findings: List[Dict]) -> Dict:
        """Generate summary of audit findings"""
        severity_counts = {
            'HIGH': sum(1 for f in findings if f.get('severity') == 'HIGH'),
            'MEDIUM': sum(1 for f in findings if f.get('severity') == 'MEDIUM'),
            'LOW': sum(1 for f in findings if f.get('severity') == 'LOW')
        }
        
        return {
            'total_findings': len(findings),
            'severity_breakdown': severity_counts,
            'finding_types': list(set(f.get('type') for f in findings)),
            'requires_escalation': severity_counts['HIGH'] > 0
        }