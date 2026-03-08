from typing import Dict, Any, List
import json
from datetime import datetime
from .base_agent import BaseAgent

class InvestigationAgent(BaseAgent):
    """Agent responsible for deep investigation of suspicious activities"""
    
    def __init__(self, bedrock_client, vector_store):
        super().__init__("InvestigationAgent", bedrock_client, vector_store)
        self.investigation_history = []
        
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_activity("Starting investigation of suspicious activities")
        
        high_risk_alerts = state.get('high_risk_alerts', [])
        audit_findings = state.get('audit_findings', [])
        compliance_violations = state.get('compliance_violations', [])
        
        investigations = []
        
        # Investigate high-risk alerts
        for alert in high_risk_alerts:
            investigation = self._investigate_alert(alert, state)
            investigations.append(investigation)
        
        # Investigate severe audit findings
        severe_findings = [f for f in audit_findings if f.get('severity') == 'HIGH']
        for finding in severe_findings[:5]:  # Limit to top 5
            investigation = self._investigate_finding(finding, state)
            investigations.append(investigation)
        
        # Prioritize investigations
        prioritized = self._prioritize_investigations(investigations)
        
        # Generate investigation reports
        investigation_reports = []
        for inv in prioritized[:3]:  # Generate reports for top 3
            report = self._generate_investigation_report(inv, state)
            investigation_reports.append(report)
        
        # Update state
        state['investigations'] = prioritized
        state['investigation_reports'] = investigation_reports
        state['investigation_timestamp'] = datetime.now().isoformat()
        
        self.investigation_history.extend(prioritized)
        
        self.log_activity(f"Completed {len(prioritized)} investigations")
        
        return state
    
    def _investigate_alert(self, alert: Dict, state: Dict) -> Dict:
        """Investigate a high-risk alert"""
        # Find related transactions
        related_txns = self._find_related_transactions(alert, state.get('transactions', []))
        
        # Analyze patterns
        patterns = self._analyze_patterns(related_txns)
        
        # Get contextual information from knowledge base
        context = self.query_knowledge_base(
            f"investigation procedures for {alert.get('alert_level')} risk transactions"
        )
        
        # Generate investigation summary using LLM
        summary_prompt = f"""
        Investigate this high-risk alert:
        Alert: {json.dumps(alert)}
        Related Transactions: {len(related_txns)}
        Patterns Detected: {json.dumps(patterns)}
        
        Provide:
        1. Root cause analysis
        2. Risk assessment
        3. Recommended actions
        4. Additional data needed
        """
        
        investigation_summary = self.invoke_llm(summary_prompt)
        
        return {
            'investigation_id': f"INV_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'alert': alert,
            'related_transactions_count': len(related_txns),
            'patterns': patterns,
            'summary': investigation_summary,
            'priority': self._calculate_priority(alert, patterns),
            'status': 'IN_PROGRESS'
        }
    
    def _investigate_finding(self, finding: Dict, state: Dict) -> Dict:
        """Investigate an audit finding"""
        # Find affected accounts
        affected_accounts = self._find_affected_accounts(finding, state)
        
        # Check historical patterns
        historical = self._check_historical_patterns(finding)
        
        return {
            'investigation_id': f"INV_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'finding': finding,
            'affected_accounts': affected_accounts,
            'historical_patterns': historical,
            'priority': 'HIGH' if finding.get('severity') == 'HIGH' else 'MEDIUM',
            'status': 'PENDING_REVIEW'
        }
    
    def _find_related_transactions(self, alert: Dict, transactions: List) -> List:
        """Find transactions related to an alert"""
        related = []
        
        # Find transactions from same account/IP/location
        for txn in transactions[-100:]:  # Look at last 100 transactions
            if (txn.get('account_id') == alert.get('account_id') or
                txn.get('location') == alert.get('location')):
                related.append(txn)
        
        return related
    
    def _analyze_patterns(self, transactions: List) -> Dict:
        """Analyze transaction patterns"""
        if not transactions:
            return {}
        
        amounts = [t.get('amount', 0) for t in transactions]
        
        return {
            'total_amount': sum(amounts),
            'avg_amount': sum(amounts) / len(amounts),
            'max_amount': max(amounts),
            'frequency': len(transactions),
            'time_pattern': self._detect_time_pattern(transactions)
        }
    
    def _detect_time_pattern(self, transactions: List) -> str:
        """Detect temporal patterns in transactions"""
        hours = [datetime.fromisoformat(t.get('timestamp')).hour for t in transactions if t.get('timestamp')]
        
        if not hours:
            return "unknown"
        
        avg_hour = sum(hours) / len(hours)
        
        if 0 <= avg_hour <= 5:
            return "late_night"
        elif 6 <= avg_hour <= 11:
            return "morning"
        elif 12 <= avg_hour <= 17:
            return "afternoon"
        else:
            return "evening"
    
    def _find_affected_accounts(self, finding: Dict, state: Dict) -> List:
        """Find accounts affected by a finding"""
        accounts = set()
        
        if 'account_id' in finding:
            accounts.add(finding['account_id'])
        
        # Add related accounts from transactions
        for txn in state.get('transactions', []):
            if txn.get('account_id') == finding.get('account_id'):
                accounts.add(txn.get('counterparty_account'))
        
        return list(accounts)
    
    def _check_historical_patterns(self, finding: Dict) -> Dict:
        """Check historical patterns for similar findings"""
        similar_findings = [
            f for f in self.investigation_history[-50:]
            if f.get('finding', {}).get('type') == finding.get('type')
        ]
        
        return {
            'similar_findings_count': len(similar_findings),
            'avg_resolution_time': 'N/A',  # Would calculate from historical data
            'common_root_causes': list(set(
                f.get('summary', '') for f in similar_findings
            ))[:3]
        }
    
    def _calculate_priority(self, alert: Dict, patterns: Dict) -> str:
        """Calculate investigation priority"""
        if alert.get('risk_score', 0) > 0.9:
            return 'CRITICAL'
        elif alert.get('risk_score', 0) > 0.7:
            return 'HIGH'
        elif patterns.get('total_amount', 0) > 50000:
            return 'HIGH'
        else:
            return 'MEDIUM'
    
    def _generate_investigation_report(self, investigation: Dict, state: Dict) -> Dict:
        """Generate detailed investigation report"""
        return {
            'report_id': investigation['investigation_id'],
            'generated_at': datetime.now().isoformat(),
            'investigation_details': investigation,
            'recommendations': self._generate_recommendations(investigation),
            'required_actions': self._determine_required_actions(investigation),
            'escalation_required': investigation.get('priority') in ['CRITICAL', 'HIGH']
        }
    
    def _generate_recommendations(self, investigation: Dict) -> List[str]:
        """Generate recommendations based on investigation"""
        recommendations = []
        
        if investigation.get('priority') == 'CRITICAL':
            recommendations.append("Immediate account freeze recommended")
            recommendations.append("Notify compliance officer immediately")
        
        if investigation.get('patterns', {}).get('total_amount', 0) > 100000:
            recommendations.append("File SAR within 24 hours")
        
        recommendations.append("Enhanced monitoring for next 30 days")
        
        return recommendations
    
    def _determine_required_actions(self, investigation: Dict) -> List[str]:
        """Determine required actions from investigation"""
        actions = []
        
        if investigation.get('priority') in ['CRITICAL', 'HIGH']:
            actions.append("Escalate to management")
            actions.append("Schedule immediate review meeting")
        
        actions.append("Document findings in case file")
        actions.append("Update risk profile for affected accounts")
        
        return actions