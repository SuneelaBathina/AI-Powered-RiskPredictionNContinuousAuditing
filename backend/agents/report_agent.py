from typing import Dict, Any, List
import json
from datetime import datetime
from .base_agent import BaseAgent

class ReportAgent(BaseAgent):
    """Agent responsible for generating comprehensive reports"""
    
    def __init__(self, bedrock_client, vector_store):
        super().__init__("ReportAgent", bedrock_client, vector_store)
        self.report_templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict:
        """Initialize report templates"""
        return {
            'executive_summary': {
                'sections': ['overview', 'key_metrics', 'critical_findings', 'recommendations']
            },
            'detailed_audit': {
                'sections': ['methodology', 'findings', 'evidence', 'risk_assessment', 'action_plan']
            },
            'regulatory': {
                'sections': ['compliance_status', 'violations', 'reports_filed', 'corrective_actions']
            },
            'investigation': {
                'sections': ['case_summary', 'evidence', 'analysis', 'conclusion', 'recommendations']
            }
        }
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_activity("Generating comprehensive reports")
        
        reports = {}
        
        # Generate different types of reports
        reports['executive_summary'] = self._generate_executive_summary(state)
        reports['detailed_audit'] = self._generate_detailed_audit_report(state)
        reports['regulatory'] = self._generate_regulatory_report(state)
        reports['investigation_summary'] = self._generate_investigation_summary(state)
        
        # Generate visualizations data
        reports['visualizations'] = self._generate_visualization_data(state)
        
        # Enhance reports with AI insights
        for report_type, report in reports.items():
            if report_type != 'visualizations':
                enhanced = self._enhance_with_ai(report, state)
                reports[report_type] = enhanced
        
        # Update state
        state['reports'] = reports
        state['report_generated_at'] = datetime.now().isoformat()
        
        self.log_activity(f"Generated {len(reports)} reports")
        
        return state
    
    def _generate_executive_summary(self, state: Dict) -> Dict:
        """Generate executive summary report"""
        return {
            'title': 'Executive Summary - Risk and Audit Report',
            'generated_at': datetime.now().isoformat(),
            'period': state.get('audit_period', 'current'),
            'overview': {
                'total_transactions': len(state.get('transactions', [])),
                'high_risk_transactions': len(state.get('high_risk_alerts', [])),
                'audit_findings': len(state.get('audit_findings', [])),
                'compliance_score': state.get('compliance_score', 0)
            },
            'critical_findings': [
                f for f in state.get('audit_findings', [])
                if f.get('severity') == 'HIGH'
            ][:5],
            'key_metrics': state.get('risk_metrics', {}),
            'summary': self._create_summary_text(state)
        }
    
    def _generate_detailed_audit_report(self, state: Dict) -> Dict:
        """Generate detailed audit report"""
        return {
            'title': 'Detailed Audit Report',
            'generated_at': datetime.now().isoformat(),
            'methodology': {
                'audit_procedures': ['Risk-based sampling', 'Transaction testing', 'Compliance review'],
                'sample_size': len(state.get('audit_findings', [])) * 10,
                'coverage': 'Comprehensive'
            },
            'findings': state.get('audit_findings', []),
            'risk_assessment': state.get('risk_metrics', {}),
            'recommendations': state.get('audit_recommendations', ''),
            'appendices': {
                'transaction_sample': state.get('assessed_transactions', [])[:50],
                'compliance_violations': state.get('compliance_violations', [])
            }
        }
    
    def _generate_regulatory_report(self, state: Dict) -> Dict:
        """Generate regulatory compliance report"""
        return {
            'title': 'Regulatory Compliance Report',
            'generated_at': datetime.now().isoformat(),
            'compliance_status': {
                'overall_score': state.get('compliance_score', 0),
                'regulatory_reports_filed': len(state.get('regulatory_reports', [])),
                'pending_reports': len([
                    r for r in state.get('regulatory_reports', [])
                    if r.get('status') == 'PENDING'
                ])
            },
            'violations': state.get('compliance_violations', []),
            'reports_filed': state.get('regulatory_reports', []),
            'corrective_actions': self._generate_corrective_actions(state)
        }
    
    def _generate_investigation_summary(self, state: Dict) -> Dict:
        """Generate investigation summary report"""
        investigations = state.get('investigations', [])
        
        return {
            'title': 'Investigation Summary Report',
            'generated_at': datetime.now().isoformat(),
            'total_investigations': len(investigations),
            'by_priority': {
                'CRITICAL': len([i for i in investigations if i.get('priority') == 'CRITICAL']),
                'HIGH': len([i for i in investigations if i.get('priority') == 'HIGH']),
                'MEDIUM': len([i for i in investigations if i.get('priority') == 'MEDIUM'])
            },
            'detailed_reports': state.get('investigation_reports', []),
            'escalations': [
                i for i in investigations
                if i.get('priority') in ['CRITICAL', 'HIGH']
            ]
        }
    
    def _generate_visualization_data(self, state: Dict) -> Dict:
        """Generate data for visualizations"""
        return {
            'risk_timeline': self._create_risk_timeline(state),
            'risk_distribution': self._create_risk_distribution(state),
            'compliance_trend': self._create_compliance_trend(state),
            'audit_findings_by_type': self._group_findings_by_type(state),
            'investigation_outcomes': self._create_investigation_outcomes(state)
        }
    
    def _create_risk_timeline(self, state: Dict) -> List[Dict]:
        """Create timeline data for risk trends"""
        transactions = state.get('assessed_transactions', [])
        timeline = []
        
        # Group by date
        from collections import defaultdict
        daily_risk = defaultdict(lambda: {'total': 0, 'high_risk': 0})
        
        for txn in transactions:
            if 'timestamp' in txn:
                date = txn['timestamp'][:10]  # YYYY-MM-DD
                daily_risk[date]['total'] += 1
                if txn.get('risk_level') == 'HIGH':
                    daily_risk[date]['high_risk'] += 1
        
        for date, counts in sorted(daily_risk.items()):
            timeline.append({
                'date': date,
                'total_transactions': counts['total'],
                'high_risk': counts['high_risk'],
                'risk_percentage': (counts['high_risk'] / counts['total'] * 100) if counts['total'] > 0 else 0
            })
        
        return timeline
    
    def _create_risk_distribution(self, state: Dict) -> Dict:
        """Create risk distribution data"""
        transactions = state.get('assessed_transactions', [])
        
        distribution = {
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0
        }
        
        for txn in transactions:
            level = txn.get('risk_level', 'LOW')
            distribution[level] += 1
        
        return distribution
    
    def _create_compliance_trend(self, state: Dict) -> List[Dict]:
        """Create compliance trend data"""
        # Simplified - would normally track over time
        return [
            {'month': 'Jan', 'score': 0.95},
            {'month': 'Feb', 'score': 0.92},
            {'month': 'Mar', 'score': 0.88},
            {'month': 'Apr', 'score': state.get('compliance_score', 0.9)}
        ]
    
    def _group_findings_by_type(self, state: Dict) -> Dict:
        """Group audit findings by type"""
        findings = state.get('audit_findings', [])
        grouped = {}
        
        for finding in findings:
            ftype = finding.get('type', 'OTHER')
            if ftype not in grouped:
                grouped[ftype] = []
            grouped[ftype].append(finding)
        
        return {k: len(v) for k, v in grouped.items()}
    
    def _create_investigation_outcomes(self, state: Dict) -> Dict:
        """Create investigation outcomes data"""
        investigations = state.get('investigations', [])
        
        outcomes = {
            'RESOLVED': 0,
            'IN_PROGRESS': 0,
            'ESCALATED': 0,
            'PENDING': 0
        }
        
        for inv in investigations:
            status = inv.get('status', 'PENDING')
            if status in outcomes:
                outcomes[status] += 1
        
        return outcomes
    
    def _create_summary_text(self, state: Dict) -> str:
        """Create summary text for executive report"""
        total_txns = len(state.get('transactions', []))
        high_risk = len(state.get('high_risk_alerts', []))
        findings = len(state.get('audit_findings', []))
        
        return (f"During this audit period, {total_txns} transactions were reviewed. "
                f"{high_risk} high-risk transactions were identified, resulting in "
                f"{findings} audit findings. Overall compliance score is "
                f"{state.get('compliance_score', 0)*100:.1f}%.")
    
    def _generate_corrective_actions(self, state: Dict) -> List[str]:
        """Generate corrective actions from findings"""
        actions = set()
        
        for finding in state.get('audit_findings', []):
            if 'recommendation' in finding:
                actions.add(finding['recommendation'])
        
        return list(actions)[:5]
    
    def _enhance_with_ai(self, report: Dict, state: Dict) -> Dict:
        """Enhance report with AI-generated insights"""
        prompt = f"""
        Enhance this report with professional insights and recommendations:
        
        Report: {json.dumps(report)}
        Context: {json.dumps(state.get('risk_metrics', {}))}
        
        Add:
        1. Key insights and trends
        2. Risk implications
        3. Strategic recommendations
        """
        
        ai_insights = self.invoke_llm(prompt)
        
        report['ai_insights'] = ai_insights
        report['enhanced_at'] = datetime.now().isoformat()
        
        return report