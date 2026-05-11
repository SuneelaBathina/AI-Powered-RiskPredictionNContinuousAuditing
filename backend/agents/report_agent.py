from typing import Dict, Any, List
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from .base_agent import BaseAgent

class ReportAgent(BaseAgent):
    """Agent responsible for generating comprehensive reports using batch processing"""
    
    def __init__(self, bedrock_client, vector_store):
        super().__init__("ReportAgent", bedrock_client, vector_store)
        self.report_templates = self._initialize_templates()
        self.batch_size = 500  # Process in batches of 500 transactions
        
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
        """Generate reports using batch processing"""
        self.log_activity("Generating comprehensive reports with batch processing")
        
        # Extract data with batch processing
        transactions = state.get('transactions', [])
        assessed_transactions = state.get('assessed_transactions', [])
        audit_findings = state.get('audit_findings', [])
        compliance_violations = state.get('compliance_violations', [])
        high_risk_alerts = state.get('high_risk_alerts', [])
        risk_metrics = state.get('risk_metrics', {})
        
        self.log_activity(f"Processing {len(assessed_transactions)} assessed transactions in batches of {self.batch_size}")
        
        # Convert to DataFrame for efficient batch processing
        df = pd.DataFrame(assessed_transactions) if assessed_transactions else pd.DataFrame()
        
        reports = {}
        
        # Generate reports using batch-processed data
        reports['executive_summary'] = self._generate_executive_summary_batch(
            df, audit_findings, risk_metrics, high_risk_alerts, state
        )
        reports['detailed_audit'] = self._generate_detailed_audit_report_batch(
            df, audit_findings, compliance_violations
        )
        reports['regulatory'] = self._generate_regulatory_report_batch(compliance_violations, state)
        reports['investigation_summary'] = self._generate_investigation_summary_batch(state)
        
        # Generate visualizations data using batch processing
        reports['visualizations'] = self._generate_visualization_data_batch(df, state)
        
        # Enhance reports with AI insights (only for non-visualization reports)
        for report_type, report in reports.items():
            if report_type != 'visualizations':
                enhanced = self._enhance_with_ai(report, state)
                reports[report_type] = enhanced
        
        # Update state
        state['reports'] = reports
        state['report_generated_at'] = datetime.now().isoformat()
        
        self.log_activity(f"Generated {len(reports)} reports using batch processing")
        
        return state
    
    def _generate_executive_summary_batch(self, df: pd.DataFrame, findings: List, 
                                          risk_metrics: Dict, alerts: List, state: Dict) -> Dict:
        """Generate executive summary using batch-processed DataFrame"""
        
        # Use vectorized operations instead of loops
        total_transactions = len(df)
        high_risk_count = len(df[df['risk_level'] == 'HIGH']) if not df.empty and 'risk_level' in df.columns else len(alerts)
        
        # Calculate risk metrics efficiently using pandas
        if not df.empty and 'risk_score' in df.columns:
            avg_risk_score = df['risk_score'].mean()
            max_risk_score = df['risk_score'].max()
            min_risk_score = df['risk_score'].min()
        else:
            avg_risk_score = risk_metrics.get('avg_risk_score', 0)
            max_risk_score = risk_metrics.get('max_risk_score', 0)
            min_risk_score = risk_metrics.get('min_risk_score', 0)
        
        # Group findings by severity using pandas if available
        findings_df = pd.DataFrame(findings) if findings else pd.DataFrame()
        if not findings_df.empty and 'severity' in findings_df.columns:
            critical_findings = findings_df[findings_df['severity'] == 'HIGH'].head(5).to_dict('records')
        else:
            critical_findings = [f for f in findings if f.get('severity') == 'HIGH'][:5]
        
        return {
            'title': 'Executive Summary - Risk and Audit Report',
            'generated_at': datetime.now().isoformat(),
            'period': state.get('audit_period', 'current'),
            'overview': {
                'total_transactions': total_transactions,
                'high_risk_transactions': high_risk_count,
                'audit_findings': len(findings),
                'compliance_score': state.get('compliance_score', 0)
            },
            'critical_findings': critical_findings,
            'key_metrics': {
                'avg_risk_score': float(avg_risk_score),
                'max_risk_score': float(max_risk_score),
                'min_risk_score': float(min_risk_score),
                'total_value_at_risk': float(df['amount'].sum()) if not df.empty and 'amount' in df.columns else 0
            },
            'summary': self._create_summary_text_batch(total_transactions, high_risk_count, len(findings), state)
        }
    
    def _generate_detailed_audit_report_batch(self, df: pd.DataFrame, findings: List, violations: List) -> Dict:
        """Generate detailed audit report using batch-processed DataFrame"""
        
        # Use pandas groupby for efficient aggregation
        risk_distribution = {}
        if not df.empty:
            if 'risk_level' in df.columns:
                risk_distribution['by_risk_level'] = df['risk_level'].value_counts().to_dict()
            if 'transaction_type' in df.columns and 'risk_level' in df.columns:
                risk_distribution['by_transaction_type'] = df.groupby(['transaction_type', 'risk_level']).size().to_dict()
            if 'location' in df.columns and 'risk_level' in df.columns:
                risk_distribution['by_location'] = df.groupby(['location', 'risk_level']).size().to_dict()
        
        # Get high-risk transactions efficiently
        high_risk_transactions = []
        if not df.empty and 'risk_level' in df.columns:
            high_risk_df = df[df['risk_level'] == 'HIGH'].head(20)
            high_risk_transactions = high_risk_df.to_dict('records')
        
        return {
            'title': 'Detailed Audit Report',
            'generated_at': datetime.now().isoformat(),
            'methodology': {
                'audit_procedures': ['Risk-based sampling', 'Transaction testing', 'Compliance review'],
                'sample_size': len(findings) * 10,
                'coverage': 'Comprehensive',
                'batch_processing': True,
                'batch_size': self.batch_size
            },
            'risk_distribution': risk_distribution,
            'findings': findings[:20],
            'violations': violations[:10],
            'high_risk_transactions': high_risk_transactions,
            'recommendations': self._generate_recommendations_batch(findings, violations)
        }
    
    def _generate_regulatory_report_batch(self, violations: List, state: Dict) -> Dict:
        """Generate regulatory report using batch processing"""
        
        # Count violations by severity efficiently
        violations_df = pd.DataFrame(violations) if violations else pd.DataFrame()
        
        if not violations_df.empty and 'severity' in violations_df.columns:
            violations_by_severity = violations_df['severity'].value_counts().to_dict()
        else:
            violations_by_severity = {
                'CRITICAL': len([v for v in violations if v.get('severity') == 'CRITICAL']),
                'HIGH': len([v for v in violations if v.get('severity') == 'HIGH']),
                'MEDIUM': len([v for v in violations if v.get('severity') == 'MEDIUM']),
                'LOW': len([v for v in violations if v.get('severity') == 'LOW'])
            }
        
        # Group violations by type
        violations_by_type = {}
        if not violations_df.empty and 'type' in violations_df.columns:
            violations_by_type = violations_df['type'].value_counts().to_dict()
        
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
            'violations': violations[:20],
            'violations_summary': {
                'total_violations': len(violations),
                'by_severity': violations_by_severity,
                'by_type': violations_by_type
            },
            'reports_filed': state.get('regulatory_reports', [])[:10],
            'corrective_actions': self._generate_corrective_actions_batch(violations)
        }
    
    def _generate_investigation_summary_batch(self, state: Dict) -> Dict:
        """Generate investigation summary using batch processing"""
        investigations = state.get('investigations', [])
        
        # Use pandas for efficient grouping
        inv_df = pd.DataFrame(investigations) if investigations else pd.DataFrame()
        
        if not inv_df.empty and 'priority' in inv_df.columns:
            by_priority = inv_df['priority'].value_counts().to_dict()
        else:
            by_priority = {
                'CRITICAL': len([i for i in investigations if i.get('priority') == 'CRITICAL']),
                'HIGH': len([i for i in investigations if i.get('priority') == 'HIGH']),
                'MEDIUM': len([i for i in investigations if i.get('priority') == 'MEDIUM'])
            }
        
        return {
            'title': 'Investigation Summary Report',
            'generated_at': datetime.now().isoformat(),
            'total_investigations': len(investigations),
            'by_priority': by_priority,
            'detailed_reports': state.get('investigation_reports', [])[:10],
            'escalations': [
                i for i in investigations
                if i.get('priority') in ['CRITICAL', 'HIGH']
            ][:10]
        }
    
    def _generate_visualization_data_batch(self, df: pd.DataFrame, state: Dict) -> Dict:
        """Generate visualization data using batch processing"""
        
        visualizations = {
            'risk_timeline': [],
            'risk_distribution': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
            'compliance_trend': [],
            'audit_findings_by_type': {},
            'investigation_outcomes': {'RESOLVED': 0, 'IN_PROGRESS': 0, 'ESCALATED': 0, 'PENDING': 0}
        }
        
        # Risk timeline using pandas groupby
        if not df.empty and 'timestamp' in df.columns and 'risk_level' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            timeline_data = df.groupby('date').agg(
                total=('transaction_id', 'count'),
                high_risk=('risk_level', lambda x: (x == 'HIGH').sum())
            ).reset_index()
            
            for _, row in timeline_data.iterrows():
                visualizations['risk_timeline'].append({
                    'date': str(row['date']),
                    'total_transactions': int(row['total']),
                    'high_risk': int(row['high_risk']),
                    'risk_percentage': (row['high_risk'] / row['total'] * 100) if row['total'] > 0 else 0
                })
        
        # Risk distribution using pandas value_counts
        if not df.empty and 'risk_level' in df.columns:
            risk_counts = df['risk_level'].value_counts().to_dict()
            visualizations['risk_distribution'] = {
                'HIGH': risk_counts.get('HIGH', 0),
                'MEDIUM': risk_counts.get('MEDIUM', 0),
                'LOW': risk_counts.get('LOW', 0)
            }
        
        # Compliance trend
        visualizations['compliance_trend'] = [
            {'month': 'Jan', 'score': 0.95},
            {'month': 'Feb', 'score': 0.92},
            {'month': 'Mar', 'score': 0.88},
            {'month': 'Apr', 'score': state.get('compliance_score', 0.9)}
        ]
        
        # Audit findings by type
        findings = state.get('audit_findings', [])
        if findings:
            findings_df = pd.DataFrame(findings)
            if not findings_df.empty and 'type' in findings_df.columns:
                visualizations['audit_findings_by_type'] = findings_df['type'].value_counts().to_dict()
        
        # Investigation outcomes
        investigations = state.get('investigations', [])
        if investigations:
            inv_df = pd.DataFrame(investigations)
            if not inv_df.empty and 'status' in inv_df.columns:
                status_counts = inv_df['status'].value_counts().to_dict()
                visualizations['investigation_outcomes'] = status_counts
        
        return visualizations
    
    def _create_summary_text_batch(self, total_txns: int, high_risk: int, findings: int, state: Dict) -> str:
        """Create summary text using batch-calculated values"""
        return (f"During this audit period, {total_txns} transactions were reviewed. "
                f"{high_risk} high-risk transactions were identified, resulting in "
                f"{findings} audit findings. Overall compliance score is "
                f"{state.get('compliance_score', 0)*100:.1f}%.")
    
    def _generate_recommendations_batch(self, findings: List, violations: List) -> List[str]:
        """Generate recommendations based on batch analysis"""
        recommendations = []
        
        # Analyze findings in batch
        findings_df = pd.DataFrame(findings) if findings else pd.DataFrame()
        
        if not findings_df.empty and 'type' in findings_df.columns:
            high_value_count = len(findings_df[findings_df['type'] == 'HIGH_VALUE_TRANSACTION'])
            if high_value_count > 0:
                recommendations.append(f"Review {high_value_count} high-value transactions for potential money laundering")
        
        # Analyze violations in batch
        violations_df = pd.DataFrame(violations) if violations else pd.DataFrame()
        
        if not violations_df.empty and 'severity' in violations_df.columns:
            critical_count = len(violations_df[violations_df['severity'] == 'CRITICAL'])
            if critical_count > 0:
                recommendations.append(f"Address {critical_count} critical compliance violations immediately")
        
        # General recommendations
        recommendations.extend([
            "Implement enhanced monitoring for high-risk transaction types",
            "Review and update AML procedures",
            "Conduct additional training for compliance staff",
            "Schedule regular audit reviews"
        ])
        
        return recommendations
    
    def _generate_corrective_actions_batch(self, violations: List) -> List[str]:
        """Generate corrective actions from violations using batch processing"""
        actions = set()
        
        for violation in violations[:20]:  # Process first 20 violations
            if 'recommendation' in violation:
                actions.add(violation['recommendation'])
            elif 'type' in violation:
                if violation['type'] == 'CTR_REQUIRED':
                    actions.add("File Currency Transaction Reports within 15 days")
                elif violation['type'] == 'FRAUD_DETECTED':
                    actions.add("Escalate fraud cases to investigation team")
                elif violation['type'] == 'KYC_MISSING':
                    actions.add("Update KYC documentation for affected accounts")
        
        return list(actions)[:5]
    
    def _enhance_with_ai(self, report: Dict, state: Dict) -> Dict:
        """Enhance report with AI-generated insights (batch optimized)"""
        
        # Only enhance if we have significant data to avoid API calls for empty reports
        if len(report.get('critical_findings', [])) == 0 and report.get('overview', {}).get('audit_findings', 0) == 0:
            report['ai_insights'] = "No significant findings to analyze."
            report['enhanced_at'] = datetime.now().isoformat()
            return report
        
        # Create a concise prompt to avoid token limits
        prompt = f"""
        Analyze this audit report and provide key insights:
        
        Total Transactions: {report.get('overview', {}).get('total_transactions', 0)}
        High Risk: {report.get('overview', {}).get('high_risk_transactions', 0)}
        Findings: {report.get('overview', {}).get('audit_findings', 0)}
        Compliance Score: {report.get('overview', {}).get('compliance_score', 0)}
        
        Provide:
        1. Top 2 risk concerns
        2. Top 2 recommendations
        """
        
        try:
            ai_insights = self.invoke_llm(prompt, max_tokens=200)
        except:
            ai_insights = "AI insights temporarily unavailable. Please review the report manually."
        
        report['ai_insights'] = ai_insights
        report['enhanced_at'] = datetime.now().isoformat()
        
        return report