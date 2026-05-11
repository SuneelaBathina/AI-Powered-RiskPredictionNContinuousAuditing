from flask import jsonify, request
from datetime import datetime
import logging
import random
import time

logger = logging.getLogger(__name__)

def register_audit_routes(app, audit_workflow):
    """Register audit workflow REST API routes (fallback for WebSocket)"""
    
    @app.route('/api/audit/start', methods=['POST'])
    def start_audit_rest():
        """Start audit workflow via REST API (fallback)"""
        try:
            data = request.json
            transactions = data.get('transactions', [])
            
            if not transactions:
                return jsonify({'error': 'No transactions provided'}), 400
            
            workflow_id = f"REST_WF_{int(time.time())}"
            logger.info(f"Starting REST audit workflow: {workflow_id} with {len(transactions)} transactions")
            
            # Simulate audit process
            findings = []
            violations = []
            alerts = []
            
            # Process transactions
            for txn in transactions[:100]:
                amount = float(txn.get('amount', 0))
                if amount > 10000:
                    findings.append({
                        'id': f'FIND_{len(findings) + 1}',
                        'type': 'HIGH_VALUE_TRANSACTION',
                        'severity': 'HIGH',
                        'transaction_id': txn.get('transaction_id'),
                        'amount': amount,
                        'description': f'High-value transaction of ${amount:,.2f} detected',
                        'recommendation': 'Review transaction documentation'
                    })
                    alerts.append({
                        'transaction_id': txn.get('transaction_id'),
                        'amount': amount,
                        'risk_score': min(0.5 + amount/50000, 0.95),
                        'transaction_type': txn.get('transaction_type', 'UNKNOWN'),
                        'location': txn.get('location', 'UNKNOWN')
                    })
                
                if txn.get('is_fraud', False):
                    violations.append({
                        'transaction_id': txn.get('transaction_id'),
                        'type': 'FRAUD_DETECTED',
                        'severity': 'CRITICAL',
                        'description': 'Fraudulent transaction detected',
                        'regulation': 'Anti-Fraud Policy'
                    })
            
            summary = {
                'total_transactions': len(transactions),
                'high_risk_transactions': len(alerts),
                'total_audit_findings': len(findings),
                'compliance_score': 0.85
            }
            
            full_results = {
                'workflow_id': workflow_id,
                'audit_findings': findings,
                'compliance_violations': violations,
                'high_risk_alerts': alerts,
                'summary': summary,
                'reports': {
                    'executive_summary': summary,
                    'recommendations': [
                        'Implement enhanced monitoring for high-value transactions',
                        'Review and update AML procedures'
                    ]
                },
                'compliance_score': 0.85,
                'risk_metrics': {
                    'average_risk_score': 0.32,
                    'high_risk_percentage': (len(alerts) / len(transactions) * 100) if transactions else 0
                }
            }
            
            return jsonify({
                'success': True,
                'workflow_id': workflow_id,
                'summary': summary,
                'full_results': full_results,
                'findings_count': len(findings),
                'violations_count': len(violations),
                'alerts_count': len(alerts)
            })
            
        except Exception as e:
            logger.error(f"Error in REST audit: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/audit/status/<workflow_id>', methods=['GET'])
    def get_audit_status(workflow_id):
        """Get audit workflow status"""
        return jsonify({
            'workflow_id': workflow_id,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        })
    
    logger.info("Audit REST API routes registered")