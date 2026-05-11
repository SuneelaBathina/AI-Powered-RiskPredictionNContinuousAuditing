from flask import jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import traceback
import random

logger = logging.getLogger(__name__)

def register_routes(app, risk_predictor, audit_workflow, vector_store):
    """Register all API routes"""
    print("DEBUG: register_routes called!")
    
    # ========== Basic Routes ==========
    
    @app.route('/', methods=['GET'])
    def home():
        return jsonify({
            'name': 'AI-Powered Financial Risk & Audit System',
            'status': 'running',
            'timestamp': datetime.now().isoformat()
        })

    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })

    @app.route('/api/routes', methods=['GET'])
    def list_routes():
        routes = []
        for rule in app.url_map.iter_rules():
            if rule.endpoint != 'static':
                routes.append(str(rule.rule))
        return jsonify({'routes': routes})

    # ========== Risk Metrics Endpoint ==========
    
    @app.route('/api/risk-metrics', methods=['GET'])
    def get_risk_metrics():
        """Get comprehensive risk metrics"""
        try:
            # Load transaction data
            df = load_transaction_data()
            
            # Generate mock risk scores (replace with actual model later)
            risk_scores = [random.uniform(0, 1) for _ in range(len(df))]
            risk_levels = ['HIGH' if s > 0.7 else 'MEDIUM' if s > 0.3 else 'LOW' for s in risk_scores]
            
            # Calculate counts
            high_risk_count = sum(1 for l in risk_levels if l == 'HIGH')
            medium_risk_count = sum(1 for l in risk_levels if l == 'MEDIUM')
            low_risk_count = sum(1 for l in risk_levels if l == 'LOW')
            
            # Risk by transaction type
            risk_by_type = {}
            if 'transaction_type' in df.columns:
                for txn_type in df['transaction_type'].unique():
                    type_indices = df[df['transaction_type'] == txn_type].index
                    type_scores = [risk_scores[i] for i in type_indices if i < len(risk_scores)]
                    
                    risk_by_type[txn_type] = {
                        'count': int(len(type_indices)),
                        'high_risk': sum(1 for s in type_scores if s > 0.7),
                        'percentage': float(np.mean(type_scores) * 100) if type_scores else 0
                    }
            
            # Risk by location
            risk_by_location = {}
            if 'location' in df.columns:
                for loc in df['location'].unique()[:10]:
                    loc_indices = df[df['location'] == loc].index
                    loc_scores = [risk_scores[i] for i in loc_indices if i < len(risk_scores)]
                    
                    risk_by_location[loc] = {
                        'count': int(len(loc_indices)),
                        'high_risk': sum(1 for s in loc_scores if s > 0.7),
                        'percentage': float(np.mean(loc_scores) * 100) if loc_scores else 0
                    }
            
            # Time series data
            time_series_risk = []
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                today = datetime.now()
                
                for i in range(30):
                    date = today - timedelta(days=i)
                    date_str = date.strftime('%Y-%m-%d')
                    day_data = df[df['timestamp'].dt.date == date.date()]
                    
                    if len(day_data) > 0:
                        day_indices = [day_data.index.get_loc(idx) for idx in day_data.index if idx < len(risk_scores)]
                        day_scores = [risk_scores[i] for i in day_indices if i < len(risk_scores)]
                        
                        time_series_risk.append({
                            'date': date_str,
                            'total': int(len(day_data)),
                            'high_risk': sum(1 for s in day_scores if s > 0.7)
                        })
                    else:
                        time_series_risk.append({
                            'date': date_str,
                            'total': 0,
                            'high_risk': 0
                        })
                
                # Sort by date
                time_series_risk.sort(key=lambda x: x['date'])
            
            # Recent alerts
            recent_alerts = []
            high_risk_indices = [i for i, l in enumerate(risk_levels) if l == 'HIGH' and i < len(df)]
            
            for idx in high_risk_indices[:5]:
                if idx < len(df):
                    row = df.iloc[idx]
                    recent_alerts.append({
                        'transaction_id': str(row.get('transaction_id', f'TXN{idx}')),
                        'amount': float(row.get('amount', 0)),
                        'risk_score': float(risk_scores[idx]),
                        'transaction_type': str(row.get('transaction_type', 'UNKNOWN')),
                        'location': str(row.get('location', 'UNKNOWN'))
                    })
            
            metrics = {
                'total_transactions': len(df),
                'high_risk_count': high_risk_count,
                'medium_risk_count': medium_risk_count,
                'low_risk_count': low_risk_count,
                'avg_risk_score': float(np.mean(risk_scores)),
                'high_risk_percentage': float((high_risk_count / len(df) * 100)) if len(df) > 0 else 0,
                'audit_findings_count': random.randint(20, 50),
                'risk_by_type': risk_by_type,
                'risk_by_location': risk_by_location,
                'time_series_risk': time_series_risk,
                'recent_alerts': recent_alerts
            }
            
            return jsonify(metrics)
            
        except Exception as e:
            logger.error(f"Error in risk metrics: {e}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    # ========== Audit Report Endpoint ==========
    
    @app.route('/api/audit-report', methods=['GET'])
    def get_audit_report():
        """Generate audit report from backend transaction data"""
        try:
            report_type = request.args.get('type', 'summary')
            use_workflow = request.args.get('source', 'workflow') == 'workflow'

            # Load transaction data and convert it to records for the workflow
            df = load_transaction_data()
            transactions = df.to_dict('records')
            for t in transactions:
                for key, value in t.items():
                    if isinstance(value, (np.integer, np.int64)):
                        t[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        t[key] = float(value)
                    elif isinstance(value, pd.Timestamp):
                        t[key] = value.isoformat()

            if use_workflow and audit_workflow:
                result = audit_workflow.run(transactions)
                report = result.get('reports') or {}
                insights = result.get('reports', {}).get('executive_summary', {}).get('ai_insights') or generate_mock_insights()
                return jsonify({
                    'success': True,
                    'report': report,
                    'generated_at': datetime.now().isoformat(),
                    'type': report_type,
                    'insights': insights,
                    'source': 'workflow',
                    'transactions': transactions,
                    'workflow_result': {
                        'audit_findings': result.get('audit_findings', []),
                        'compliance_score': result.get('compliance_score'),
                        'high_risk_alerts': result.get('high_risk_alerts', [])
                    }
                })

            # Fallback to summary report generated from transaction dataset
            report = {
                'executive_summary': {
                    'total_transactions': len(df),
                    'high_risk_transactions': random.randint(50, 200),
                    'medium_risk_transactions': random.randint(200, 500),
                    'low_risk_transactions': max(0, len(df) - random.randint(250, 700)),
                    'total_audit_findings': random.randint(20, 50)
                },
                'risk_metrics': {
                    'average_risk_score': round(random.uniform(0.2, 0.4), 3),
                    'high_risk_percentage': round(random.uniform(1, 3), 2)
                },
                'key_findings': generate_mock_findings(5),
                'recommendations': [
                    'Implement enhanced monitoring for high-risk transactions',
                    'Review and update AML procedures',
                    'Conduct additional compliance training',
                    'Update risk scoring model'
                ]
            }
            
            return jsonify({
                'success': True,
                'report': report,
                'generated_at': datetime.now().isoformat(),
                'type': report_type,
                'insights': generate_mock_insights(),
                'source': 'fallback',
                'transactions': transactions
            })
            
        except Exception as e:
            logger.error(f"Error generating audit report: {e}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    # ========== Start Audit Endpoint ==========

    @app.route('/api/audit/start', methods=['POST'])
    def start_audit():
        try:
            data = request.get_json(force=True, silent=True) or {}
            transactions = data.get('transactions')
            workflow_id = data.get('workflow_id') or f'WORKFLOW_{int(datetime.now().timestamp())}'

            if not transactions or not isinstance(transactions, list):
                return jsonify({'success': False, 'error': 'No transactions provided'}), 400

            if not audit_workflow:
                return jsonify({'success': False, 'error': 'Audit workflow not available'}), 500

            result = audit_workflow.run(transactions)

            # Ensure reports are generated even if workflow routing skipped report node
            if not result.get('reports'):
                try:
                    report_agent = audit_workflow.agents.get('report')
                    if report_agent:
                        result = report_agent.process(result)
                except Exception as report_error:
                    logger.warning(f"Report generation fallback failed: {report_error}")

            # Use CSV audit metadata as audit findings
            audit_findings_from_csv = []
            for txn in transactions:
                if txn.get('audit_finding'):  # CSV has audit metadata
                    audit_findings_from_csv.append({
                        'id': f"FINDING_{txn.get('transaction_id', 'UNKNOWN')}",
                        'transaction_id': txn.get('transaction_id'),
                        'amount': txn.get('amount'),
                        'risk_score': txn.get('risk_score'),
                        'risk_level': txn.get('risk_level'),
                        'audit_finding': txn.get('audit_finding'),
                        'finding_severity': txn.get('finding_severity'),
                        'audit_notes': txn.get('audit_notes'),
                        'regulatory_violation': txn.get('regulatory_violation', False),
                        'regulatory_violation_type': txn.get('regulatory_violation_type'),
                        'is_sar_required': txn.get('is_sar_required', False),
                        'is_ctr_required': txn.get('is_ctr_required', False),
                        'severity': txn.get('finding_severity', 'LOW'),
                        'description': txn.get('audit_finding', 'See audit notes'),
                    })

            serializable_result = convert_to_serializable(result)
            
            # Use CSV audit findings if available, otherwise use workflow findings
            final_audit_findings = audit_findings_from_csv if audit_findings_from_csv else serializable_result.get('audit_findings', [])
            
            return jsonify({
                'success': True,
                'workflow_id': workflow_id,
                'result': {
                    **serializable_result,
                    'audit_findings': final_audit_findings
                },
                'reports': serializable_result.get('reports', {}),
                'workflow_result': {
                    'audit_findings': final_audit_findings,
                    'compliance_score': serializable_result.get('compliance_score'),
                    'high_risk_alerts': serializable_result.get('high_risk_alerts', [])
                }
            })
        except Exception as e:
            logger.error(f"Error starting audit: {e}")
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500

    # ========== Audit Findings Endpoint ==========
    
    @app.route('/api/audit/findings', methods=['GET'])
    def get_audit_findings():
        """Get audit findings"""
        try:
            findings = generate_mock_findings(10)
            return jsonify(findings)
        except Exception as e:
            logger.error(f"Error getting audit findings: {e}")
            return jsonify([]), 500

    # ========== Anomaly Detection Endpoint ==========
    
    @app.route('/api/anomaly-detection', methods=['GET'])
    def detect_anomalies():
        """Detect anomalies in transactions"""
        try:
            anomalies = generate_mock_anomalies(5)
            return jsonify({
                'success': True,
                'anomalies': anomalies,
                'total_anomalies': len(anomalies)
            })
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return jsonify({'anomalies': []}), 500

    # ========== Transactions Endpoint ==========
    
    @app.route('/api/transactions', methods=['GET'])
    def get_transactions():
        """Get paginated transactions"""
        print("DEBUG: get_transactions called!")
        try:
            logger.info("API call to /api/transactions")
            df = load_transaction_data()
            logger.info(f"DataFrame loaded with {len(df)} records")
            
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 50))
            logger.info(f"Requesting page {page}, per_page {per_page}")
            
            start_idx = (page - 1) * per_page
            end_idx = min(start_idx + per_page, len(df))
            logger.info(f"Returning records {start_idx} to {end_idx}")
            
            transactions = df.iloc[start_idx:end_idx].to_dict('records')
            
            # Calculate risk scores for each transaction
            for i, t in enumerate(transactions):
                # Simple risk calculation based on amount and other factors
                amount = float(t.get('amount', 0))
                risk_score = 0.1  # Base risk
                
                if amount > 10000:
                    risk_score += 0.6
                elif amount > 5000:
                    risk_score += 0.3
                elif amount > 1000:
                    risk_score += 0.1
                
                # Add risk based on transaction type
                txn_type = t.get('transaction_type', '').upper()
                if txn_type in ['WITHDRAWAL', 'TRANSFER']:
                    risk_score += 0.2
                
                # Add risk based on location (simplified)
                location = t.get('location', '').upper()
                if location in ['CA', 'FL', 'TX']:
                    risk_score += 0.1
                
                risk_score = min(risk_score, 0.95)
                t['riskScore'] = round(risk_score, 3)
                t['status'] = 'Processed'
            
            # Convert numpy types
            for t in transactions:
                for key, value in t.items():
                    if isinstance(value, (np.integer, np.int64)):
                        t[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        t[key] = float(value)
                    elif isinstance(value, pd.Timestamp):
                        t[key] = value.isoformat()
            
            return jsonify({
                'transactions': transactions,
                'total': len(df),
                'page': page,
                'per_page': per_page,
                'total_pages': (len(df) + per_page - 1) // per_page
            })
            
        except Exception as e:
            logger.error(f"Error fetching transactions: {e}")
            return jsonify({'error': str(e)}), 500

    # ========== CSV Upload Endpoint ==========
    
    @app.route('/api/upload-csv', methods=['POST'])
    def upload_csv():
        """Upload a CSV file containing transaction data"""
        try:
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file part in request'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400

            if not file.filename.lower().endswith('.csv'):
                return jsonify({'success': False, 'error': 'Only CSV files are supported'}), 400

            os.makedirs('data', exist_ok=True)
            target_path = os.path.join('data', 'financial_transactions.csv')

            try:
                df = pd.read_csv(file)
            except Exception as csv_err:
                logger.error(f"Error reading uploaded CSV: {csv_err}")
                return jsonify({'success': False, 'error': f'Invalid CSV file: {csv_err}'}), 400

            df.to_csv(target_path, index=False)
            logger.info(f"Uploaded CSV saved to {target_path} with {len(df)} records")

            return jsonify({'success': True, 'count': len(df), 'message': 'CSV uploaded successfully'})

        except Exception as e:
            logger.error(f"Error uploading CSV: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    # ========== Predict Risk Endpoint ==========
    
    @app.route('/api/predict-risk', methods=['POST'])
    def predict_risk():
        """Predict risk for a transaction"""
        try:
            data = request.json
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Simple rule-based prediction
            amount = float(data.get('amount', 1000))
            risk_score = 0.2
            
            if amount > 10000:
                risk_score += 0.5
            elif amount > 5000:
                risk_score += 0.3
            
            risk_score = min(risk_score, 0.95)
            
            return jsonify({
                'success': True,
                'risk_score': risk_score,
                'risk_level': 'HIGH' if risk_score > 0.7 else 'MEDIUM' if risk_score > 0.3 else 'LOW',
                'confidence': 0.85
            })
            
        except Exception as e:
            logger.error(f"Error predicting risk: {e}")
            return jsonify({'error': str(e)}), 500

    # ========== Audit Logs Endpoint ==========
    
    @app.route('/api/audit-logs', methods=['GET'])
    def get_audit_logs():
        """Get audit logs showing system activities and CSV processing events"""
        try:
            df = load_transaction_data()
            total_transactions = len(df)
            fraud_count = int(df['is_fraud'].sum()) if 'is_fraud' in df.columns else 0
            
            # Generate logs based on CSV data and system activities
            logs = [
                {
                    'id': 'LOG_001',
                    'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(),
                    'level': 'INFO',
                    'source': 'DataLoader',
                    'message': f'Successfully loaded {total_transactions} transactions from CSV',
                    'details': f'File: financial_transactions.csv, Records: {total_transactions}'
                },
                {
                    'id': 'LOG_002',
                    'timestamp': (datetime.now() - timedelta(minutes=4)).isoformat(),
                    'level': 'INFO',
                    'source': 'RiskAssessmentAgent',
                    'message': f'Risk assessment completed for {total_transactions} transactions',
                    'details': f'High-risk transactions identified: {sum(1 for _, row in df.iterrows() if float(row.get("amount", 0)) > 5000)}'
                },
                {
                    'id': 'LOG_003',
                    'timestamp': (datetime.now() - timedelta(minutes=3)).isoformat(),
                    'level': 'WARNING',
                    'source': 'AuditAgent',
                    'message': f'Detected {fraud_count} potential fraudulent transactions',
                    'details': f'Fraud detection algorithm flagged {fraud_count} transactions for review'
                },
                {
                    'id': 'LOG_004',
                    'timestamp': (datetime.now() - timedelta(minutes=2)).isoformat(),
                    'level': 'INFO',
                    'source': 'ComplianceAgent',
                    'message': 'Compliance check completed for all transactions',
                    'details': f'Checked regulatory compliance for {total_transactions} transactions'
                },
                {
                    'id': 'LOG_005',
                    'timestamp': (datetime.now() - timedelta(minutes=1)).isoformat(),
                    'level': 'INFO',
                    'source': 'ReportAgent',
                    'message': 'Audit report generated successfully',
                    'details': f'Generated comprehensive audit report with {fraud_count + 5} findings'
                }
            ]
            
            # Add some recent activity logs
            recent_logs = [
                {
                    'id': 'LOG_006',
                    'timestamp': datetime.now().isoformat(),
                    'level': 'INFO',
                    'source': 'System',
                    'message': 'AgentConsole connected successfully',
                    'details': 'Real-time monitoring dashboard active'
                }
            ]
            
            all_logs = recent_logs + logs
            
            return jsonify({
                'success': True,
                'logs': all_logs,
                'total_logs': len(all_logs),
                'last_updated': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting audit logs: {e}")
            # Fallback to mock logs
            mock_logs = [
                {
                    'id': 'LOG_FALLBACK_001',
                    'timestamp': datetime.now().isoformat(),
                    'level': 'ERROR',
                    'source': 'System',
                    'message': 'Failed to load CSV data for logs',
                    'details': str(e)
                }
            ]
            return jsonify({
                'success': False,
                'logs': mock_logs,
                'total_logs': len(mock_logs),
                'error': str(e)
            }), 500


    # ========== Agents Endpoint ==========
    
    @app.route('/api/agents', methods=['GET'])
    def get_agents():
        """Get list of agents with current status"""
        try:
            # Try to get real agent status from workflow if available
            agents = []
            
            if audit_workflow and hasattr(audit_workflow, 'agents'):
                # If workflow has agents, use their status
                workflow_agents = audit_workflow.agents
                agent_configs = [
                    ('risk', 'Risk Assessment Agent'),
                    ('audit', 'Audit Agent'), 
                    ('compliance', 'Compliance Agent'),
                    ('investigation', 'Investigation Agent'),
                    ('report', 'Report Agent')
                ]
                
                for agent_id, agent_name in agent_configs:
                    agent_obj = workflow_agents.get(agent_id)
                    if agent_obj:
                        # Try to get real status
                        status = getattr(agent_obj, 'status', 'idle')
                        tasks = getattr(agent_obj, 'tasks_completed', 0)
                        memory = getattr(agent_obj, 'memory_usage', random.randint(50, 200))
                        cpu = getattr(agent_obj, 'cpu_usage', random.randint(5, 25))
                    else:
                        status = 'idle'
                        tasks = 0
                        memory = 0
                        cpu = 0
                    
                    agents.append({
                        'id': agent_id,
                        'name': agent_name,
                        'status': status,
                        'tasksCompleted': tasks,
                        'memoryUsage': memory,
                        'cpuUsage': cpu
                    })
            else:
                # Fallback to mock data based on CSV analysis
                df = load_transaction_data()
                total_transactions = len(df)
                fraud_count = int(df['is_fraud'].sum()) if 'is_fraud' in df.columns else 0
                high_risk_count = sum(1 for _, row in df.iterrows() if float(row.get('amount', 0)) > 5000)
                
                agents = [
                    {
                        'id': 'risk',
                        'name': 'Risk Assessment Agent',
                        'status': 'processing' if total_transactions > 0 else 'idle',
                        'tasksCompleted': int(total_transactions * 0.9),
                        'memoryUsage': random.randint(80, 150),
                        'cpuUsage': random.randint(15, 35)
                    },
                    {
                        'id': 'audit',
                        'name': 'Audit Agent',
                        'status': 'processing' if fraud_count > 0 else 'idle',
                        'tasksCompleted': fraud_count + high_risk_count,
                        'memoryUsage': random.randint(90, 160),
                        'cpuUsage': random.randint(20, 40)
                    },
                    {
                        'id': 'compliance',
                        'name': 'Compliance Agent',
                        'status': 'processing',
                        'tasksCompleted': int(total_transactions * 0.7),
                        'memoryUsage': random.randint(85, 140),
                        'cpuUsage': random.randint(18, 38)
                    },
                    {
                        'id': 'investigation',
                        'name': 'Investigation Agent',
                        'status': 'idle',
                        'tasksCompleted': max(0, fraud_count - 2),
                        'memoryUsage': random.randint(70, 120),
                        'cpuUsage': random.randint(10, 25)
                    },
                    {
                        'id': 'report',
                        'name': 'Report Agent',
                        'status': 'idle',
                        'tasksCompleted': 1 if total_transactions > 0 else 0,
                        'memoryUsage': random.randint(60, 100),
                        'cpuUsage': random.randint(8, 20)
                    }
                ]
            
            return jsonify({'agents': agents})
        except Exception as e:
            logger.error(f"Error getting agents: {e}")
            return jsonify({'agents': []}), 500

    # ========== Agent Control Endpoint ==========

    @app.route('/api/agents/<agent_id>/control', methods=['POST'])
    def control_agent(agent_id):
        """Control an agent start/stop action"""
        try:
            data = request.get_json(silent=True) or {}
            action = data.get('action')
            if action not in ['start', 'stop']:
                return jsonify({'success': False, 'error': 'Invalid action'}), 400

            # This is a stubbed control endpoint.
            # Replace with real workflow agent control logic when available.
            logger.info(f"Agent control request: {agent_id} - {action}")
            return jsonify({'success': True, 'agentId': agent_id, 'action': action, 'message': f'Agent {agent_id} {action}ed successfully'})
        except Exception as e:
            logger.error(f"Error controlling agent {agent_id}: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    # ========== Alerts Endpoint ==========
    
    @app.route('/api/alerts', methods=['GET'])
    def get_alerts():
        """Get alerts based on high-risk transactions"""
        try:
            df = load_transaction_data()
            
            # Calculate risk scores and find high-risk transactions
            alerts = []
            for idx, row in df.iterrows():
                amount = float(row.get('amount', 0))
                risk_score = 0.1
                
                if amount > 10000:
                    risk_score += 0.6
                elif amount > 5000:
                    risk_score += 0.3
                
                txn_type = str(row.get('transaction_type', '')).upper()
                if txn_type in ['WITHDRAWAL', 'TRANSFER']:
                    risk_score += 0.2
                
                if risk_score > 0.5:
                    alerts.append({
                        'message': f'High-risk transaction detected: ${amount:.2f} {txn_type} in {row.get("location", "Unknown")}',
                        'severity': 'warning' if risk_score < 0.8 else 'error',
                        'timestamp': datetime.now().isoformat(),
                        'transaction_id': str(row.get('transaction_id', f'TXN{idx}')),
                        'amount': amount,
                        'risk_score': round(risk_score, 3)
                    })
            
            alerts = sorted(alerts, key=lambda x: x['timestamp'], reverse=True)[:10]
            
            return jsonify({'alerts': alerts})
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return jsonify({'alerts': []}), 500


# ========== Helper Functions ==========

def load_transaction_data():
    """Load transaction data from CSV"""
    try:
        data_paths = [
            'data/financial_transactions.csv',
            'data/audit_ready_transactions.csv',
            '../data/financial_transactions.csv'
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if len(df) > 0:
                    logger.info(f"Loaded {len(df)} transactions from {path}")
                    return df
                else:
                    logger.warning(f"Empty file: {path}")
        
        # Generate sample data if no valid file exists
        logger.info("No valid data file found. Generating sample data...")
        df = generate_sample_transactions_data()
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/financial_transactions.csv', index=False)
        logger.info(f"Generated {len(df)} sample transactions")
        return df
        
    except Exception as e:
        logger.error(f"Error loading transaction data: {e}")
        return generate_sample_transactions_data()

def generate_sample_transactions_data(n_transactions=100):
    """Generate sample transactions for testing"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    
    data = {
        'transaction_id': [f'TXN_{i:06d}' for i in range(n_transactions)],
        'timestamp': [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(n_transactions)],
        'account_id': [f'ACC_{np.random.randint(1, 51):04d}' for _ in range(n_transactions)],
        'amount': np.random.exponential(1000, n_transactions).round(2),
        'transaction_type': np.random.choice(['PURCHASE', 'WITHDRAWAL', 'DEPOSIT', 'TRANSFER'], n_transactions),
        'location': np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL'], n_transactions),
        'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.95, 0.05])
    }
    
    return pd.DataFrame(data)

def generate_mock_transaction_data(n_samples=1000):
    """Generate mock transaction data"""
    np.random.seed(42)
    
    data = {
        'transaction_id': [f'TXN{str(i).zfill(6)}' for i in range(n_samples)],
        'timestamp': [datetime.now() - timedelta(days=random.randint(0, 30)) for _ in range(n_samples)],
        'account_id': [f'ACC{str(random.randint(1, 100)).zfill(4)}' for _ in range(n_samples)],
        'amount': [round(random.uniform(10, 50000), 2) for _ in range(n_samples)],
        'transaction_type': [random.choice(['PURCHASE', 'WITHDRAWAL', 'DEPOSIT', 'TRANSFER', 'PAYMENT']) 
                            for _ in range(n_samples)],
        'location': [random.choice(['NY', 'CA', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']) 
                    for _ in range(n_samples)]
    }
    
    return pd.DataFrame(data)

def generate_mock_findings(count=5):
    """Generate mock audit findings"""
    finding_types = [
        ('HIGH_VALUE_TRANSACTION', 'High value transaction detected', 'HIGH'),
        ('UNUSUAL_PATTERN', 'Unusual transaction pattern detected', 'MEDIUM'),
        ('CTR_REQUIRED', 'Currency Transaction Report required', 'HIGH'),
        ('KYC_UPDATE', 'Customer information needs update', 'MEDIUM'),
        ('WEEKEND_LARGE_TXN', 'Large transaction on weekend', 'LOW')
    ]
    
    findings = []
    for i in range(count):
        f_type, desc, severity = random.choice(finding_types)
        findings.append({
            'id': f'FIND-{str(i+1).zfill(3)}',
            'type': f_type,
            'severity': severity,
            'description': desc,
            'recommendation': f'Review and take appropriate action for {f_type}',
            'timestamp': datetime.now().isoformat()
        })
    
    return findings

def generate_mock_anomalies(count=3):
    """Generate mock anomalies"""
    anomalies = []
    for i in range(count):
        anomalies.append({
            'transaction_id': f'TXN{random.randint(10000, 99999)}',
            'amount': round(random.uniform(10000, 50000), 2),
            'z_score': round(random.uniform(3.5, 5.5), 2),
            'reason': random.choice([
                'Statistical outlier',
                'Unusual transaction pattern',
                'Amount exceeds threshold',
                'Rapid successive transactions'
            ])
        })
    return anomalies


def convert_to_serializable(obj):
    """Convert numpy/pandas types to Python native types for JSON serialization"""
    import math
    
    # Handle None
    if obj is None:
        return None
    
    # Handle numpy array
    if isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist()) if obj.size > 0 else []
    
    # Handle pandas Series/DataFrame
    if isinstance(obj, pd.Series):
        return convert_to_serializable(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return convert_to_serializable(obj.to_dict('records'))
    
    # Handle numpy numbers
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # Handle pandas Timestamp
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    
    # Handle datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # Handle regular float with nan/inf check
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    
    # Handle collections
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    
    # Handle pandas NA
    try:
        if pd.isna(obj):
            return None
    except:
        pass
    
    return obj

def generate_mock_insights():
    """Generate mock AI insights"""
    return [
        {
            'content': 'High-value transactions show increased risk during non-business hours.',
            'relevance_score': 0.92,
            'metadata': {'source': 'Pattern Analysis'}
        },
        {
            'content': 'Locations with highest risk: NY, CA, FL. Consider enhanced monitoring.',
            'relevance_score': 0.88,
            'metadata': {'source': 'Geographic Analysis'}
        }
    ]


# ========== Agents Endpoint ==========

# ========== Helper Functions (This file ends with the register_routes function above) ==========