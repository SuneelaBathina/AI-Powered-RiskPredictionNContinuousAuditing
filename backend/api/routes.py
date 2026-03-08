from flask import jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import urllib

logger = logging.getLogger(__name__)

def register_routes(app, risk_predictor, audit_workflow, vector_store):
    """Register all API routes"""
    
    # ========== Basic Routes ==========
    @app.route('/')
    def home():
        """Root endpoint"""
        return jsonify({
            'name': 'AI-Powered Financial Risk & Audit System',
            'version': '1.0.0',
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'endpoints': [
                '/',
                '/api/health',
                '/api/test',
                '/api/routes',
                '/api/status',
                '/api/transactions',
                '/api/risk-metrics',
                '/api/audit-report',
                '/api/audit/findings',
                '/api/anomaly-detection',
                '/api/predict-risk',
                '/api/feature-importance'
            ]
        })

    @app.route('/api/test', methods=['GET'])
    def test_endpoint():
        """Simple test endpoint"""
        return jsonify({
            'message': 'API is working!',
            'timestamp': datetime.now().isoformat()
        })

    @app.route('/api/routes', methods=['GET'])
    def list_routes():
        """List all available routes"""
        routes = []
        for rule in app.url_map.iter_rules():
            if rule.endpoint != 'static':
                methods = ','.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))
                routes.append({
                    'endpoint': rule.endpoint,
                    'path': str(rule.rule),
                    'methods': methods
                })
        return jsonify({
            'routes': routes, 
            'count': len(routes),
            'timestamp': datetime.now().isoformat()
        })

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'risk_predictor': risk_predictor is not None,
                'audit_workflow': audit_workflow is not None,
                'vector_store': vector_store is not None
            }
        })

    @app.route('/api/status', methods=['GET'])
    def system_status():
        """Get detailed system status"""
        return jsonify({
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'risk_predictor': {
                    'initialized': risk_predictor is not None,
                    'has_model': risk_predictor is not None and hasattr(risk_predictor, 'model') and risk_predictor.model is not None
                },
                'vector_store': {
                    'initialized': vector_store is not None,
                    'document_count': len(vector_store.documents) if vector_store and hasattr(vector_store, 'documents') else 0
                },
                'audit_workflow': {
                    'initialized': audit_workflow is not None,
                    'has_agents': audit_workflow is not None and hasattr(audit_workflow, 'agents')
                }
            },
            'data': {
                'transactions_file': os.path.exists('data/financial_transactions.csv'),
                'model_file': os.path.exists(getattr(app.config, 'MODEL_PATH', 'models/saved/risk_model.pkl'))
            }
        })

    # ========== Transaction Routes ==========
    @app.route('/api/transactions', methods=['GET'])
    def get_transactions():
        """Get paginated transactions"""
        try:
            # Check if file exists
            data_path = 'data/financial_transactions.csv'
            if not os.path.exists(data_path):
                return jsonify({
                    'transactions': [],
                    'total': 0,
                    'page': 1,
                    'per_page': 50,
                    'total_pages': 0,
                    'message': 'No transaction data available'
                })
            
            # Load transactions from CSV
            df = pd.read_csv(data_path)
            
            # Pagination
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 50))
            
            start_idx = (page - 1) * per_page
            end_idx = min(start_idx + per_page, len(df))
            
            # Convert to dict
            transactions = df.iloc[start_idx:end_idx].to_dict('records')
            
            # Convert numpy types
            for t in transactions:
                for key, value in t.items():
                    if isinstance(value, (np.integer, np.int64)):
                        t[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        t[key] = float(value)
                    elif pd.isna(value):
                        t[key] = None
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

    @app.route('/api/transactions/<transaction_id>', methods=['GET'])
    def get_transaction(transaction_id):
        """Get single transaction by ID"""
        try:
            df = pd.read_csv('data/financial_transactions.csv')
            transaction = df[df['transaction_id'] == transaction_id].to_dict('records')
            
            if not transaction:
                return jsonify({'error': 'Transaction not found'}), 404
            
            t = transaction[0]
            for key, value in t.items():
                if isinstance(value, (np.integer, np.int64)):
                    t[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    t[key] = float(value)
                elif isinstance(value, pd.Timestamp):
                    t[key] = value.isoformat()
            
            return jsonify(t)
        except Exception as e:
            logger.error(f"Error fetching transaction: {e}")
            return jsonify({'error': str(e)}), 500

    # ========== Risk Prediction Routes ==========
    @app.route('/api/predict-risk', methods=['POST', 'GET'])
    def predict_risk():
        """Predict risk for a transaction"""
        try:
            # Handle GET requests with query parameters
            if request.method == 'GET':
                data = {
                    'transaction_id': request.args.get('transaction_id', 'TEST123'),
                    'amount': float(request.args.get('amount', 1000)),
                    'transaction_type': request.args.get('transaction_type', 'PURCHASE'),
                    'location': request.args.get('location', 'NY'),
                    'account_id': request.args.get('account_id', 'ACC001'),
                    'account_balance': float(request.args.get('account_balance', 10000))
                }
            else:  # POST
                data = request.json
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            if not risk_predictor:
                # Return mock data if predictor not available
                return jsonify({
                    'transaction_id': data.get('transaction_id', 'unknown'),
                    'risk_score': 0.75,
                    'risk_level': 'HIGH',
                    'confidence': 0.85,
                    'timestamp': datetime.now().isoformat(),
                    'note': 'Using mock data - risk predictor not initialized'
                })
            
            # Make prediction
            risk_result = risk_predictor.predict(data)
            
            return jsonify({
                'transaction_id': data.get('transaction_id', 'unknown'),
                'risk_score': risk_result['risk_score'],
                'risk_level': risk_result['risk_level'],
                'confidence': risk_result['confidence'],
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error predicting risk: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/predict-batch', methods=['POST'])
    def predict_batch():
        """Predict risk for batch of transactions"""
        try:
            data = request.json
            transactions = data.get('transactions', [])
            
            if not transactions:
                return jsonify({'error': 'No transactions provided'}), 400
            
            if not risk_predictor:
                return jsonify({'error': 'Risk predictor not initialized'}), 503
            
            df = pd.DataFrame(transactions)
            results = risk_predictor.predict_batch(df)
            
            return jsonify({
                'results': results,
                'count': len(results),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return jsonify({'error': str(e)}), 500

    # ========== Risk Metrics Routes ==========
    @app.route('/api/risk-metrics', methods=['GET'])
    def get_risk_metrics():
        """Get aggregate risk metrics"""
        try:
            df = pd.read_csv('data/financial_transactions.csv')
            
            # Calculate basic metrics
            metrics = {
                'total_transactions': len(df),
                'avg_amount': float(df['amount'].mean()),
                'total_amount': float(df['amount'].sum()),
                'transaction_types': df['transaction_type'].value_counts().to_dict(),
                'locations': df['location'].value_counts().to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add fraud metrics if available
            if 'is_fraud' in df.columns:
                metrics['fraud_count'] = int(df['is_fraud'].sum())
                metrics['fraud_rate'] = float(df['is_fraud'].mean())
            
            return jsonify(metrics)
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return jsonify({'error': str(e)}), 500

    # ========== Audit Routes ==========
    @app.route('/api/audit-report', methods=['GET'])
    def get_audit_report():
        """Generate audit report"""
        try:
            report_type = request.args.get('type', 'summary')
            
            # Simple report for testing
            report = {
                'summary': {
                    'total_transactions': 1000,
                    'high_risk_count': 50,
                    'medium_risk_count': 150,
                    'low_risk_count': 800
                },
                'generated_at': datetime.now().isoformat(),
                'type': report_type
            }
            
            return jsonify({
                'report': report,
                'generated_at': datetime.now().isoformat(),
                'type': report_type
            })
        except Exception as e:
            logger.error(f"Error generating audit report: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/audit/findings', methods=['GET'])
    def get_audit_findings():
        """Get audit findings"""
        try:
            # Mock findings for testing
            findings = [
                {
                    'id': 'FINDING_001',
                    'type': 'HIGH_VALUE',
                    'severity': 'MEDIUM',
                    'description': 'High value transaction detected',
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'id': 'FINDING_002',
                    'type': 'UNUSUAL_PATTERN',
                    'severity': 'HIGH',
                    'description': 'Unusual transaction pattern detected',
                    'timestamp': datetime.now().isoformat()
                }
            ]
            
            return jsonify(findings)
        except Exception as e:
            logger.error(f"Error getting audit findings: {e}")
            return jsonify({'error': str(e)}), 500

    # ========== Anomaly Detection Routes ==========
    @app.route('/api/anomaly-detection', methods=['GET'])
    def detect_anomalies():
        """Detect anomalies in transactions"""
        try:
            df = pd.read_csv('data/financial_transactions.csv')
            
            anomalies = []
            
            # Simple anomaly detection based on amount
            mean_amount = df['amount'].mean()
            std_amount = df['amount'].std()
            
            for idx, row in df.iterrows():
                z_score = abs((row['amount'] - mean_amount) / std_amount) if std_amount > 0 else 0
                
                if z_score > 3:
                    anomalies.append({
                        'transaction_id': row['transaction_id'],
                        'amount': float(row['amount']),
                        'z_score': float(z_score),
                        'reason': 'Statistical outlier'
                    })
                
                if len(anomalies) >= 20:
                    break
            
            return jsonify({
                'anomalies': anomalies,
                'total_anomalies': len(anomalies),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return jsonify({'error': str(e)}), 500

    # ========== Feature Importance Routes ==========
    @app.route('/api/feature-importance', methods=['GET'])
    def get_feature_importance():
        """Get feature importance from the risk model"""
        try:
            if risk_predictor and hasattr(risk_predictor, 'get_feature_importance'):
                importance = risk_predictor.get_feature_importance(top_n=10)
                return jsonify({
                    'feature_importance': importance,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                # Mock data for testing
                return jsonify({
                    'feature_importance': {
                        'amount': 0.35,
                        'hour': 0.20,
                        'location': 0.15,
                        'transaction_type': 0.12,
                        'account_age': 0.10,
                        'device_type': 0.08
                    },
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return jsonify({'error': str(e)}), 500

    # ========== Agent Routes ==========
    @app.route('/api/agent-status', methods=['GET'])
    def get_agent_status():
        """Get status of all agents"""
        try:
            if audit_workflow and hasattr(audit_workflow, 'agents'):
                agents = audit_workflow.agents
                status = {}
                
                for name, agent in agents.items():
                    status[name] = {
                        'status': getattr(agent, 'status', 'idle'),
                        'tasks_completed': getattr(agent, 'metrics', {}).get('tasks_completed', 0),
                        'last_active': datetime.now().isoformat()
                    }
                
                return jsonify({
                    'agents': status,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                # Mock data for testing
                return jsonify({
                    'agents': {
                        'risk': {'status': 'idle', 'tasks_completed': 150},
                        'audit': {'status': 'idle', 'tasks_completed': 75},
                        'compliance': {'status': 'active', 'tasks_completed': 200},
                        'investigation': {'status': 'idle', 'tasks_completed': 25},
                        'report': {'status': 'idle', 'tasks_completed': 50}
                    },
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Error getting agent status: {e}")
            return jsonify({'error': str(e)}), 500

    logger.info("=" * 50)
    logger.info("All API routes registered successfully")
    logger.info(f"Total routes: {len(list(app.url_map.iter_rules())) - 1}")  # -1 for static
    logger.info("=" * 50)
    
    return app