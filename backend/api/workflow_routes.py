from flask import jsonify, request
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def register_workflow_routes(app, audit_workflow):
    """Register workflow-related API routes"""
    
    @app.route('/api/workflow/start', methods=['POST'])
    def start_audit_workflow():
        """Start a new audit workflow"""
        try:
            data = request.json
            transactions = data.get('transactions', [])
            config = data.get('config', {})
            
            if not transactions:
                return jsonify({'error': 'No transactions provided'}), 400
            
            logger.info(f"Starting audit workflow with {len(transactions)} transactions")
            
            # Run workflow
            result = audit_workflow.run(transactions, config)
            
            return jsonify({
                'success': True,
                'workflow_id': result.get('workflow_id'),
                'status': 'completed',
                'summary': result.get('reports', {}).get('executive_summary', {}),
                'findings_count': len(result.get('audit_findings', [])),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error starting workflow: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/workflow/status', methods=['GET'])
    def get_workflow_status():
        """Get current workflow status"""
        try:
            status = audit_workflow.get_status()
            return jsonify(status)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/workflow/findings', methods=['GET'])
    def get_workflow_findings():
        """Get findings from last workflow"""
        # This would normally query from database
        return jsonify({
            'findings': [],
            'total': 0
        })
    
    @app.route('/api/workflow/report/<workflow_id>', methods=['GET'])
    def get_workflow_report(workflow_id):
        """Get report for specific workflow"""
        state = audit_workflow.get_workflow_state(workflow_id)
        
        if not state:
            return jsonify({'error': 'Workflow not found'}), 404
        
        return jsonify({
            'workflow_id': workflow_id,
            'report': state.get('reports', {}),
            'findings': state.get('audit_findings', []),
            'summary': state.get('audit_summary', {})
        })
    
    @app.route('/api/workflow/continuous/start', methods=['POST'])
    def start_continuous_audit():
        """Start continuous auditing mode"""
        try:
            data = request.json
            batch_size = data.get('batch_size', 100)
            
            # This would normally start a background process
            # For now, just return acknowledgment
            return jsonify({
                'success': True,
                'message': 'Continuous audit started',
                'batch_size': batch_size,
                'status': 'running',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/workflow/continuous/stop', methods=['POST'])
    def stop_continuous_audit():
        """Stop continuous auditing mode"""
        return jsonify({
            'success': True,
            'message': 'Continuous audit stopped',
            'timestamp': datetime.now().isoformat()
        })
    
    logger.info("Workflow routes registered successfully")