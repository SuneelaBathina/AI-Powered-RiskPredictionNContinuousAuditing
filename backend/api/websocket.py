from flask_socketio import emit, join_room, leave_room
from flask import request
import logging
from datetime import datetime
import pandas as pd
import math

logger = logging.getLogger(__name__)


def convert_to_serializable(obj):
    """Convert non-serializable objects to JSON serializable format"""
    if obj is None:
        return None
    if pd.isna(obj) or (isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj))):
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    try:
        return str(obj)
    except Exception:
        return None


def register_socket_handlers(socketio, audit_workflow):
    """Register WebSocket event handlers for real-time updates"""
    
    @socketio.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {
            'status': 'connected',
            'timestamp': datetime.now().isoformat()
        })
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")
    
    @socketio.on('subscribe_workflow')
    def handle_subscribe_workflow(data):
        """Subscribe to workflow updates"""
        workflow_id = data.get('workflow_id')
        if workflow_id:
            room = f"workflow_{workflow_id}"
            join_room(room)
            emit('subscribed', {
                'workflow_id': workflow_id,
                'room': room,
                'status': 'subscribed'
            })
            logger.info(f"Client {request.sid} subscribed to {room}")
    
    @socketio.on('unsubscribe_workflow')
    def handle_unsubscribe_workflow(data):
        """Unsubscribe from workflow updates"""
        workflow_id = data.get('workflow_id')
        if workflow_id:
            room = f"workflow_{workflow_id}"
            leave_room(room)
            emit('unsubscribed', {
                'workflow_id': workflow_id,
                'status': 'unsubscribed'
            })
    
    @socketio.on('start_audit')
    def handle_start_audit(data):
        """Start audit workflow via WebSocket"""
        logger.info(f"Handle start_audit called by client {request.sid}")
        transactions = data.get('transactions', [])
        workflow_id = data.get('workflow_id')
        
        logger.info(f"Received {len(transactions)} transactions, workflow_id={workflow_id}")
        
        if not transactions:
            logger.warning("No transactions in request")
            emit('audit_error', {'error': 'No transactions provided'})
            return
        
        if not audit_workflow:
            logger.error("Audit workflow is None - workflow not initialized")
            emit('audit_error', {'error': 'Audit workflow not initialized. Server may not have started properly.'})
            return
        
        logger.info(f"Audit workflow initialized: {audit_workflow}")
        
        # Run workflow asynchronously
        def run_async():
            room = request.sid
            try:
                logger.info(f"Starting async audit run for workflow {workflow_id}")
                socketio.emit('audit_started', {
                    'status': 'started',
                    'workflow_id': workflow_id,
                    'timestamp': datetime.now().isoformat()
                }, room=room)
                
                logger.info(f"Calling audit_workflow.run() with {len(transactions)} transactions")
                
                # Define progress callback to emit phase updates
                def phase_progress(phase_number, phase_name):
                    logger.info(f"Workflow phase {phase_number}: {phase_name}")
                    socketio.emit('phase_progress', {
                        'phase': phase_name,
                        'phase_number': phase_number,
                        'total_phases': 5,
                        'timestamp': datetime.now().isoformat()
                    }, room=room)
                    
                    # Update agent statuses based on current phase
                    # Note: We can't access 'findings' or 'result' here as they're not defined yet
                    if phase_name == 'risk_assessment':
                        socketio.emit('AGENT_UPDATE', {'agent': 'risk', 'state': {'status': 'processing', 'tasks': phase_number}}, room=room)
                        # Set others to idle
                        socketio.emit('AGENT_UPDATE', {'agent': 'audit', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
                        socketio.emit('AGENT_UPDATE', {'agent': 'compliance', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
                        socketio.emit('AGENT_UPDATE', {'agent': 'investigation', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
                        socketio.emit('AGENT_UPDATE', {'agent': 'report', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
                    elif phase_name == 'audit_procedures':
                        socketio.emit('AGENT_UPDATE', {'agent': 'risk', 'state': {'status': 'completed', 'tasks': len(transactions)}}, room=room)
                        socketio.emit('AGENT_UPDATE', {'agent': 'audit', 'state': {'status': 'processing', 'tasks': phase_number}}, room=room)
                    elif phase_name == 'compliance_check':
                        socketio.emit('AGENT_UPDATE', {'agent': 'audit', 'state': {'status': 'completed', 'tasks': phase_number}}, room=room)
                        socketio.emit('AGENT_UPDATE', {'agent': 'compliance', 'state': {'status': 'processing', 'tasks': phase_number}}, room=room)
                    elif phase_name == 'investigation':
                        socketio.emit('AGENT_UPDATE', {'agent': 'compliance', 'state': {'status': 'completed', 'tasks': phase_number}}, room=room)
                        socketio.emit('AGENT_UPDATE', {'agent': 'investigation', 'state': {'status': 'processing', 'tasks': phase_number}}, room=room)
                    elif phase_name == 'report_generation':
                        socketio.emit('AGENT_UPDATE', {'agent': 'investigation', 'state': {'status': 'completed', 'tasks': phase_number}}, room=room)
                        socketio.emit('AGENT_UPDATE', {'agent': 'report', 'state': {'status': 'processing', 'tasks': phase_number}}, room=room)
                
                # Run workflow with progress callback and timeout
                import threading
                import time
                
                result = None
                workflow_error = None
                
                def run_workflow():
                    nonlocal result, workflow_error
                    try:
                        result = audit_workflow.run(transactions, progress_callback=phase_progress)
                        logger.info("Workflow completed successfully")
                    except Exception as e:
                        workflow_error = e
                        logger.error(f"Workflow execution error: {e}")
                
                # Start workflow in a thread with timeout and progress simulation
                workflow_thread = threading.Thread(target=run_workflow)
                workflow_thread.start()
                
                # Simulate progress updates while workflow runs
                phases = [
                    ('risk_assessment', 'risk'),
                    ('audit_procedures', 'audit'), 
                    ('compliance_check', 'compliance'),
                    ('investigation', 'investigation'),
                    ('report_generation', 'report')
                ]
                
                for i, (phase_name, agent_id) in enumerate(phases):
                    if not workflow_thread.is_alive():
                        break  # Workflow completed
                    
                    # Update current agent to processing
                    socketio.emit('AGENT_UPDATE', {'agent': agent_id, 'state': {'status': 'processing', 'tasks': i+1}}, room=room)
                    
                    # Send phase progress
                    socketio.emit('phase_progress', {
                        'phase': phase_name,
                        'phase_number': i,
                        'total_phases': 5,
                        'timestamp': datetime.now().isoformat()
                    }, room=room)
                    
                    # Wait a bit before next phase
                    time.sleep(2)
                
                # Wait for workflow to complete or timeout
                workflow_thread.join(timeout=30)  # 30 second timeout
                
                if workflow_thread.is_alive():
                    logger.warning("Workflow timed out after 30 seconds")
                    socketio.emit('audit_error', {'error': 'Workflow timed out after 30 seconds'}, room=room)
                    # Reset agent statuses on timeout
                    for phase_name, agent_id in phases:
                        socketio.emit('AGENT_UPDATE', {'agent': agent_id, 'state': {'status': 'idle', 'tasks': 0}}, room=room)
                    return
                
                if workflow_error:
                    logger.error(f"Workflow failed with error: {workflow_error}")
                    socketio.emit('audit_error', {'error': f'Audit failed: {str(workflow_error)}'}, room=room)
                    # Reset agent statuses on error
                    socketio.emit('AGENT_UPDATE', {'agent': 'risk', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
                    socketio.emit('AGENT_UPDATE', {'agent': 'audit', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
                    socketio.emit('AGENT_UPDATE', {'agent': 'compliance', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
                    socketio.emit('AGENT_UPDATE', {'agent': 'investigation', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
                    socketio.emit('AGENT_UPDATE', {'agent': 'report', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
                    return
                
                if not result:
                    logger.error("Workflow returned no result")
                    socketio.emit('audit_error', {'error': 'Workflow returned no result'}, room=room)
                    return
                
                logger.info(f"Audit workflow completed. Result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
                
                # Send findings in chunks (or at least one progress event)
                findings = result.get('audit_findings', [])
                logger.info(f"Got {len(findings)} audit findings from workflow")
                
                if findings:
                    for i in range(0, len(findings), 10):
                        findings_chunk = findings[i:i+10]
                        socketio.emit('audit_findings_chunk', {
                            'findings': convert_to_serializable(findings_chunk),
                            'progress': i + len(findings_chunk),
                            'total': len(findings)
                        }, room=room)
                else:
                    # Even if no findings, emit at least one progress event
                    socketio.emit('audit_findings_chunk', {
                        'findings': [],
                        'progress': 1,
                        'total': 1
                    }, room=room)
                
                # Prepare executive summary for frontend
                exec_summary = convert_to_serializable(result.get('reports', {}).get('executive_summary', {}))
                
                # Build comprehensive result object
                full_results = {
                    'audit_findings': convert_to_serializable(result.get('audit_findings', [])),
                    'compliance_violations': convert_to_serializable(result.get('compliance_violations', [])),
                    'high_risk_alerts': convert_to_serializable(result.get('high_risk_alerts', [])),
                    'reports': convert_to_serializable(result.get('reports', {})),
                    'compliance_score': result.get('compliance_score', 0),
                    'risk_metrics': convert_to_serializable(result.get('risk_metrics', {}))
                }
                
                logger.info(f"Emitting audit_complete event with full results")
                socketio.emit('audit_complete', {
                    'status': 'complete',
                    'workflow_id': result.get('workflow_id'),
                    'summary': exec_summary,
                    'findings_count': len(findings),
                    'full_results': full_results,
                    'timestamp': datetime.now().isoformat()
                }, room=room)
                
                # Update agent statuses to completed
                socketio.emit('AGENT_UPDATE', {'agent': 'risk', 'state': {'status': 'completed', 'tasks': len(transactions)}}, room=room)
                socketio.emit('AGENT_UPDATE', {'agent': 'audit', 'state': {'status': 'completed', 'tasks': len(findings)}}, room=room)
                socketio.emit('AGENT_UPDATE', {'agent': 'compliance', 'state': {'status': 'completed', 'tasks': len(result.get('compliance_violations', []))}}, room=room)
                socketio.emit('AGENT_UPDATE', {'agent': 'investigation', 'state': {'status': 'completed', 'tasks': len(result.get('investigations', []))}}, room=room)
                socketio.emit('AGENT_UPDATE', {'agent': 'report', 'state': {'status': 'completed', 'tasks': 1}}, room=room)
        
            except Exception as e:
                logger.error(f"Async audit error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                socketio.emit('audit_error', {'error': f'Audit failed: {str(e)}'}, room=room)
                
                # Reset agent statuses on error
                socketio.emit('AGENT_UPDATE', {'agent': 'risk', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
                socketio.emit('AGENT_UPDATE', {'agent': 'audit', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
                socketio.emit('AGENT_UPDATE', {'agent': 'compliance', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
                socketio.emit('AGENT_UPDATE', {'agent': 'investigation', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
                socketio.emit('AGENT_UPDATE', {'agent': 'report', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
        
        logger.info(f"Starting background task for workflow {workflow_id}")
        socketio.start_background_task(run_async)
    
    @socketio.on('get_workflow_status')
    def handle_get_status(data):
        """Get real-time workflow status"""
        if not audit_workflow:
            emit('workflow_status', {'status': 'not_initialized'})
            return
        try:
            if hasattr(audit_workflow, 'get_status'):
                status = audit_workflow.get_status()
            else:
                status = {'status': 'unknown'}
            emit('workflow_status', status)
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            emit('workflow_status', {'status': 'error', 'error': str(e)})
    
    @socketio.on('START_WORKFLOW')
    def handle_start_workflow(data):
        """Start workflow from AgentConsole"""
        logger.info("Starting workflow from AgentConsole")
        try:
            # Load transactions from CSV
            df = pd.read_csv('data/financial_transactions.csv')
            transactions = df.to_dict('records')
            
            # Emit agent updates - initialize all as idle, will be updated during workflow
            socketio.emit('AGENT_UPDATE', {'agent': 'risk', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
            socketio.emit('AGENT_UPDATE', {'agent': 'audit', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
            socketio.emit('AGENT_UPDATE', {'agent': 'compliance', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
            socketio.emit('AGENT_UPDATE', {'agent': 'investigation', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
            socketio.emit('AGENT_UPDATE', {'agent': 'report', 'state': {'status': 'idle', 'tasks': 0}}, room=room)
            
            # Start audit
            handle_start_audit({'transactions': transactions[:100], 'workflow_id': 'agent_console'})  # Limit to 100 for demo
            
        except Exception as e:
            logger.error(f"Error starting workflow: {e}")
            emit('workflow_error', {'error': str(e)})
    
    @socketio.on('STOP_WORKFLOW')
    def handle_stop_workflow(data):
        """Stop workflow from AgentConsole"""
        logger.info("Stopping workflow from AgentConsole")
        room = request.sid
        socketio.emit('AGENT_UPDATE', {'agent': 'risk', 'state': {'status': 'idle'}}, room=room)
        socketio.emit('AGENT_UPDATE', {'agent': 'audit', 'state': {'status': 'idle'}}, room=room)
        socketio.emit('AGENT_UPDATE', {'agent': 'compliance', 'state': {'status': 'idle'}}, room=room)
        socketio.emit('AGENT_UPDATE', {'agent': 'investigation', 'state': {'status': 'idle'}}, room=room)
        socketio.emit('AGENT_UPDATE', {'agent': 'report', 'state': {'status': 'idle'}}, room=room)
        socketio.emit('workflow_stopped', {}, room=room)
    
    @socketio.on('REFRESH_AGENTS')
    def handle_refresh_agents(data):
        """Refresh agent status"""
        logger.info("Refreshing agents")
        room = request.sid
        # Send current status
        socketio.emit('AGENT_UPDATE', {'agent': 'risk', 'state': {'status': 'idle', 'tasks': 0, 'memory': []}}, room=room)
        socketio.emit('AGENT_UPDATE', {'agent': 'audit', 'state': {'status': 'idle', 'tasks': 0, 'memory': []}}, room=room)
        socketio.emit('AGENT_UPDATE', {'agent': 'compliance', 'state': {'status': 'idle', 'tasks': 0, 'memory': []}}, room=room)
        socketio.emit('AGENT_UPDATE', {'agent': 'investigation', 'state': {'status': 'idle', 'tasks': 0, 'memory': []}}, room=room)
        socketio.emit('AGENT_UPDATE', {'agent': 'report', 'state': {'status': 'idle', 'tasks': 0, 'memory': []}}, room=room)
    
    logger.info("WebSocket handlers registered successfully")