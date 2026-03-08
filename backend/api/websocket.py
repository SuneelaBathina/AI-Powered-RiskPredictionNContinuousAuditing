from flask_socketio import emit, join_room, leave_room
from flask import request
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

def register_socket_handlers(socketio, audit_workflow):
    """Register WebSocket event handlers"""
    
    @socketio.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {'status': 'connected', 'timestamp': datetime.now().isoformat()})

    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")

    @socketio.on('subscribe')
    def handle_subscribe(data):
        """Subscribe to specific channels"""
        channel = data.get('channel')
        if channel:
            join_room(channel)
            emit('subscribed', {'channel': channel, 'status': 'success'})
            logger.info(f"Client {request.sid} subscribed to {channel}")

    @socketio.on('unsubscribe')
    def handle_unsubscribe(data):
        """Unsubscribe from channels"""
        channel = data.get('channel')
        if channel:
            leave_room(channel)
            emit('unsubscribed', {'channel': channel, 'status': 'success'})

    @socketio.on('start_monitoring')
    def handle_start_monitoring(data):
        """Start real-time transaction monitoring"""
        transaction_id = data.get('transaction_id')
        if transaction_id:
            # Start monitoring logic here
            emit('monitoring_started', {
                'transaction_id': transaction_id,
                'status': 'monitoring',
                'timestamp': datetime.now().isoformat()
            }, room=request.sid)

    @socketio.on('request_agent_update')
    def handle_agent_update(data):
        """Request real-time agent status update"""
        agent_name = data.get('agent')
        
        if audit_workflow and hasattr(audit_workflow, 'agents'):
            agent = audit_workflow.agents.get(agent_name)
            if agent:
                emit('agent_status', {
                    'agent': agent_name,
                    'status': getattr(agent, 'status', 'idle'),
                    'metrics': getattr(agent, 'metrics', {}),
                    'timestamp': datetime.now().isoformat()
                }, room=request.sid)

    @socketio.on('run_audit')
    def handle_run_audit(data):
        """Trigger audit workflow via WebSocket"""
        transactions = data.get('transactions', [])
        
        if audit_workflow and transactions:
            # Run audit in background
            def run_async():
                try:
                    result = audit_workflow.run(transactions)
                    # Convert result to serializable format
                    serializable_result = convert_to_serializable(result)
                    emit('audit_complete', {
                        'status': 'complete',
                        'result': serializable_result,
                        'timestamp': datetime.now().isoformat()
                    }, room=request.sid)
                except Exception as e:
                    logger.error(f"Error in async audit: {e}")
                    emit('audit_error', {
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }, room=request.sid)
            
            socketio.start_background_task(run_async)
            emit('audit_started', {'status': 'started', 'timestamp': datetime.now().isoformat()}, room=request.sid)

    @socketio.on('ping')
    def handle_ping():
        """Heartbeat ping"""
        emit('pong', {'timestamp': datetime.now().isoformat()}, room=request.sid)

    logger.info("WebSocket handlers registered successfully")


def convert_to_serializable(obj):
    """Convert non-serializable objects to JSON serializable format"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    if hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    try:
        # Try to convert to string if all else fails
        return str(obj)
    except:
        return None