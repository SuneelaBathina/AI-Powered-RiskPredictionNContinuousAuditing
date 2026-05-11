from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from datetime import datetime
import logging
import uuid
import json

from .state import AuditState, AuditConfig
from agents.risk_agent import RiskAssessmentAgent
from agents.audit_agent import AuditAgent
from agents.compliance_agent import ComplianceAgent
from agents.investigation_agent import InvestigationAgent
from agents.report_agent import ReportAgent

logger = logging.getLogger(__name__)


class AuditWorkflow:
    """Orchestrates the complete audit process using LangGraph"""
    
    def __init__(self, risk_predictor, vector_store, bedrock_client):
        self.risk_predictor = risk_predictor
        self.vector_store = vector_store
        self.bedrock_client = bedrock_client
        self.workflow_id = None
        
        # Default configuration
        self.config = {
            'enable_parallel_processing': True,
            'risk_threshold_high': 0.7,
            'risk_threshold_medium': 0.3,
            'ctr_threshold': 10000,
            'sar_threshold': 5000,
            'max_investigations': 10,
            'auto_escalate': True,
            'notification_emails': ['compliance@bank.com']
        }
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Build workflow graph
        self.graph = self._build_graph()
        self.memory = MemorySaver()
    
    def _initialize_agents(self):
        """Initialize all agents for the workflow"""
        return {
            'risk': RiskAssessmentAgent(
                self.bedrock_client, 
                self.vector_store, 
                self.risk_predictor
            ),
            'audit': AuditAgent(self.bedrock_client, self.vector_store),
            'compliance': ComplianceAgent(self.bedrock_client, self.vector_store),
            'investigation': InvestigationAgent(self.bedrock_client, self.vector_store),
            'report': ReportAgent(self.bedrock_client, self.vector_store)
        }
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(AuditState)
        
        # Add nodes
        workflow.add_node("risk_assessment", self._risk_assessment_node)
        workflow.add_node("audit_procedures", self._audit_procedures_node)
        workflow.add_node("compliance_check", self._compliance_check_node)
        workflow.add_node("investigation", self._investigation_node)
        workflow.add_node("report_generation", self._report_generation_node)
        workflow.add_node("escalation", self._escalation_node)
        
        # Define edges
        workflow.set_entry_point("risk_assessment")
        
        # Conditional routing
        workflow.add_conditional_edges(
            "risk_assessment",
            self._route_after_risk_assessment,
            {
                "audit": "audit_procedures",
                "investigation": "investigation",
                "compliance": "compliance_check"
            }
        )
        
        workflow.add_edge("audit_procedures", "compliance_check")
        workflow.add_edge("compliance_check", "investigation")
        workflow.add_edge("investigation", "report_generation")
        
        workflow.add_conditional_edges(
            "report_generation",
            self._should_escalate,
            {
                "escalate": "escalation",
                "complete": END
            }
        )
        
        workflow.add_edge("escalation", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _risk_assessment_node(self, state: AuditState) -> AuditState:
        """Run risk assessment on all transactions"""
        logger.info("Running risk assessment agent...")
        state['current_phase'] = 'risk_assessment'
        state['timestamp'] = datetime.now().isoformat()
        self._emit_progress(state, 0, 'risk_assessment')
        
        try:
            result = self.agents['risk'].process(state)
            return result
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            state['errors'].append(str(e))
            return state
    
    def _audit_procedures_node(self, state: AuditState) -> AuditState:
        """Execute audit procedures"""
        logger.info("Running audit procedures...")
        state['current_phase'] = 'audit'
        self._emit_progress(state, 1, 'audit_procedures')
        
        try:
            result = self.agents['audit'].process(state)
            return result
        except Exception as e:
            logger.error(f"Audit procedures failed: {e}")
            state['errors'].append(str(e))
            return state
    
    def _compliance_check_node(self, state: AuditState) -> AuditState:
        """Perform compliance checks"""
        logger.info("Running compliance checks...")
        state['current_phase'] = 'compliance'
        self._emit_progress(state, 2, 'compliance_check')
        
        try:
            result = self.agents['compliance'].process(state)
            return result
        except Exception as e:
            logger.error(f"Compliance check failed: {e}")
            state['errors'].append(str(e))
            return state
    
    def _investigation_node(self, state: AuditState) -> AuditState:
        """Investigate high-risk findings"""
        logger.info("Running investigation agent...")
        state['current_phase'] = 'investigation'
        self._emit_progress(state, 3, 'investigation')
        
        try:
            result = self.agents['investigation'].process(state)
            return result
        except Exception as e:
            logger.error(f"Investigation failed: {e}")
            state['errors'].append(str(e))
            return state
    
    def _report_generation_node(self, state: AuditState) -> AuditState:
        """Generate audit reports"""
        logger.info("Generating audit reports...")
        state['current_phase'] = 'reporting'
        self._emit_progress(state, 4, 'report_generation')
        
        try:
            result = self.agents['report'].process(state)
            return result
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            state['errors'].append(str(e))
            return state
    
    def _escalation_node(self, state: AuditState) -> AuditState:
        """Escalate critical findings"""
        logger.info("Escalating critical findings...")
        state['current_phase'] = 'escalation'
        
        # Add escalation details
        state['reports']['escalation'] = {
            'timestamp': datetime.now().isoformat(),
            'reason': 'Critical findings requiring management attention',
            'findings': [f for f in state.get('audit_findings', []) 
                        if f.get('severity') == 'CRITICAL'],
            'notified': self.config.get('notification_emails', [])
        }
        
        return state
    
    def _route_after_risk_assessment(self, state: AuditState) -> str:
        """Determine next step based on risk assessment"""
        high_risk_count = len(state.get('high_risk_alerts', []))
        
        if high_risk_count > 10:
            return "investigation"  # Immediate investigation
        elif high_risk_count > 0:
            return "audit"  # Normal audit path
        else:
            return "compliance"  # Skip audit if no risks
    
    def _emit_progress(self, state: AuditState, phase_number: int, phase_name: str):
        """Invoke the progress callback if set."""
        callback = state.get('progress_callback')
        if callback:
            try:
                callback(phase_number, phase_name)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def _should_escalate(self, state: AuditState) -> str:
        """Determine if findings should be escalated"""
        critical_findings = [f for f in state.get('audit_findings', []) 
                            if f.get('severity') == 'CRITICAL']
        
        if critical_findings and self.config.get('auto_escalate'):
            return "escalate"
        return "complete"
    
    def run(self, transactions: List[Dict], config: Optional[Dict] = None, progress_callback=None) -> Dict[str, Any]:
        """Run the complete audit workflow"""
        
        # Update configuration if provided
        if config:
            self.config.update(config)
        
        # Generate workflow ID
        self.workflow_id = f"AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        logger.info(f"Starting audit workflow: {self.workflow_id}")
        logger.info(f"Processing {len(transactions)} transactions")
        
        # Initialize state
        initial_state: AuditState = {
            'transactions': transactions,
            'audit_period': {
                'start': datetime.now().isoformat(),
                'end': datetime.now().isoformat()
            },
            'config': self.config,
            'assessed_transactions': [],
            'high_risk_alerts': [],
            'risk_metrics': {},
            'audit_findings': [],
            'audit_summary': {},
            'audit_recommendations': '',
            'compliance_violations': [],
            'regulatory_reports': [],
            'compliance_score': 1.0,
            'compliance_report': '',
            'investigations': [],
            'investigation_reports': [],
            'reports': {},
            'current_phase': 'initializing',
            'timestamp': datetime.now().isoformat(),
            'errors': [],
            'workflow_id': self.workflow_id,
            'progress_callback': progress_callback
        }
        
        # Run the workflow
        try:
            final_state = self.graph.invoke(initial_state)
            logger.info(f"Audit workflow completed: {self.workflow_id}")
            
            # Add workflow summary
            final_state['reports']['workflow_summary'] = {
                'workflow_id': self.workflow_id,
                'started_at': initial_state['timestamp'],
                'completed_at': datetime.now().isoformat(),
                'phases_completed': final_state['current_phase'],
                'errors_count': len(final_state.get('errors', [])),
                'total_transactions_processed': len(transactions),
                'high_risk_found': len(final_state.get('high_risk_alerts', []))
            }
            
            return final_state
            
        except Exception as e:
            logger.error(f"Audit workflow failed: {e}")
            initial_state['errors'].append(str(e))
            return initial_state
    
    def run_continuous(self, transaction_stream, batch_size: int = 100):
        """Run continuous auditing on a stream of transactions"""
        logger.info("Starting continuous audit mode...")
        
        batch = []
        for transaction in transaction_stream:
            batch.append(transaction)
            
            if len(batch) >= batch_size:
                # Process batch
                result = self.run(batch)
                
                # Yield real-time alerts
                for alert in result.get('high_risk_alerts', [])[:5]:
                    yield {
                        'type': 'REAL_TIME_ALERT',
                        'data': alert,
                        'timestamp': datetime.now().isoformat(),
                        'workflow_id': self.workflow_id
                    }
                
                # Yield summary
                yield {
                    'type': 'BATCH_SUMMARY',
                    'data': {
                        'batch_size': len(batch),
                        'high_risk_count': len(result.get('high_risk_alerts', [])),
                        'findings_count': len(result.get('audit_findings', [])),
                        'compliance_score': result.get('compliance_score', 0)
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                batch = []
        
        # Process remaining transactions
        if batch:
            result = self.run(batch)
            yield {
                'type': 'FINAL_BATCH',
                'data': result.get('reports', {}),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            'workflow_id': self.workflow_id,
            'status': 'active' if self.workflow_id else 'idle',
            'agents': {
                name: {
                    'status': getattr(agent, 'status', 'idle'),
                    'tasks_completed': getattr(agent, 'metrics', {}).get('tasks_completed', 0)
                }
                for name, agent in self.agents.items()
            }
        }
    
    def get_workflow_state(self, workflow_id: str = None) -> Optional[Dict]:
        """Retrieve saved workflow state"""
        # This would normally load from a database
        # For now, returns None
        return None