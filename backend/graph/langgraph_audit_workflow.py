"""
LangGraph-based Audit Workflow - Uses LangGraph for agent orchestration
Integrates with existing audit agents for comprehensive financial audit
"""

import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict
import pandas as pd

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("LangGraph not available, using fallback implementation")

logger = logging.getLogger(__name__)


# Define state schema for LangGraph
class AuditState(TypedDict, total=False):
    """State schema for the audit workflow"""
    workflow_id: str
    transactions: List[Dict]
    assessed_transactions: List[Dict]
    high_risk_alerts: List[Dict]
    audit_findings: List[Dict]
    compliance_violations: List[Dict]
    reports: Dict
    compliance_score: float
    risk_metrics: Dict
    current_phase: str
    timestamp: str
    progress_callback: Optional[callable]


class LangGraphAuditWorkflow:
    """LangGraph-based orchestration of audit agents"""
    
    def __init__(self, risk_predictor, vector_store=None, bedrock_client=None):
        """
        Initialize the LangGraph audit workflow
        
        Args:
            risk_predictor: ML model for risk prediction
            vector_store: Optional vector store for RAG
            bedrock_client: Optional AWS Bedrock client for LLM
        """
        self.risk_predictor = risk_predictor
        self.vector_store = vector_store
        self.bedrock_client = bedrock_client
        self.current_workflow_id: Optional[str] = None
        self.current_state: AuditState = {}
        self.status = "idle"
        
        # Initialize agents
        self._initialize_agents()
        
        # Build the graph
        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_graph()
            logger.info("✓ LangGraph workflow initialized")
        else:
            self.graph = None
            logger.warning("⚠ LangGraph unavailable, will use fallback mode")
    
    def _initialize_agents(self):
        """Initialize the audit agents"""
        try:
            # Import agents
            from ..agents.risk_agent import RiskAssessmentAgent
            from ..agents.audit_agent import AuditAgent
            from ..agents.compliance_agent import ComplianceAgent
            from ..agents.investigation_agent import InvestigationAgent
            from ..agents.report_agent import ReportAgent
            
            self.risk_agent = RiskAssessmentAgent(self.bedrock_client, self.vector_store, self.risk_predictor)
            self.audit_agent = AuditAgent(self.bedrock_client, self.vector_store)
            self.compliance_agent = ComplianceAgent(self.bedrock_client, self.vector_store)
            self.investigation_agent = InvestigationAgent(self.bedrock_client, self.vector_store)
            self.report_agent = ReportAgent(self.bedrock_client, self.vector_store)
            
            logger.info("✓ All audit agents initialized")
        except Exception as e:
            logger.warning(f"⚠ Failed to initialize agents: {e}")
            logger.info("Will use fallback implementation")
            self.risk_agent = None
            self.audit_agent = None
            self.compliance_agent = None
            self.investigation_agent = None
            self.report_agent = None
    
    def _build_graph(self) -> Optional['StateGraph']:
        """Build the LangGraph workflow graph"""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        try:
            graph = StateGraph(AuditState)
            
            # Add nodes for each phase
            graph.add_node("risk_assessment", self._node_risk_assessment)
            graph.add_node("audit_procedures", self._node_audit_procedures)
            graph.add_node("compliance_check", self._node_compliance_check)
            graph.add_node("investigation", self._node_investigation)
            graph.add_node("report_generation", self._node_report_generation)
            
            # Add edges to create workflow
            graph.add_edge("risk_assessment", "audit_procedures")
            graph.add_edge("audit_procedures", "compliance_check")
            
            # Conditional edge: only investigate if high-risk alerts exist
            graph.add_conditional_edges(
                "compliance_check",
                self._should_investigate,
                {
                    "yes": "investigation",
                    "no": "report_generation"
                }
            )
            
            graph.add_edge("investigation", "report_generation")
            graph.add_edge("report_generation", END)
            
            # Set entry point
            graph.set_entry_point("risk_assessment")
            
            logger.info("✓ LangGraph workflow graph built successfully")
            return graph.compile()
        
        except Exception as e:
            logger.error(f"✗ Failed to build LangGraph: {e}")
            return None
    
    def _node_risk_assessment(self, state: AuditState) -> AuditState:
        """Node: Risk Assessment phase"""
        logger.info(f"[{state['workflow_id']}] Executing Risk Assessment node")
        
        if self.risk_agent:
            try:
                state = self.risk_agent.process(state)
            except Exception as e:
                logger.warning(f"Risk agent failed: {e}, using fallback")
                state = self._fallback_risk_assessment(state)
        else:
            state = self._fallback_risk_assessment(state)
        
        state['current_phase'] = 'risk_assessment'
        self._emit_progress(state, 0, 'risk_assessment')
        return state
    
    def _node_audit_procedures(self, state: AuditState) -> AuditState:
        """Node: Audit Procedures phase"""
        logger.info(f"[{state['workflow_id']}] Executing Audit Procedures node")
        
        if self.audit_agent:
            try:
                state = self.audit_agent.process(state)
            except Exception as e:
                logger.warning(f"Audit agent failed: {e}, using fallback")
                state = self._fallback_audit_procedures(state)
        else:
            state = self._fallback_audit_procedures(state)
        
        state['current_phase'] = 'audit_procedures'
        self._emit_progress(state, 1, 'audit_procedures')
        return state
    
    def _node_compliance_check(self, state: AuditState) -> AuditState:
        """Node: Compliance Check phase"""
        logger.info(f"[{state['workflow_id']}] Executing Compliance Check node")
        
        if self.compliance_agent:
            try:
                state = self.compliance_agent.process(state)
            except Exception as e:
                logger.warning(f"Compliance agent failed: {e}, using fallback")
                state = self._fallback_compliance_check(state)
        else:
            state = self._fallback_compliance_check(state)
        
        state['current_phase'] = 'compliance_check'
        self._emit_progress(state, 2, 'compliance_check')
        return state
    
    def _node_investigation(self, state: AuditState) -> AuditState:
        """Node: Investigation phase"""
        logger.info(f"[{state['workflow_id']}] Executing Investigation node")
        
        if self.investigation_agent:
            try:
                state = self.investigation_agent.process(state)
            except Exception as e:
                logger.warning(f"Investigation agent failed: {e}, using fallback")
                state = self._fallback_investigation(state)
        else:
            state = self._fallback_investigation(state)
        
        state['current_phase'] = 'investigation'
        self._emit_progress(state, 3, 'investigation')
        return state
    
    def _node_report_generation(self, state: AuditState) -> AuditState:
        """Node: Report Generation phase"""
        logger.info(f"[{state['workflow_id']}] Executing Report Generation node")
        
        if self.report_agent:
            try:
                state = self.report_agent.process(state)
            except Exception as e:
                logger.warning(f"Report agent failed: {e}, using fallback")
                state = self._fallback_report_generation(state)
        else:
            state = self._fallback_report_generation(state)
        
        state['current_phase'] = 'report_generation'
        self._emit_progress(state, 4, 'report_generation')
        return state
    
    def _should_investigate(self, state: AuditState) -> str:
        """Determine if investigation phase should run"""
        high_risk_alerts = state.get('high_risk_alerts', [])
        return "yes" if len(high_risk_alerts) > 0 else "no"
    
    def _fallback_risk_assessment(self, state: AuditState) -> AuditState:
        """Fallback implementation for risk assessment"""
        transactions = state.get('transactions', [])
        assessed_transactions = []
        high_risk_alerts = []
        
        for transaction in transactions:
            risk_result = self.risk_predictor.predict(transaction)
            assessed_txn = {
                **transaction,
                'risk_score': risk_result.get('risk_score', 0),
                'risk_level': risk_result.get('risk_level', 'LOW'),
                'confidence': risk_result.get('confidence', 0.5)
            }
            assessed_transactions.append(assessed_txn)
            
            if risk_result.get('risk_level') == 'HIGH':
                high_risk_alerts.append({
                    'transaction_id': transaction.get('transaction_id'),
                    'risk_score': risk_result.get('risk_score'),
                    'amount': transaction.get('amount'),
                    'alert_level': 'HIGH'
                })
        
        state['assessed_transactions'] = assessed_transactions
        state['high_risk_alerts'] = high_risk_alerts
        return state
    
    def _fallback_audit_procedures(self, state: AuditState) -> AuditState:
        """Fallback implementation for audit procedures"""
        assessed_txns = state.get('assessed_transactions', [])
        high_risk_sample = [t for t in assessed_txns if t.get('risk_level') == 'HIGH'][:10]
        
        findings = []
        for txn in high_risk_sample:
            findings.append({
                'transaction_id': txn.get('transaction_id'),
                'finding_type': 'HIGH_RISK_TRANSACTION',
                'severity': 'HIGH',
                'description': f"High-risk transaction detected: ${txn.get('amount')} at {txn.get('location')}"
            })
        
        state['audit_findings'] = findings
        return state
    
    def _fallback_compliance_check(self, state: AuditState) -> AuditState:
        """Fallback implementation for compliance check"""
        transactions = state.get('transactions', [])
        violations = []
        
        for txn in transactions:
            # Check AML thresholds
            if txn.get('amount', 0) > 10000:
                violations.append({
                    'transaction_id': txn.get('transaction_id'),
                    'violation_type': 'AML_CTR_REQUIRED',
                    'requirement': 'Currency Transaction Report required',
                    'severity': 'HIGH'
                })
            
            # Check international transaction compliance
            if txn.get('is_international'):
                violations.append({
                    'transaction_id': txn.get('transaction_id'),
                    'violation_type': 'INTERNATIONAL_VERIFICATION',
                    'requirement': 'Enhanced due diligence required',
                    'severity': 'MEDIUM'
                })
        
        state['compliance_violations'] = violations
        return state
    
    def _fallback_investigation(self, state: AuditState) -> AuditState:
        """Fallback implementation for investigation"""
        # Investigation would perform deep dives on high-risk alerts
        high_risk_alerts = state.get('high_risk_alerts', [])
        audit_findings = state.get('audit_findings', [])
        
        # Add investigation notes to findings
        for alert in high_risk_alerts[:5]:
            audit_findings.append({
                'transaction_id': alert.get('transaction_id'),
                'finding_type': 'INVESTIGATION_NOTE',
                'severity': 'MEDIUM',
                'description': f"Deep investigation of transaction {alert.get('transaction_id')}"
            })
        
        state['audit_findings'] = audit_findings
        return state
    
    def _fallback_report_generation(self, state: AuditState) -> AuditState:
        """Fallback implementation for report generation"""
        findings = state.get('audit_findings', [])
        violations = state.get('compliance_violations', [])
        transactions = state.get('transactions', [])
        
        compliance_score = 1.0 - (len(violations) / max(len(transactions), 1))
        compliance_score = max(0, min(1, compliance_score))
        
        reports = {
            'executive_summary': {
                'total_transactions': len(transactions),
                'findings_count': len(findings),
                'violations_count': len(violations),
                'compliance_score': compliance_score,
                'compliance_status': 'COMPLIANT' if compliance_score > 0.8 else 'NON_COMPLIANT',
                'risk_level': 'HIGH' if len(violations) > 0 else 'LOW'
            },
            'findings_report': findings,
            'compliance_report': violations
        }
        
        state['reports'] = reports
        state['compliance_score'] = compliance_score
        return state
    
    def _emit_progress(self, state: AuditState, phase_number: int, phase_name: str):
        """Emit progress callback if provided"""
        if state.get('progress_callback'):
            try:
                state['progress_callback'](phase_number, phase_name)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def run(self, transactions: List[Dict], progress_callback=None) -> Dict[str, Any]:
        """
        Execute the audit workflow using LangGraph
        
        Args:
            transactions: List of transaction dictionaries to audit
            progress_callback: Optional callback(phase_number, phase_name) for progress updates
            
        Returns:
            Dictionary with audit results
        """
        workflow_id = f"AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.current_workflow_id = workflow_id
        self.status = "processing"
        
        logger.info(f"Starting LangGraph audit workflow: {workflow_id}")
        logger.info(f"Processing {len(transactions)} transactions")
        
        try:
            # Initialize state
            initial_state: AuditState = {
                'workflow_id': workflow_id,
                'transactions': transactions,
                'assessed_transactions': [],
                'high_risk_alerts': [],
                'audit_findings': [],
                'compliance_violations': [],
                'reports': {},
                'compliance_score': 1.0,
                'risk_metrics': {},
                'current_phase': 'initialization',
                'timestamp': datetime.now().isoformat(),
                'progress_callback': progress_callback
            }
            
            # Execute workflow
            if self.graph:
                logger.info("Using LangGraph execution")
                try:
                    result_state = self.graph.invoke(initial_state)
                    logger.info(f"LangGraph execution completed successfully")
                    return self._format_results(result_state)
                except Exception as e:
                    logger.error(f"LangGraph execution failed: {e}, falling back to sequential execution")
                    return self._execute_sequential(initial_state)
            else:
                logger.info("Using sequential execution (LangGraph unavailable)")
                return self._execute_sequential(initial_state)
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'workflow_id': workflow_id,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _execute_sequential(self, state: AuditState) -> Dict[str, Any]:
        """Execute workflow phases sequentially (fallback)"""
        logger.info(f"[{state['workflow_id']}] Executing sequential workflow")
        
        state = self._node_risk_assessment(state)
        state = self._node_audit_procedures(state)
        state = self._node_compliance_check(state)
        
        if self._should_investigate(state) == "yes":
            state = self._node_investigation(state)
        
        state = self._node_report_generation(state)
        
        return self._format_results(state)
    
    def _format_results(self, state: AuditState) -> Dict[str, Any]:
        """Format the final results"""
        return {
            'workflow_id': state.get('workflow_id'),
            'status': 'completed',
            'audit_findings': state.get('audit_findings', []),
            'compliance_violations': state.get('compliance_violations', []),
            'high_risk_alerts': state.get('high_risk_alerts', []),
            'reports': state.get('reports', {}),
            'compliance_score': state.get('compliance_score', 0),
            'risk_metrics': state.get('risk_metrics', {}),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_status(self) -> str:
        """Get current workflow status"""
        return self.status
