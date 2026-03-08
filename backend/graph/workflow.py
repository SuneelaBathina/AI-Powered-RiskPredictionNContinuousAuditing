from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator
from datetime import datetime
import logging

from agents.risk_agent import RiskAssessmentAgent
from agents.audit_agent import AuditAgent
from agents.compliance_agent import ComplianceAgent
from agents.investigation_agent import InvestigationAgent
from agents.report_agent import ReportAgent

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State schema for the agent workflow"""
    transactions: List[Dict]
    assessed_transactions: Annotated[List[Dict], operator.add]
    high_risk_alerts: Annotated[List[Dict], operator.add]
    risk_metrics: Dict
    audit_findings: Annotated[List[Dict], operator.add]
    audit_summary: Dict
    audit_recommendations: str
    compliance_violations: Annotated[List[Dict], operator.add]
    regulatory_reports: Annotated[List[Dict], operator.add]
    compliance_report: str
    compliance_score: float
    investigations: Annotated[List[Dict], operator.add]
    investigation_reports: Annotated[List[Dict], operator.add]
    reports: Dict
    current_phase: str
    timestamp: str
    errors: List[str]

class AuditWorkflow:
    """Main workflow orchestrator using LangGraph"""
    
    def __init__(self, risk_predictor, vector_store, bedrock_client):
        self.risk_predictor = risk_predictor
        self.vector_store = vector_store
        self.bedrock_client = bedrock_client
        
        # Initialize agents
        self.agents = {
            'risk': RiskAssessmentAgent(bedrock_client, vector_store, risk_predictor),
            'audit': AuditAgent(bedrock_client, vector_store),
            'compliance': ComplianceAgent(bedrock_client, vector_store),
            'investigation': InvestigationAgent(bedrock_client, vector_store),
            'report': ReportAgent(bedrock_client, vector_store)
        }
        
        # Build workflow graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the agent workflow graph"""
        
        # Define graph nodes
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("risk_assessment", self._risk_assessment_node)
        workflow.add_node("audit", self._audit_node)
        workflow.add_node("compliance", self._compliance_node)
        workflow.add_node("investigation", self._investigation_node)
        workflow.add_node("report_generation", self._report_node)
        
        # Define edges
        workflow.set_entry_point("risk_assessment")
        
        # Conditional routing based on findings
        workflow.add_conditional_edges(
            "risk_assessment",
            self._should_audit,
            {
                "audit": "audit",
                "investigation": "investigation",
                "compliance": "compliance"
            }
        )
        
        workflow.add_edge("audit", "compliance")
        workflow.add_edge("compliance", "investigation")
        workflow.add_edge("investigation", "report_generation")
        workflow.add_edge("report_generation", END)
        
        # Add parallel processing option
        if hasattr(self, 'enable_parallel') and self.enable_parallel:
            workflow.add_node("parallel_investigations", self._parallel_investigations)
        
        return workflow.compile()
    
    def _risk_assessment_node(self, state: AgentState) -> AgentState:
        """Risk assessment agent node"""
        logger.info("Running risk assessment agent")
        state['current_phase'] = 'risk_assessment'
        return self.agents['risk'].process(state)
    
    def _audit_node(self, state: AgentState) -> AgentState:
        """Audit agent node"""
        logger.info("Running audit agent")
        state['current_phase'] = 'audit'
        return self.agents['audit'].process(state)
    
    def _compliance_node(self, state: AgentState) -> AgentState:
        """Compliance agent node"""
        logger.info("Running compliance agent")
        state['current_phase'] = 'compliance'
        return self.agents['compliance'].process(state)
    
    def _investigation_node(self, state: AgentState) -> AgentState:
        """Investigation agent node"""
        logger.info("Running investigation agent")
        state['current_phase'] = 'investigation'
        return self.agents['investigation'].process(state)
    
    def _report_node(self, state: AgentState) -> AgentState:
        """Report generation agent node"""
        logger.info("Running report generation agent")
        state['current_phase'] = 'reporting'
        return self.agents['report'].process(state)
    
    def _parallel_investigations(self, state: AgentState) -> AgentState:
        """Parallel investigation processing for multiple high-risk items"""
        logger.info("Running parallel investigations")
        
        # Split investigations by priority
        high_priority = [a for a in state.get('high_risk_alerts', []) if a.get('risk_score', 0) > 0.9]
        medium_priority = [a for a in state.get('high_risk_alerts', []) if 0.7 < a.get('risk_score', 0) <= 0.9]
        
        # Process in parallel (simulated here)
        all_investigations = []
        for alert in high_priority + medium_priority:
            inv = self.agents['investigation']._investigate_alert(alert, state)
            all_investigations.append(inv)
        
        state['investigations'] = all_investigations
        return state
    
    def _should_audit(self, state: AgentState) -> str:
        """Determine next step based on risk assessment"""
        high_risk_count = len(state.get('high_risk_alerts', []))
        
        if high_risk_count > 10:
            return "investigation"  # Immediate investigation for many high-risk items
        elif high_risk_count > 0:
            return "audit"  # Normal audit path
        else:
            return "compliance"  # Skip to compliance if no risks
    
    def run(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Run the complete audit workflow"""
        logger.info(f"Starting audit workflow with {len(transactions)} transactions")
        
        # Initialize state
        initial_state = {
            'transactions': transactions,
            'assessed_transactions': [],
            'high_risk_alerts': [],
            'risk_metrics': {},
            'audit_findings': [],
            'audit_summary': {},
            'audit_recommendations': '',
            'compliance_violations': [],
            'regulatory_reports': [],
            'compliance_report': '',
            'compliance_score': 1.0,
            'investigations': [],
            'investigation_reports': [],
            'reports': {},
            'current_phase': 'initializing',
            'timestamp': datetime.now().isoformat(),
            'errors': []
        }
        
        # Run the workflow
        try:
            final_state = self.graph.invoke(initial_state)
            logger.info("Audit workflow completed successfully")
            return final_state
        except Exception as e:
            logger.error(f"Error in audit workflow: {e}")
            initial_state['errors'].append(str(e))
            return initial_state
    
    def run_continuous(self, transaction_stream):
        """Run continuous auditing on a stream of transactions"""
        logger.info("Starting continuous audit mode")
        
        batch_size = 100
        batch = []
        
        for transaction in transaction_stream:
            batch.append(transaction)
            
            if len(batch) >= batch_size:
                # Process batch
                result = self.run(batch)
                
                # Yield real-time alerts
                for alert in result.get('high_risk_alerts', []):
                    yield {
                        'type': 'REAL_TIME_ALERT',
                        'data': alert,
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Reset batch
                batch = []