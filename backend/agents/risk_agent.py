from typing import Dict, Any, List
import json
from .base_agent import BaseAgent

class RiskAssessmentAgent(BaseAgent):
    """Agent responsible for assessing transaction risks"""
    
    def __init__(self, bedrock_client, vector_store, risk_predictor):
        super().__init__("RiskAssessmentAgent", bedrock_client, vector_store)
        self.risk_predictor = risk_predictor
        
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_activity("Starting risk assessment")
        
        transactions = state.get('transactions', [])
        if not transactions:
            self.log_activity("No transactions to assess", "warning")
            return state
        
        assessed_transactions = []
        high_risk_alerts = []
        
        for transaction in transactions:
            # Get ML-based risk score
            risk_result = self.risk_predictor.predict(transaction)
            
            # Enhance with contextual analysis
            context_query = f"Risk factors for transaction type {transaction.get('transaction_type', 'unknown')} amount ${transaction.get('amount', 0)}"
            context_docs = self.query_knowledge_base(context_query)
            
            # Generate risk explanation using LLM
            explanation_prompt = f"""
            Analyze this transaction for risk:
            Transaction: {json.dumps(transaction)}
            ML Risk Score: {risk_result['risk_score']}
            Risk Level: {risk_result['risk_level']}
            
            Context from knowledge base: {context_docs}
            
            Provide a detailed risk assessment including:
            1. Key risk factors
            2. Suspicious patterns
            3. Recommended actions
            """
            
            risk_explanation = self.invoke_llm(explanation_prompt)
            
            assessed_txn = {
                **transaction,
                'risk_score': risk_result['risk_score'],
                'risk_level': risk_result['risk_level'],
                'confidence': risk_result['confidence'],
                'risk_explanation': risk_explanation,
                'assessed_at': state.get('timestamp')
            }
            
            assessed_transactions.append(assessed_txn)
            
            # Generate alert for high-risk transactions
            if risk_result['risk_level'] == 'HIGH':
                high_risk_alerts.append({
                    'transaction_id': transaction.get('transaction_id'),
                    'risk_score': risk_result['risk_score'],
                    'amount': transaction.get('amount'),
                    'alert_level': 'HIGH',
                    'requires_immediate_review': True
                })
        
        # Update state
        state['assessed_transactions'] = assessed_transactions
        state['high_risk_alerts'] = high_risk_alerts
        state['risk_metrics'] = self._calculate_risk_metrics(assessed_transactions)
        
        self.update_memory('last_assessment_count', len(assessed_transactions))
        self.update_memory('high_risk_count', len(high_risk_alerts))
        
        self.log_activity(f"Completed risk assessment. Found {len(high_risk_alerts)} high-risk transactions")
        
        return state
    
    def _calculate_risk_metrics(self, transactions: List[Dict]) -> Dict:
        """Calculate aggregate risk metrics"""
        if not transactions:
            return {}
        
        risk_scores = [t.get('risk_score', 0) for t in transactions]
        
        return {
            'avg_risk_score': sum(risk_scores) / len(risk_scores),
            'max_risk_score': max(risk_scores),
            'min_risk_score': min(risk_scores),
            'high_risk_count': sum(1 for t in transactions if t.get('risk_level') == 'HIGH'),
            'medium_risk_count': sum(1 for t in transactions if t.get('risk_level') == 'MEDIUM'),
            'low_risk_count': sum(1 for t in transactions if t.get('risk_level') == 'LOW'),
            'total_value_at_risk': sum(
                t.get('amount', 0) for t in transactions 
                if t.get('risk_level') in ['HIGH', 'MEDIUM']
            )
        }