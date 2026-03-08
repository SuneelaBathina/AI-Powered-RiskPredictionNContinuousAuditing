from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
from datetime import datetime
from langchain.llms import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, bedrock_client, vector_store):
        self.name = name
        self.bedrock = bedrock_client
        self.vector_store = vector_store
        self.memory = []
        self.metrics = {
            'tasks_completed': 0,
            'avg_response_time': 0,
            'success_rate': 1.0
        }
        
    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and return updated state"""
        pass
    
    def query_knowledge_base(self, query: str, k: int = 3) -> List[Dict]:
        """Query the RAG knowledge base"""
        return self.vector_store.similarity_search(query, k=k)
    
    def invoke_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Invoke LLM through Bedrock"""
        try:
            response = self.bedrock.invoke_model(prompt, max_tokens)
            return response
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            return ""
    
    def update_memory(self, key: str, value: Any):
        """Update agent memory"""
        self.memory.append({
            'timestamp': datetime.now().isoformat(),
            'key': key,
            'value': value
        })
        # Keep only last 100 memories
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]
    
    def get_context(self, state: Dict[str, Any]) -> str:
        """Get relevant context from memory and knowledge base"""
        context = f"Agent: {self.name}\n"
        context += f"Current state: {state.get('current_phase', 'unknown')}\n"
        
        # Add recent memories
        if self.memory:
            context += "\nRecent memories:\n"
            for mem in self.memory[-5:]:
                context += f"- {mem['key']}: {mem['value']}\n"
        
        return context
    
    def log_activity(self, activity: str, level: str = "info"):
        """Log agent activity"""
        log_msg = f"[{self.name}] {activity}"
        if level == "info":
            logger.info(log_msg)
        elif level == "warning":
            logger.warning(log_msg)
        elif level == "error":
            logger.error(log_msg)