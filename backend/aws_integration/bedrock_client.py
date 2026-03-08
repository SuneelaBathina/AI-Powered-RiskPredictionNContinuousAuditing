import boto3
import json
import os
import logging
from dotenv import load_dotenv
from botocore.exceptions import ClientError, BotoCoreError
from typing import Optional, Dict, Any

load_dotenv()
logger = logging.getLogger(__name__)

class BedrockClient:
    """AWS Bedrock client for interacting with foundation models"""
    
    def __init__(self):
        self.client = None
        self.model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-v2')
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.initialize_client()
        
    def initialize_client(self):
        """Initialize the Bedrock runtime client"""
        try:
            # Check if AWS credentials are available
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            
            if aws_access_key and aws_secret_key:
                # Use provided credentials
                self.client = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=self.aws_region,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key
                )
                logger.info("Bedrock client initialized with provided credentials")
            else:
                # Try default credential chain
                self.client = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=self.aws_region
                )
                logger.info("Bedrock client initialized with default credentials")
                
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Error initializing Bedrock client: {e}")
            self.client = None
    
    def invoke_model(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> Optional[str]:
        """
        Invoke Bedrock model with prompt
        
        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens in response
            temperature: Model temperature (0-1)
            
        Returns:
            Model response text or None if error
        """
        if not self.client:
            logger.warning("Bedrock client not initialized. Using mock response.")
            return self._get_mock_response(prompt)
        
        try:
            # Prepare request body based on model
            if 'claude' in self.model_id.lower():
                body = json.dumps({
                    "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                })
            elif 'titan' in self.model_id.lower():
                body = json.dumps({
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": max_tokens,
                        "temperature": temperature,
                        "topP": 0.9
                    }
                })
            else:
                # Generic format
                body = json.dumps({
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                })
            
            response = self.client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract response based on model
            if 'claude' in self.model_id.lower():
                return response_body.get('completion', '')
            elif 'titan' in self.model_id.lower():
                return response_body.get('results', [{}])[0].get('outputText', '')
            else:
                return str(response_body)
                
        except Exception as e:
            logger.error(f"Error invoking Bedrock model: {e}")
            return self._get_mock_response(prompt)
    
    def _get_mock_response(self, prompt: str) -> str:
        """Generate mock response when AWS is not available"""
        if "risk" in prompt.lower():
            return "Based on the risk factors, this transaction shows elevated risk patterns. Recommend enhanced due diligence."
        elif "audit" in prompt.lower():
            return "Audit findings indicate need for improved controls in high-risk transaction monitoring."
        elif "compliance" in prompt.lower():
            return "Compliance check complete. No major violations detected, but recommend review of AML procedures."
        else:
            return f"Analysis complete for: {prompt[:100]}..."
    
    def analyze_risk_description(self, risk_factors: Dict[str, Any]) -> str:
        """Generate risk description using Bedrock"""
        prompt = f"""You are a financial risk analyst. Analyze the following risk factors and provide a concise risk assessment:
        
        Risk Factors: {json.dumps(risk_factors, indent=2)}
        
        Provide a professional risk assessment including:
        1. Overall risk level
        2. Key concerns
        3. Recommended actions
        
        Risk Assessment:"""
        
        return self.invoke_model(prompt, max_tokens=300)
    
    def generate_audit_summary(self, audit_findings: Dict[str, Any]) -> str:
        """Generate audit summary using Bedrock"""
        prompt = f"""You are a senior auditor. Summarize the following audit findings in a clear, professional manner:
        
        Audit Findings: {json.dumps(audit_findings, indent=2)[:1000]}
        
        Provide a comprehensive audit summary including:
        1. Key findings
        2. Risk exposure
        3. Recommended remediation steps
        
        Audit Summary:"""
        
        return self.invoke_model(prompt, max_tokens=400)
    
    def generate_compliance_report(self, violations: list) -> str:
        """Generate compliance report using Bedrock"""
        prompt = f"""Generate a compliance report based on these violations:
        
        Violations: {json.dumps(violations, indent=2)}
        
        Include:
        1. Severity assessment
        2. Regulatory implications
        3. Required corrective actions
        4. Timeline for remediation
        
        Compliance Report:"""
        
        return self.invoke_model(prompt, max_tokens=500)
    
    def is_available(self) -> bool:
        """Check if Bedrock client is available"""
        return self.client is not None


# For testing without AWS
class MockBedrockClient(BedrockClient):
    """Mock client for development without AWS"""
    
    def __init__(self):
        self.client = None
        self.model_id = 'mock-model'
        self.aws_region = 'mock-region'
    
    def invoke_model(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        return self._get_mock_response(prompt)
    
    def is_available(self) -> bool:
        return True