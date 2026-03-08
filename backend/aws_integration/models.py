"""AWS model configurations and response parsers"""

from typing import Dict, Any, Optional
import json

class BedrockModelConfig:
    """Configuration for different Bedrock models"""
    
    MODELS = {
        'claude': {
            'id': 'anthropic.claude-v2',
            'max_tokens': 100000,
            'supports_streaming': True,
            'input_format': 'prompt',
            'response_path': ['completion']
        },
        'claude-instant': {
            'id': 'anthropic.claude-instant-v1',
            'max_tokens': 100000,
            'supports_streaming': True,
            'input_format': 'prompt',
            'response_path': ['completion']
        },
        'titan-text': {
            'id': 'amazon.titan-text-express-v1',
            'max_tokens': 8000,
            'supports_streaming': False,
            'input_format': 'inputText',
            'response_path': ['results', 0, 'outputText']
        },
        'jurassic': {
            'id': 'ai21.j2-ultra-v1',
            'max_tokens': 8000,
            'supports_streaming': False,
            'input_format': 'prompt',
            'response_path': ['completions', 0, 'data', 'text']
        }
    }
    
    @classmethod
    def get_model_config(cls, model_key: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return cls.MODELS.get(model_key, cls.MODELS['claude'])
    
    @classmethod
    def format_request(cls, model_key: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Format request for specific model"""
        config = cls.get_model_config(model_key)
        
        if model_key.startswith('claude'):
            return {
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": kwargs.get('max_tokens', 500),
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 0.9),
                "stop_sequences": kwargs.get('stop_sequences', ["\n\nHuman:"])
            }
        elif model_key.startswith('titan'):
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": kwargs.get('max_tokens', 500),
                    "temperature": kwargs.get('temperature', 0.7),
                    "topP": kwargs.get('top_p', 0.9),
                    "stopSequences": kwargs.get('stop_sequences', [])
                }
            }
        else:
            return {
                "prompt": prompt,
                "maxTokens": kwargs.get('max_tokens', 500),
                "temperature": kwargs.get('temperature', 0.7),
                "topP": kwargs.get('top_p', 0.9)
            }
    
    @classmethod
    def parse_response(cls, model_key: str, response_body: Dict) -> str:
        """Parse response from specific model"""
        config = cls.get_model_config(model_key)
        
        try:
            if model_key.startswith('claude'):
                return response_body.get('completion', '')
            elif model_key.startswith('titan'):
                results = response_body.get('results', [])
                if results:
                    return results[0].get('outputText', '')
            elif model_key.startswith('jurassic'):
                completions = response_body.get('completions', [])
                if completions:
                    return completions[0].get('data', {}).get('text', '')
            else:
                return str(response_body)
        except Exception as e:
            return f"Error parsing response: {e}"
        
        return ""


class BedrockResponseParser:
    """Parse responses from different Bedrock models"""
    
    @staticmethod
    def parse_claude_response(response: Dict) -> str:
        """Parse Claude model response"""
        return response.get('completion', '')
    
    @staticmethod
    def parse_titan_response(response: Dict) -> str:
        """Parse Titan model response"""
        results = response.get('results', [])
        if results:
            return results[0].get('outputText', '')
        return ''
    
    @staticmethod
    def parse_jurassic_response(response: Dict) -> str:
        """Parse Jurassic model response"""
        completions = response.get('completions', [])
        if completions:
            return completions[0].get('data', {}).get('text', '')
        return ''
    
    @staticmethod
    def parse_generic_response(response: Dict) -> str:
        """Parse generic response"""
        return json.dumps(response)


class BedrockErrorHandler:
    """Handle Bedrock API errors"""
    
    ERROR_MESSAGES = {
        'AccessDeniedException': 'AWS credentials do not have permission to access Bedrock',
        'ValidationException': 'Invalid request parameters',
        'ResourceNotFoundException': 'The specified model or resource was not found',
        'ThrottlingException': 'Request throttled. Please retry with backoff',
        'ServiceUnavailable': 'Bedrock service is temporarily unavailable',
        'ModelTimeoutException': 'Model response timed out',
        'ModelErrorException': 'Model encountered an error processing the request'
    }
    
    @classmethod
    def get_error_message(cls, error_code: str) -> str:
        """Get user-friendly error message"""
        return cls.ERROR_MESSAGES.get(error_code, f'Unknown error: {error_code}')
    
    @classmethod
    def should_retry(cls, error_code: str) -> bool:
        """Determine if operation should be retried"""
        retryable_errors = ['ThrottlingException', 'ServiceUnavailable', 'ModelTimeoutException']
        return error_code in retryable_errors