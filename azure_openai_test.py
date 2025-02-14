import os
from openai import AzureOpenAI
from typing import Dict, Any
import json

def test_azure_openai_connection(config: Dict[str, Any]) -> tuple[bool, str]:
    """
    Test connection to Azure OpenAI service
    Returns tuple of (success: bool, message: str)
    """
    try:
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=config["azure_openai_key"],
            api_version="2023-05-15",
            azure_endpoint=config["azure_openai_endpoint"]
        )
        
        # Simple test prompt
        response = client.chat.completions.create(
            model=config["azure_deployment_name"],  # Use model instead of deployment_name
            messages=[
                {"role": "user", "content": "Hello, can you confirm this is working?"}
            ],
            max_tokens=30
        )
        
        if response and response.choices:
            return True, f"Successfully connected to Azure OpenAI service! Response: {response.choices[0].message.content}"
        else:
            return False, "Connection successful but no response received."
            
    except Exception as e:
        return False, f"Failed to connect to Azure OpenAI: {str(e)}"

def main():
    # Test configuration
    config = {
        "azure_openai_endpoint": "https://saura-m74c37z2-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-08-01-preview",
        "azure_openai_key": "7xRZ9LBwI7AMqAQ9X5ep3FKUjuhVmRtfV95JNhCNLE60YDZ8wZHRJQQJ99BBACHYHv6XJ3w3AAAAACOGmsC9",
        "azure_deployment_name": "gpt-35-turbo"
    }
    
    print("Testing Azure OpenAI Connection...")
    success, message = test_azure_openai_connection(config)
    
    if success:
        print("✅ " + message)
    else:
        print("❌ " + message)

if __name__ == "__main__":
    main() 