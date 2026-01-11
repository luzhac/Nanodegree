from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""
    all_messages = []
    # TODO: Define system prompt
    system_prompt = context
    
    # TODO: Set context in messages
    all_messages = all_messages+[{"role": "system", "content": system_prompt}]
     
    
    # TODO: Add chat history
    all_messages = all_messages + conversation_history
    all_messages = all_messages + [{"role": "user", "content": user_message}]
    # TODO: Create OpenAI Client
    client = OpenAI(api_key=openai_key)
    # TODO: Send request to OpenAI
    resp = client.chat.completions.create(
        model=model,
        messages=all_messages
    )

    # TODO: Return response

    return resp.choices[0].message.content