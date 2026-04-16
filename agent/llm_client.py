import json
from groq import Groq

class LLMClient:
    """
    Real LLM Client using Groq API.
    This class handles the communication with Llama-3 models to transform 
    natural language into structured tool calls (JSON).
    """

    def __init__(self):
        # Replace with your actual Groq API Key
        import os
	from dotenv import load_dotenv
	load_dotenv()
	self.api_key = os.getenv("GROQ_API_KEY") 
        self.client = Groq(api_key=self.api_key)
        
        # Using Llama-3.3 70B for high-reasoning capabilities
        self.model = "llama-3.1-8b-instant"
        
    def chat(self, messages):
        """
        Sends the conversation history to Groq and receives a structured JSON response.
        """
        print(f"\n--- CALLING GROQ AI (Reasoning Engine: llama-3.1-8b) ---")
        
        try:
            # API Call to Groq - We use the model Instant to avoid errors 429
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            response_content = completion.choices[0].message.content
            #print(f"DEBUG GROQ RESPONSE: {response_content}")
            return response_content

        except Exception as e:
            # If the API fails, it gives back a JSON that doesn't crash the main
            print(f"\n[SYSTEM ERROR]: API Groq not available: {e}")
            
            import json
            safe_error = {
                "thought": "I cannot process the request right now due to API limits.",
                "tool": "error_handler", 
                "args": {}
            }
            return json.dumps(safe_error)
    
    def chat_text(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 500) -> str:
        """
        Returns plain text (no JSON enforcement).
        Used for report commentary.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            text = completion.choices[0].message.content or ""
            return text.strip()

        except Exception as e:
            # Keep demo clean: return a short, non-blocking message
            return f"- AI commentary unavailable (Groq error: {str(e)})"
