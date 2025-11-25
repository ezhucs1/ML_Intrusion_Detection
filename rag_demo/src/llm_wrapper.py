"""
LLM Wrapper for Phi-3 Model
Supports both Ollama (recommended) and Transformers
"""

import os
from typing import List, Optional

class LLMWrapper:
    """Wrapper for Phi-3 LLM using Ollama or Transformers"""
    
    def __init__(self, model_name: str = "phi3:mini", use_ollama: bool = True):
        """
        Initialize LLM wrapper
        
        Args:
            model_name: Model name (for Ollama: "phi3:mini", for transformers: "microsoft/Phi-3-mini-4k-instruct")
            use_ollama: If True, use Ollama (recommended for demo). If False, use transformers directly
        """
        self.model_name = model_name
        self.use_ollama = use_ollama
        
        if use_ollama:
            try:
                import ollama
                self.ollama_client = ollama
                print(f"Using Ollama with model: {model_name}")
                # Check if model is available
                try:
                    models = self.ollama_client.list()
                    model_names = [m['name'] for m in models.get('models', [])]
                    if model_name not in model_names:
                        print(f"⚠️  Model {model_name} not found. Please run: ollama pull {model_name}")
                        print("   Attempting to use model anyway...")
                except Exception as e:
                    print(f"⚠️  Could not check for model: {e}")
                    print("   Attempting to use model anyway...")
            except ImportError:
                print("⚠️  Ollama not installed. Install with: pip install ollama")
                print("   Falling back to transformers...")
                self.use_ollama = False
        
        if not self.use_ollama:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                print(f"Loading {model_name} with transformers...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                print("✓ Model loaded")
            except ImportError:
                raise ImportError("Please install transformers: pip install transformers torch accelerate")
            except Exception as e:
                print(f"⚠️  Error loading model: {e}")
                print("   Consider using Ollama instead: https://ollama.ai")
                raise
    
    def generate(self, prompt: str, context: Optional[str] = None, 
                max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: User query
            context: Retrieved context from RAG
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        # Build full prompt with context
        if context:
            full_prompt = f"""You are an AI assistant helping with network intrusion detection analysis.

Context from dataset:
{context}

User Question: {prompt}

Please provide a helpful answer based on the context above. If the context doesn't contain enough information, say so."""
        else:
            full_prompt = f"""You are an AI assistant helping with network intrusion detection analysis.

User Question: {prompt}

Please provide a helpful answer."""
        
        if self.use_ollama:
            try:
                response = self.ollama_client.generate(
                    model=self.model_name,
                    prompt=full_prompt,
                    options={
                        'temperature': temperature,
                        'num_predict': max_tokens
                    }
                )
                return response['response']
            except Exception as e:
                return f"Error generating response: {e}\n\nPlease ensure Ollama is running and the model is installed:\n  ollama pull {self.model_name}"
        else:
            # Use transformers
            import torch
            inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from response
            response = response[len(full_prompt):].strip()
            return response

if __name__ == "__main__":
    # Test LLM
    print("Testing LLM wrapper...")
    try:
        llm = LLMWrapper(use_ollama=True)
        response = llm.generate("What is network intrusion detection?")
        print("\nResponse:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")

