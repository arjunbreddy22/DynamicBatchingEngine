import torch
import time
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from request import Request


class LLMBackend:
    """
    Real LLM backend using HuggingFace transformers.
    Handles actual token generation instead of simulation.
    """
    
    def __init__(self, model_name="gpt2", device=None, max_length=50):
        """
        Initialize the LLM backend
        
        Args:
            model_name: HuggingFace model name (default: gpt2)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length for generation
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"🚀 Loading {model_name} on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use left padding for decoder-only models (avoids warning and is more efficient)
        self.tokenizer.padding_side = "left"
            
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
            
        self.model.eval()  # Set to evaluation mode
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model parameters: {self.model.num_parameters():,}")
        if self.device == "cuda":
            print(f"🔥 GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def generate_batch(self, requests: List[Request]) -> Dict[int, str]:
        """
        Generate tokens for a batch of requests
        
        Args:
            requests: List of Request objects with prompts
            
        Returns:
            Dictionary mapping request IDs to generated text
        """
        # CRITICAL IMPLEMENTATION NOTE:
        # This function serves as a simplified placeholder and does NOT represent
        # an efficient, real-world LLM generation pipeline. In its current form,
        # it re-processes the entire original prompt on every single call to
        # generate just one new token.
        #
        # Key limitations:
        # 1. No Persistent KV Cache: The KV cache is created and destroyed
        #    within each `.generate()` call. It is not reused across steps,
        #    which is the most important optimization in LLM serving.
        # 2. No Prefill/Decode Separation: This implementation is always in
        #    "prefill" mode. It never transitions to the efficient, single-token
        #    "decode" phase.
        #
        # A real implementation (like vLLM or TGI) would perform a single prefill
        # for each request, store the resulting KV cache, and then for all
        # subsequent steps, it would only feed the last generated token and the
        # past_key_values (the cache) back into the model. This code does not
        # do that, making it a good demo for scheduling but a poor demo for
        # efficient generation.
        if not requests:
            return {}
        
        # Prepare batch inputs
        prompts = []
        request_ids = []
        
        for req in requests:
            if hasattr(req, 'prompt') and req.prompt:
                # CRITICAL FIX: Use the full context (original prompt + generated text so far)
                # This ensures we continue generation from where we left off, not from the beginning
                full_context = req.prompt + getattr(req, 'generated_text', '')
                prompts.append(full_context)
                request_ids.append(req.id)
            else:
                # Generate a default prompt if none provided
                prompts.append("The quick brown fox")
                request_ids.append(req.id)
        
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate tokens
        with torch.no_grad():
            # Generate one token at a time to simulate streaming
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,  # Generate only 1 token per step
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode results
        results = {}
        generated_sequences = outputs.sequences
        
        for i, req_id in enumerate(request_ids):
            # Get the newly generated token(s)
            input_length = inputs.input_ids[i].shape[0]
            new_tokens = generated_sequences[i][input_length:]
            
            if len(new_tokens) > 0:
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                results[req_id] = generated_text
            else:
                results[req_id] = ""
        
        return results
    
    def estimate_memory_usage(self, batch_size: int, avg_seq_length: int = 20) -> float:
        """
        Estimate GPU memory usage for a given batch
        
        Args:
            batch_size: Number of sequences in batch
            avg_seq_length: Average sequence length
            
        Returns:
            Estimated memory usage in GB
        """
        # Rough estimation based on model size and sequence length
        model_memory = self.model.num_parameters() * 2 / 1024**3  # 2 bytes per param (float16)
        
        # KV cache memory: batch_size * seq_length * hidden_size * num_layers * 2 (K and V)
        hidden_size = self.model.config.hidden_size
        num_layers = self.model.config.num_hidden_layers
        
        kv_memory = (batch_size * avg_seq_length * hidden_size * num_layers * 2 * 2) / 1024**3
        
        return model_memory + kv_memory
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "num_parameters": self.model.num_parameters(),
            "vocab_size": self.tokenizer.vocab_size,
            "max_length": self.max_length,
            "memory_allocated": torch.cuda.memory_allocated() / 1024**3 if self.device == "cuda" else 0
        }


class RealRequest(Request):
    """Extended Request class for real LLM inference"""
    
    def __init__(self, request_id, arrival_time, tokens_needed, prompt="The quick brown fox"):
        super().__init__(request_id, arrival_time, tokens_needed)
        self.prompt = prompt
        self.generated_text = ""
        self.generation_history = []  # Track generated tokens
        
    def add_generated_token(self, token_text: str):
        """Add a newly generated token"""
        self.generated_text += token_text
        self.generation_history.append(token_text)
        
    def is_complete(self) -> bool:
        """Check if generation is complete"""
        return (self.tokens_left <= 0 or 
                self.generated_text.endswith('.') or 
                self.generated_text.endswith('!') or 
                self.generated_text.endswith('?') or
                len(self.generation_history) >= 20)  # Max tokens safety


def create_sample_requests(num_requests: int = 5) -> List[RealRequest]:
    """Create sample requests with different prompts for testing"""
    
    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In the future, artificial intelligence",
        "The best way to learn programming is",
        "Climate change is a serious issue that",
        "Machine learning algorithms can help",
        "The history of computers began with",
        "Space exploration has always been",
        "Renewable energy sources include",
        "The human brain is complex because"
    ]
    
    requests = []
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        tokens_needed = torch.randint(3, 8, (1,)).item()  # 3-7 tokens
        
        req = RealRequest(
            request_id=i,
            arrival_time=i * 2,  # Requests arrive every 2 steps
            tokens_needed=tokens_needed,
            prompt=prompt
        )
        requests.append(req)
    
    return requests


if __name__ == "__main__":
    # Test the LLM backend
    print("🧪 Testing LLM Backend...")
    
    # Initialize backend
    backend = LLMBackend("gpt2")
    
    # Create sample requests
    requests = create_sample_requests(3)
    
    print(f"\n📝 Sample requests:")
    for req in requests:
        print(f"  ID {req.id}: '{req.prompt}' (need {req.tokens_left} tokens)")
    
    # Test batch generation
    print(f"\n🔄 Generating tokens...")
    start_time = time.time()
    
    results = backend.generate_batch(requests)
    
    generation_time = time.time() - start_time
    
    print(f"\n✅ Generation completed in {generation_time:.3f} seconds")
    print(f"📊 Results:")
    for req_id, generated in results.items():
        print(f"  ID {req_id}: '{generated}'")
    
    # Print model info
    info = backend.get_model_info()
    print(f"\n🤖 Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}") 