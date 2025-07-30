import time
from typing import List, Dict, Any
from collections import deque

from llm_backend import LLMBackend, RealRequest


class RealNaiveBatcher:
    """Naive batching with real LLM backend"""
    
    def __init__(self, llm_backend: LLMBackend, window_size=20):
        self.llm_backend = llm_backend
        self.window_size = window_size
        self.requests = []
        self.current_finished = []
        self.current_batch = []
        self.last_batch_time = -window_size
        
        # Performance metrics
        self.total_generation_time = 0.0
        self.total_batches = 0

    def add_request(self, request: RealRequest):
        self.requests.append(request)

    def step(self, current_time):
        self.current_finished = []

        # Start a new batch every window_size steps
        if (current_time - self.last_batch_time) >= self.window_size and self.requests:
            self.current_batch = self.requests[:]
            self.requests = []
            self.last_batch_time = current_time

        # Process the current batch with real LLM
        if self.current_batch:
            start_time = time.time()
            
            # Generate one token for each request in the batch
            results = self.llm_backend.generate_batch(self.current_batch)
            
            generation_time = time.time() - start_time
            self.total_generation_time += generation_time
            self.total_batches += 1
            
            # Update requests with generated tokens
            for req in self.current_batch:
                if req.id in results:
                    generated_text = results[req.id]
                    req.add_generated_token(generated_text)
                    req.tokens_left -= 1

            # Check if all requests in batch are complete
            if all(req.is_complete() for req in self.current_batch):
                self.current_finished = self.current_batch[:]
                self.current_batch = []

    def collect_finished(self, current_time):
        for req in self.current_finished:
            req.finish_time = current_time
        return self.current_finished
    
    def get_performance_metrics(self):
        return {
            "total_generation_time": self.total_generation_time,
            "total_batches": self.total_batches,
            "avg_generation_time_per_batch": self.total_generation_time / max(1, self.total_batches)
        }


class RealIterativeBatcher:
    """Iterative batching with real LLM backend"""
    
    def __init__(self, llm_backend: LLMBackend):
        self.llm_backend = llm_backend
        self.requests = []
        self.current_finished = []
        
        # Performance metrics
        self.total_generation_time = 0.0
        self.total_batches = 0

    def add_request(self, request: RealRequest):
        self.requests.append(request)

    def step(self):
        self.current_finished = []
        
        if not self.requests:
            return
            
        start_time = time.time()
        
        # Generate tokens for all active requests
        results = self.llm_backend.generate_batch(self.requests)
        
        generation_time = time.time() - start_time
        self.total_generation_time += generation_time
        self.total_batches += 1
        
        # Update requests and remove completed ones
        remaining_requests = []
        
        for req in self.requests:
            if req.id in results:
                generated_text = results[req.id]
                req.add_generated_token(generated_text)
                req.tokens_left -= 1
            
            if req.is_complete():
                self.current_finished.append(req)
            else:
                remaining_requests.append(req)
        
        self.requests = remaining_requests

    def collect_finished(self, current_time):
        for req in self.current_finished:
            req.finish_time = current_time
        return self.current_finished
    
    def get_performance_metrics(self):
        return {
            "total_generation_time": self.total_generation_time,
            "total_batches": self.total_batches,
            "avg_generation_time_per_batch": self.total_generation_time / max(1, self.total_batches)
        }


class RealDynamicBatcher:
    """Dynamic batching with real LLM backend"""
    
    def __init__(self, llm_backend: LLMBackend, 
                 max_tokens_per_batch=512, 
                 max_seqs_per_batch=16,
                 max_wait_time=50,
                 gpu_memory_limit=1024):
        
        self.llm_backend = llm_backend
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_seqs_per_batch = max_seqs_per_batch
        self.max_wait_time = max_wait_time
        self.gpu_memory_limit = gpu_memory_limit
        
        # State
        self.waiting_queue = deque()
        self.active_batch = []
        self.current_finished = []
        
        # Metrics
        self.batch_sizes_history = []
        self.token_usage_history = []
        self.total_generation_time = 0.0
        self.total_batches = 0

    def add_request(self, request: RealRequest):
        request.queue_time = 0
        self.waiting_queue.append(request)

    def step(self, current_time):
        self.current_finished = []
        
        # Update waiting times for queued requests
        for req in self.waiting_queue:
            req.queue_time += 1
        
        # Process current active batch (if any)
        if self.active_batch:
            self._process_active_batch(current_time)
        
        # If no active batch, form new batch
        if not self.active_batch:
            self._form_new_batch(current_time)
        
        # Record metrics
        if self.active_batch:
            self.batch_sizes_history.append(len(self.active_batch))
            total_tokens = sum(req.tokens_left for req in self.active_batch)
            self.token_usage_history.append(total_tokens)

    def _process_active_batch(self, current_time):
        """Process one token step for the active batch using real LLM"""
        start_time = time.time()
        
        # Generate tokens for all requests in the batch
        results = self.llm_backend.generate_batch(self.active_batch)
        
        generation_time = time.time() - start_time
        self.total_generation_time += generation_time
        self.total_batches += 1
        
        # Update requests with generated tokens
        finished_this_step = []
        remaining_requests = []
        
        for req in self.active_batch:
            if req.id in results:
                generated_text = results[req.id]
                req.add_generated_token(generated_text)
                req.tokens_left -= 1
            
            if req.is_complete():
                req.finish_time = current_time
                finished_this_step.append(req)
            else:
                remaining_requests.append(req)
        
        self.active_batch = remaining_requests
        self.current_finished = finished_this_step

    def _form_new_batch(self, current_time):
        """Dynamic batch formation using real LLM memory constraints"""
        if not self.waiting_queue:
            return
        
        # Convert deque to list for easier manipulation
        candidates = list(self.waiting_queue)
        self.waiting_queue.clear()
        
        # Apply dynamic batching policy
        selected_batch = self._policy_select(candidates, current_time)
        
        # Remaining requests go back to queue
        remaining = [req for req in candidates if req not in selected_batch]
        self.waiting_queue.extend(remaining)
        
        # Set the active batch
        self.active_batch = selected_batch

    def _policy_select(self, candidates: List[RealRequest], current_time) -> List[RealRequest]:
        """Dynamic batching policy for real LLM"""
        if not candidates:
            return []
        
        # Sort candidates by priority
        def priority_score(req):
            urgency_score = req.queue_time / self.max_wait_time
            short_job_bonus = max(0, (10 - req.tokens_left) / 10)
            return urgency_score + short_job_bonus
        
        candidates.sort(key=priority_score, reverse=True)
        
        # Greedy selection with constraints
        selected = []
        total_tokens = 0
        total_seqs = 0
        
        for req in candidates:
            # Check constraints
            would_exceed_seqs = (total_seqs + 1) > self.max_seqs_per_batch
            would_exceed_tokens = (total_tokens + req.tokens_left) > self.max_tokens_per_batch
            
            # Estimate memory usage with real LLM
            estimated_memory = self._estimate_memory_usage(selected + [req])
            would_exceed_memory = estimated_memory > self.gpu_memory_limit
            
            # Force inclusion if request has waited too long
            force_include = req.queue_time >= self.max_wait_time
            
            if force_include or (not would_exceed_seqs and not would_exceed_tokens and not would_exceed_memory):
                selected.append(req)
                total_tokens += req.tokens_left
                total_seqs += 1
                
                # Early termination if we hit soft limit
                if total_seqs >= self.max_seqs_per_batch * 0.8:
                    break
        
        return selected

    def _estimate_memory_usage(self, requests: List[RealRequest]) -> float:
        """Estimate memory usage using real LLM backend"""
        if not requests:
            return 0.0
        
        # Use the LLM backend's memory estimation
        batch_size = len(requests)
        avg_seq_length = sum(len(req.prompt.split()) for req in requests) / batch_size
        
        return self.llm_backend.estimate_memory_usage(batch_size, int(avg_seq_length))

    def collect_finished(self, current_time):
        return self.current_finished

    def get_metrics(self):
        """Get comprehensive metrics including LLM performance"""
        base_metrics = {
            'avg_batch_size': sum(self.batch_sizes_history) / len(self.batch_sizes_history) if self.batch_sizes_history else 0,
            'max_batch_size': max(self.batch_sizes_history) if self.batch_sizes_history else 0,
            'avg_token_usage': sum(self.token_usage_history) / len(self.token_usage_history) if self.token_usage_history else 0,
            'total_batches': len(self.batch_sizes_history),
            'queue_length': len(self.waiting_queue),
            'active_requests': len(self.active_batch)
        }
        
        # Add LLM performance metrics
        llm_metrics = {
            'total_generation_time': self.total_generation_time,
            'llm_batches_processed': self.total_batches,
            'avg_generation_time_per_batch': self.total_generation_time / max(1, self.total_batches)
        }
        
        return {**base_metrics, **llm_metrics}


def create_real_sample_requests(num_requests: int = 10) -> List[RealRequest]:
    """Create sample requests with varied prompts for real LLM testing"""
    
    prompts = [
        "The quick brown fox jumps over",
        "Once upon a time in a land",
        "Artificial intelligence will revolutionize",
        "The best way to learn programming",
        "Climate change affects our planet",
        "Machine learning algorithms help",
        "The history of computers started",
        "Space exploration reveals new",
        "Renewable energy sources like solar",
        "The human brain processes information"
    ]
    
    requests = []
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        tokens_needed = 3 + (i % 5)  # 3-7 tokens needed
        
        req = RealRequest(
            request_id=i,
            arrival_time=i,  # Requests arrive every step
            tokens_needed=tokens_needed,
            prompt=prompt
        )
        requests.append(req)
    
    return requests


if __name__ == "__main__":
    # Test the real LLM batchers
    print("🧪 Testing Real LLM Batchers...")
    
    # Initialize LLM backend
    backend = LLMBackend("gpt2")
    
    # Create sample requests
    requests = create_real_sample_requests(5)
    
    print(f"\n📝 Sample requests:")
    for req in requests:
        print(f"  ID {req.id}: '{req.prompt}' (need {req.tokens_left} tokens)")
    
    # Test Real Iterative Batcher
    print(f"\n🚀 Testing Real Iterative Batcher...")
    iterative_batcher = RealIterativeBatcher(backend)
    
    for req in requests:
        iterative_batcher.add_request(req)
    
    step = 0
    while iterative_batcher.requests and step < 10:
        iterative_batcher.step()
        finished = iterative_batcher.collect_finished(step)
        
        if finished:
            print(f"  Step {step}: {len(finished)} requests completed")
            for req in finished:
                print(f"    ID {req.id}: '{req.generated_text}'")
        
        step += 1
    
    metrics = iterative_batcher.get_performance_metrics()
    print(f"\n📊 Performance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
    
    print(f"\n✅ Real LLM Batcher test completed!") 