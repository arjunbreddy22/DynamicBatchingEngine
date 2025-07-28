import time
from collections import deque
from typing import List, Optional

class DynamicBatcher:
    def __init__(self, 
                 max_tokens_per_batch=512, 
                 max_seqs_per_batch=16,
                 max_wait_time=50,  # max steps a request can wait
                 gpu_memory_limit=1024):  # simulated memory limit
        
        # Configuration
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_seqs_per_batch = max_seqs_per_batch
        self.max_wait_time = max_wait_time
        self.gpu_memory_limit = gpu_memory_limit
        
        # State
        self.waiting_queue = deque()  # requests waiting to be processed
        self.active_batch = []        # requests currently being processed
        self.current_finished = []    # requests finished this step
        
        # Metrics
        self.batch_sizes_history = []
        self.token_usage_history = []
        
    def add_request(self, request):
        """Add a new request to the waiting queue"""
        request.queue_time = 0  # track how long it's been waiting
        self.waiting_queue.append(request)
    
    def step(self, current_time):
        """Main scheduling loop - called every time step"""
        self.current_finished = []
        
        # 1. Update waiting times for queued requests
        for req in self.waiting_queue:
            req.queue_time += 1
        
        # 2. Process current active batch (if any)
        if self.active_batch:
            self._process_active_batch(current_time)
        
        # 3. If no active batch or batch finished, form new batch
        if not self.active_batch:
            self._form_new_batch(current_time)
        
        # 4. Record metrics
        if self.active_batch:
            self.batch_sizes_history.append(len(self.active_batch))
            total_tokens = sum(req.tokens_left for req in self.active_batch)
            self.token_usage_history.append(total_tokens)
    
    def _process_active_batch(self, current_time):
        """Process one token step for the active batch"""
        # Generate one token for each request in the batch
        for req in self.active_batch:
            req.tokens_left -= 1
        
        # Remove finished requests
        finished_this_step = []
        remaining_requests = []
        
        for req in self.active_batch:
            if req.tokens_left <= 0:
                req.finish_time = current_time
                finished_this_step.append(req)
            else:
                remaining_requests.append(req)
        
        self.active_batch = remaining_requests
        self.current_finished = finished_this_step
    
    def _form_new_batch(self, current_time):
        """Dynamic batch formation"""
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
    
    def _policy_select(self, candidates: List, current_time) -> List:
        """
        Dynamic batching policy - decides which requests to batch together
        This is the core intelligence of the dynamic batcher
        """
        if not candidates:
            return []
        
        # Sort candidates by priority (multiple criteria)
        def priority_score(req):
            # Higher score = higher priority
            urgency_score = req.queue_time / self.max_wait_time  # 0 to 1+
            short_job_bonus = max(0, (10 - req.tokens_left) / 10)  # favor short jobs
            return urgency_score + short_job_bonus
        
        candidates.sort(key=priority_score, reverse=True)
        
        # Greedy selection with multiple constraints
        selected = []
        total_tokens = 0
        total_seqs = 0
        
        for req in candidates:
            # Check if adding this request would violate constraints
            would_exceed_seqs = (total_seqs + 1) > self.max_seqs_per_batch
            would_exceed_tokens = (total_tokens + req.tokens_left) > self.max_tokens_per_batch
            would_exceed_memory = self._estimate_memory_usage(selected + [req]) > self.gpu_memory_limit
            
            # Force inclusion if request has waited too long (SLA protection)
            force_include = req.queue_time >= self.max_wait_time
            
            if force_include or (not would_exceed_seqs and not would_exceed_tokens and not would_exceed_memory):
                # _policy_select
                if force_include:
                    # *still* check memory; if it breaks, chunk the request
                    if would_exceed_memory:
                        req.tokens_left = min(req.tokens_left, self.max_tokens_per_batch - total_tokens)
                    selected.append(req)
                    continue

                selected.append(req)
                total_tokens += req.tokens_left
                total_seqs += 1
                
                # Early termination if we hit a soft limit
                if total_seqs >= self.max_seqs_per_batch * 0.8:  # 80% of max
                    break
        
        return selected
    
    def _estimate_memory_usage(self, requests) -> int:
        """Simulate GPU memory usage estimation"""
        # Simple heuristic: each token uses some memory for KV cache
        total_tokens = sum(req.tokens_left for req in requests)
        base_memory = len(requests) * 10  # base memory per sequence
        kv_memory = total_tokens * 2      # memory per token
        return base_memory + kv_memory
    
    def collect_finished(self, current_time):
        """Return finished requests (called by simulation)"""
        return self.current_finished
    
    def get_metrics(self):
        """Get batching metrics for analysis"""
        if not self.batch_sizes_history:
            return {}
        
        return {
            'avg_batch_size': sum(self.batch_sizes_history) / len(self.batch_sizes_history),
            'max_batch_size': max(self.batch_sizes_history),
            'avg_token_usage': sum(self.token_usage_history) / len(self.token_usage_history),
            'total_batches': len(self.batch_sizes_history),
            'queue_length': len(self.waiting_queue),
            'active_requests': len(self.active_batch)
        }