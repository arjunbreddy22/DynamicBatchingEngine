#!/usr/bin/env python3
"""
Dynamic Batching Engine Demonstration

This script runs a comprehensive demonstration of all three batching strategies
and highlights the key improvements of dynamic batching over traditional approaches.
"""

import sys
import time
from simulation import run_simulation, print_stats
from naive_batching import NaiveBatcher
from iterative_batching import IterativeBatcher
from dynamic_batching import DynamicBatcher


def print_header(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def print_section(title):
    """Print a formatted subsection header"""
    print(f"\n{title}")
    print("-" * len(title))


def demonstrate_batching_strategies():
    """Demonstrate and compare all three batching strategies"""
    
    print_header("DYNAMIC BATCHING ENGINE DEMONSTRATION")
    
    print("""
This demonstration compares three LLM inference batching strategies:

1. 🐌 NAIVE BATCHING: Fixed time windows (traditional approach)
   - Collects requests for fixed intervals (20 steps)
   - Processes entire batch together until all complete
   - Simple but high latency, especially for short requests

2. ⚡ ITERATIVE BATCHING: Token-by-token processing
   - Processes all active requests every step
   - Returns requests immediately when complete
   - Low latency but no resource constraints

3. 🎯 DYNAMIC BATCHING: Our adaptive approach
   - Adapts batch size based on queue, memory, and SLA constraints
   - Prioritizes urgent and short requests
   - Balances latency and throughput optimally
    """)
    
    # Simulation parameters
    SIM_DURATION = 1000
    MIN_ARRIVAL = 1
    MAX_ARRIVAL = 5
    MAX_TOKENS = 10
    
    print_section("SIMULATION PARAMETERS")
    print(f"Duration:           {SIM_DURATION} time steps")
    print(f"Request arrival:    Every {MIN_ARRIVAL}-{MAX_ARRIVAL} steps (random)")
    print(f"Request length:     1-{MAX_TOKENS} tokens (random)")
    print(f"Naive window size:  20 steps")
    print(f"Dynamic constraints:")
    print(f"  - Max batch size:    8 requests")
    print(f"  - Max tokens/batch:  100 tokens")
    print(f"  - Max wait time:     30 steps (SLA)")
    print(f"  - Memory limit:      500 units")
    
    print_section("RUNNING SIMULATIONS")
    
    # Initialize batching strategies
    naive_batcher = NaiveBatcher(window_size=20)
    iterative_batcher = IterativeBatcher()
    dynamic_batcher = DynamicBatcher(
        max_tokens_per_batch=100,
        max_seqs_per_batch=8,
        max_wait_time=30,
        gpu_memory_limit=500
    )
    
    print("⏳ Running naive batching simulation...")
    start_time = time.time()
    naive_results = run_simulation(naive_batcher, SIM_DURATION, MIN_ARRIVAL, MAX_ARRIVAL, MAX_TOKENS)
    naive_time = time.time() - start_time
    
    print("⏳ Running iterative batching simulation...")
    start_time = time.time()
    iterative_results = run_simulation(iterative_batcher, SIM_DURATION, MIN_ARRIVAL, MAX_ARRIVAL, MAX_TOKENS)
    iterative_time = time.time() - start_time
    
    print("⏳ Running dynamic batching simulation...")
    start_time = time.time()
    dynamic_results = run_simulation(dynamic_batcher, SIM_DURATION, MIN_ARRIVAL, MAX_ARRIVAL, MAX_TOKENS)
    dynamic_time = time.time() - start_time
    
    print_section("PERFORMANCE RESULTS")
    
    # Print results for each strategy
    print_stats(naive_results, "🐌 Naive")
    print_stats(iterative_results, "⚡ Iterative") 
    print_stats(dynamic_results, "🎯 Dynamic")
    
    # Calculate and display improvements
    if naive_results and iterative_results and dynamic_results:
        naive_avg = sum(req.finish_time - req.arrival_time for req in naive_results) / len(naive_results)
        iter_avg = sum(req.finish_time - req.arrival_time for req in iterative_results) / len(iterative_results)
        dyn_avg = sum(req.finish_time - req.arrival_time for req in dynamic_results) / len(dynamic_results)
        
        naive_p95 = sorted([req.finish_time - req.arrival_time for req in naive_results])[int(0.95 * len(naive_results))]
        iter_p95 = sorted([req.finish_time - req.arrival_time for req in iterative_results])[int(0.95 * len(iterative_results))]
        dyn_p95 = sorted([req.finish_time - req.arrival_time for req in dynamic_results])[int(0.95 * len(dynamic_results))]
        
        print_section("🚀 KEY IMPROVEMENTS")
        print(f"Dynamic vs Naive Batching:")
        print(f"  • Average latency:    {((naive_avg - dyn_avg) / naive_avg * 100):+6.1f}% improvement")
        print(f"  • 95th percentile:    {((naive_p95 - dyn_p95) / naive_p95 * 100):+6.1f}% improvement")
        print(f"  • Requests served:    {len(dynamic_results) - len(naive_results):+4d} more requests")
        
        print(f"\nDynamic vs Iterative Batching:")
        print(f"  • Average latency:    {((iter_avg - dyn_avg) / iter_avg * 100):+6.1f}% improvement")
        print(f"  • 95th percentile:    {((iter_p95 - dyn_p95) / iter_p95 * 100):+6.1f}% improvement")
        print(f"  • Requests served:    {len(dynamic_results) - len(iterative_results):+4d} more requests")
    
    # Display dynamic batcher specific metrics
    metrics = dynamic_batcher.get_metrics()
    if metrics:
        print_section("🎯 DYNAMIC BATCHER INSIGHTS")
        print(f"Average batch size:     {metrics['avg_batch_size']:.2f} requests")
        print(f"Maximum batch size:     {metrics['max_batch_size']} requests")
        print(f"Average token usage:    {metrics['avg_token_usage']:.1f} tokens/batch")
        print(f"Total batches created:  {metrics['total_batches']}")
        print(f"Batch efficiency:       {(metrics['avg_batch_size'] / metrics['max_batch_size'] * 100):.1f}% of max capacity")
        
        if metrics['queue_length'] > 0:
            print(f"⚠️  Final queue length:    {metrics['queue_length']} (requests still waiting)")
        else:
            print("✅ All requests processed successfully")
    
    print_section("⏱️  SIMULATION RUNTIME")
    print(f"Naive batching:     {naive_time:.3f} seconds")
    print(f"Iterative batching: {iterative_time:.3f} seconds")
    print(f"Dynamic batching:   {dynamic_time:.3f} seconds")


def demonstrate_edge_cases():
    """Demonstrate how dynamic batching handles edge cases"""
    
    print_header("EDGE CASE DEMONSTRATIONS")
    
    print("""
Let's see how dynamic batching handles challenging scenarios:
""")
    
    print_section("1. 🚨 SLA PROTECTION TEST")
    print("Scenario: Mix of short and long requests to test timeout protection")
    
    # Create a scenario with mixed request lengths
    from request import Request
    
    batcher = DynamicBatcher(
        max_tokens_per_batch=20,
        max_seqs_per_batch=4,
        max_wait_time=5,  # Very short SLA
        gpu_memory_limit=100
    )
    
    # Add requests with different lengths
    short_req = Request(1, 0, 1)
    long_req = Request(2, 0, 10)
    medium_req = Request(3, 0, 5)
    
    batcher.add_request(short_req)
    batcher.add_request(long_req)
    batcher.add_request(medium_req)
    
    # Simulate some waiting time
    long_req.queue_time = 6  # Exceeds SLA
    
    batcher.step(0)
    
    batch_ids = [req.id for req in batcher.active_batch]
    
    print(f"Requests in batch: {batch_ids}")
    if 2 in batch_ids:  # long_req was included despite size
        print("✅ SLA protection worked: Long request included despite token limit")
    else:
        print("❌ SLA protection failed")
    
    print_section("2. 📊 RESOURCE CONSTRAINT TEST")
    print("Scenario: Many small requests vs few large requests")
    
    # Test batch size limits
    batcher2 = DynamicBatcher(max_seqs_per_batch=3)
    
    for i in range(8):
        req = Request(i, 0, 1)
        batcher2.add_request(req)
    
    batcher2.step(0)
    
    print(f"Batch size limit: 3 requests")
    print(f"Active batch size: {len(batcher2.active_batch)}")
    print(f"Waiting queue: {len(batcher2.waiting_queue)}")
    
    if len(batcher2.active_batch) == 3:
        print("✅ Batch size constraint respected")
    else:
        print("❌ Batch size constraint violated")


def main():
    """Main demonstration function"""
    
    try:
        # Run main demonstration
        demonstrate_batching_strategies()
        
        # Run edge case tests
        demonstrate_edge_cases()
        
        print_header("DEMONSTRATION COMPLETE")
        print("""
🎉 Dynamic Batching Engine demonstration completed successfully!

Key takeaways:
• Dynamic batching provides the best balance of latency and throughput
• SLA protection ensures no request waits too long
• Resource constraints prevent system overload
• Adaptive scheduling responds to real-time conditions

Next steps:
• Run 'python test_batching.py' for comprehensive unit tests
• Run 'python visualize_results.py' for detailed performance analysis
• Explore the code to understand the algorithms
        """)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 