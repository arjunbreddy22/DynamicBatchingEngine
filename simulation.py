import random
import statistics
import inspect

# Import your batching modules
from naive_batching import NaiveBatcher
from iterative_batching import IterativeBatcher
from request import Request
'''
class Request:
    """
    Represents an LLM request in the simulation.
    """
    def __init__(self, request_id, arrival_time, tokens_needed):
        self.id = request_id
        self.arrival_time = arrival_time
        self.tokens_left = tokens_needed
        self.finish_time = None
'''

def run_simulation(batcher, sim_duration=1000, min_arrival=1, max_arrival=5, max_tokens=10):
    """
    Runs the simulation for a given batching strategy.
    Batcher must implement:
      - add_request(req)
      - step() or step(current_time)
      - collect_finished(current_time) -> list of finished Request objects
    """
    current_time = 0
    next_arrival = random.randint(min_arrival, max_arrival)
    completed_requests = []
    request_id = 0

    # Determine if batcher.step expects current_time
    step_params = inspect.signature(batcher.step).parameters
    step_accepts_time = len(step_params) > 0

    while current_time < sim_duration:
        # 1) Generate new request if it's arrival time
        if current_time == next_arrival:
            tokens = random.randint(1, max_tokens)
            req = Request(request_id, current_time, tokens)
            request_id += 1
            batcher.add_request(req)
            # Schedule next arrival
            next_arrival += random.randint(min_arrival, max_arrival)

        # 2) Process one "token generation" step
        if step_accepts_time:
            batcher.step(current_time)
        else:
            batcher.step()

        # 3) Collect any finished requests
        finished = batcher.collect_finished(current_time)
        for req in finished:
            completed_requests.append(req)

        current_time += 1

    return completed_requests

def print_stats(results, name):
    """
    Prints basic latency statistics for a list of completed Request objects.
    """
    latencies = [req.finish_time - req.arrival_time for req in results]
    if not latencies:
        print(f"{name}: No requests completed.")
        return

    avg_latency = statistics.mean(latencies)
    p95_latency = statistics.quantiles(latencies, n=20)[-1]  # 95th percentile
    print(f"{name} batching:")
    print(f"  Requests served:     {len(latencies)}")
    print(f"  Avg latency (steps): {avg_latency:.2f}")
    print(f"  95th perc latency:   {p95_latency:.2f}")
    print()

if __name__ == "__main__":
    # Simulation parameters
    SIM_DURATION = 1000    # total time steps
    MIN_ARRIVAL = 1        # min interval between arrivals
    MAX_ARRIVAL = 5        # max interval between arrivals
    MAX_TOKENS = 10        # max tokens per request

    # Initialize batching strategies
    naive_batcher = NaiveBatcher(window_size=20)
    iterative_batcher = IterativeBatcher()

    # Run simulations
    naive_results = run_simulation(naive_batcher, SIM_DURATION, MIN_ARRIVAL, MAX_ARRIVAL, MAX_TOKENS)
    iterative_results = run_simulation(iterative_batcher, SIM_DURATION, MIN_ARRIVAL, MAX_ARRIVAL, MAX_TOKENS)

    # Print metrics
    print_stats(naive_results, "Naive")
    print_stats(iterative_results, "Iterative")
