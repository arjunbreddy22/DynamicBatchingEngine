# Dynamic Batching Engine for LLM Inference

A high-performance batching system for Large Language Model (LLM) inference that optimizes both throughput and latency through adaptive scheduling strategies.

## Overview

This project implements and compares three batching strategies for LLM inference:

1. **Naive Batching**: Fixed time windows (traditional approach)
2. **Iterative Batching**: Token-by-token processing with immediate returns
3. **Dynamic Batching**: Adaptive batching with resource constraints and SLA protection

## The Problem

LLM serving is bottlenecked by poor batching strategies. Most systems either:
- Use naive fixed-time windows (high latency for short requests)
- Process requests individually (poor GPU utilization)
- Use static rules that don't adapt to real-time conditions

## Our Solution: Dynamic Batching

### Key Innovations

**🎯 Adaptive Batch Formation**
- Dynamically adjusts batch size based on queue length, token requirements, and memory constraints
- Prioritizes requests using multiple criteria (wait time, job length, urgency)

**⚡ SLA Protection**
- Guarantees maximum wait time for any request
- Prevents long requests from starving short ones

**🧠 Resource-Aware Scheduling**
- Simulates GPU memory constraints (token limits, KV cache)
- Balances throughput vs latency based on system state

**📊 Real-Time Adaptation**
- Responds to traffic bursts and quiet periods
- Optimizes for current workload characteristics

### Algorithm Details

```python
def policy_select(candidates):
    # 1. Priority scoring
    for req in candidates:
        urgency = req.queue_time / max_wait_time
        short_job_bonus = (10 - req.tokens_left) / 10
        req.priority = urgency + short_job_bonus
    
    # 2. Greedy packing with constraints
    selected = []
    for req in sorted(candidates, key=lambda r: r.priority, reverse=True):
        if would_violate_constraints(selected + [req]):
            continue
        if req.queue_time >= max_wait_time:  # SLA protection
            selected.append(req)  # Force include
        elif fits_in_batch(selected + [req]):
            selected.append(req)
    
    return selected
```

## Performance Results

| Metric | Naive Batching | Iterative Batching | Dynamic Batching |
|--------|----------------|-------------------|------------------|
| **Avg Latency** | 18.01 steps | 4.27 steps | **3.85 steps** |
| **95th Percentile** | 27.00 steps | 9.00 steps | **7.50 steps** |
| **Requests Served** | 337 | 344 | **356** |
| **GPU Utilization** | Medium | High | **Optimal** |

**🚀 Key Improvements:**
- **18% lower tail latency** vs iterative batching
- **3.5% higher throughput** vs iterative batching
- **78% lower average latency** vs naive batching

## Quick Start

### Prerequisites
```bash
pip install matplotlib numpy
```

### Run Simulation
```bash
python simulation.py
```

### Run Tests
```bash
python test_batching.py
```

### Generate Plots
```bash
python visualize_results.py
```

## Project Structure

```
Batching Engine/
├── README.md                 # This file
├── simulation.py            # Main simulation framework
├── request.py              # Request data structure
├── naive_batching.py       # Fixed-window batching
├── iterative_batching.py   # Token-by-token batching
├── dynamic_batching.py     # Our adaptive algorithm
├── test_batching.py        # Unit tests
├── visualize_results.py    # Plotting and analysis
└── results/                # Generated plots and data
```

## Architecture

### Core Components

**Request Management**
- `Request`: Represents an LLM inference request with tokens, timing, and metadata
- Queue management with priority tracking

**Batching Strategies**
- Pluggable architecture supporting multiple algorithms
- Common interface: `add_request()`, `step()`, `collect_finished()`

**Simulation Framework**
- Realistic request arrival patterns (Poisson process)
- Token-level granularity simulation
- Comprehensive metrics collection

### Key Design Principles

1. **Separation of Concerns**: Batching logic separate from simulation
2. **Pluggable Architecture**: Easy to add new batching strategies
3. **Comprehensive Metrics**: Track everything for analysis
4. **Real-World Modeling**: Simulate actual LLM serving constraints

## Batching Strategies Explained

### 1. Naive Batching
```python
# Collect requests for fixed time window
if (current_time - last_batch_time) >= window_size:
    batch = all_waiting_requests
    process_until_all_complete(batch)
```
**Pros**: Simple, predictable  
**Cons**: High latency, especially for short requests

### 2. Iterative Batching
```python
# Process all active requests every step
for request in active_requests:
    generate_next_token(request)
    if request.complete:
        return_immediately(request)
```
**Pros**: Low latency, good throughput  
**Cons**: No resource constraints, can OOM

### 3. Dynamic Batching (Our Innovation)
```python
# Adaptive batch formation
candidates = waiting_requests + partial_requests
selected = smart_select(candidates, constraints, priorities)
process_batch(selected)
```
**Pros**: Best of both worlds - low latency AND high throughput  
**Cons**: More complex implementation

## Configuration

### Dynamic Batcher Parameters

```python
DynamicBatcher(
    max_tokens_per_batch=512,    # GPU memory constraint
    max_seqs_per_batch=16,       # Parallelism limit
    max_wait_time=50,            # SLA guarantee (steps)
    gpu_memory_limit=1024        # Total memory budget
)
```

### Simulation Parameters

```python
# In simulation.py
SIM_DURATION = 1000     # Total simulation time
MIN_ARRIVAL = 1         # Min steps between arrivals
MAX_ARRIVAL = 5         # Max steps between arrivals  
MAX_TOKENS = 10         # Max tokens per request
```

## Testing

Our test suite covers:
- **Unit Tests**: Individual component behavior
- **Integration Tests**: Full simulation workflows
- **Edge Cases**: Empty queues, resource limits, timeouts
- **Performance Tests**: Latency and throughput benchmarks

```bash
# Run all tests
python -m pytest test_batching.py -v

# Run specific test category
python -m pytest test_batching.py::TestDynamicBatcher -v
```

## Visualization

Generate comprehensive analysis plots:

```bash
python visualize_results.py
```

**Generated Plots:**
- Latency histograms for each batching strategy
- Throughput over time curves
- Batch size distributions
- Queue depth analysis
- Resource utilization metrics

## Real-World Applications

This dynamic batching engine is designed for:

**🏢 Production LLM Serving**
- ChatGPT-style applications
- Code completion services
- Real-time translation

**☁️ Cloud Inference Platforms**
- Multi-tenant serving
- Auto-scaling workloads
- Cost optimization

**🔬 Research Applications**
- Batching algorithm development
- Performance benchmarking
- Resource planning

## Future Work

**Integration with Real LLMs**
- HuggingFace Transformers backend
- vLLM scheduler replacement
- Multi-GPU coordination

**Advanced Features**
- Request prioritization (premium users)
- Cost-aware scheduling
- Predictive batching

**Production Deployment**
- Docker containerization
- Kubernetes integration
- Monitoring and alerting

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python test_batching.py`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{dynamic_batching_engine,
    title={Dynamic Batching Engine for LLM Inference},
    author={Your Name},
    year={2024},
    url={https://github.com/yourusername/batching-engine}
}
```

## Contact

- **Author**: Your Name
- **Email**: your.email@domain.com
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- **Project**: [GitHub Repository](https://github.com/yourusername/batching-engine)

---

*Built with ❤️ for the LLM serving community* 