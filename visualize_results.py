import matplotlib.pyplot as plt
import numpy as np
import json
import os
from collections import defaultdict
import statistics

from simulation import run_simulation
from naive_batching import NaiveBatcher
from iterative_batching import IterativeBatcher
from dynamic_batching import DynamicBatcher


def run_multiple_simulations(num_runs=10, sim_duration=1000):
    """Run multiple simulations to get statistically significant results"""
    
    print(f"Running {num_runs} simulations with {sim_duration} steps each...")
    
    all_results = {
        'naive': [],
        'iterative': [],
        'dynamic': []
    }
    
    batch_metrics = {
        'naive': [],
        'iterative': [],
        'dynamic': []
    }
    
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}")
        
        # Initialize batchers
        naive_batcher = NaiveBatcher(window_size=20)
        iterative_batcher = IterativeBatcher()
        dynamic_batcher = DynamicBatcher(
            max_tokens_per_batch=100,
            max_seqs_per_batch=8,
            max_wait_time=30,
            gpu_memory_limit=500
        )
        
        # Run simulations
        naive_results = run_simulation(naive_batcher, sim_duration, 1, 5, 10)
        iterative_results = run_simulation(iterative_batcher, sim_duration, 1, 5, 10)
        dynamic_results = run_simulation(dynamic_batcher, sim_duration, 1, 5, 10)
        
        all_results['naive'].append(naive_results)
        all_results['iterative'].append(iterative_results)
        all_results['dynamic'].append(dynamic_results)
        
        # Collect batch metrics
        batch_metrics['dynamic'].append(dynamic_batcher.get_metrics())
    
    return all_results, batch_metrics


def calculate_statistics(all_results):
    """Calculate comprehensive statistics from multiple simulation runs"""
    
    stats = {}
    
    for strategy, runs in all_results.items():
        strategy_stats = {
            'latencies': [],
            'requests_served': [],
            'avg_latency': [],
            'p95_latency': [],
            'p99_latency': [],
            'throughput': []
        }
        
        for run_results in runs:
            if not run_results:
                continue
                
            latencies = [req.finish_time - req.arrival_time for req in run_results]
            
            strategy_stats['latencies'].extend(latencies)
            strategy_stats['requests_served'].append(len(run_results))
            strategy_stats['avg_latency'].append(statistics.mean(latencies))
            
            if len(latencies) >= 20:  # Need enough data for percentiles
                strategy_stats['p95_latency'].append(np.percentile(latencies, 95))
                strategy_stats['p99_latency'].append(np.percentile(latencies, 99))
            
            # Throughput: requests per time unit
            if run_results:
                max_time = max(req.finish_time for req in run_results)
                strategy_stats['throughput'].append(len(run_results) / max_time)
        
        stats[strategy] = strategy_stats
    
    return stats


def create_latency_histogram(stats, save_path='results'):
    """Create latency distribution histograms"""
    
    plt.figure(figsize=(15, 5))
    
    strategies = ['naive', 'iterative', 'dynamic']
    colors = ['#ff7f7f', '#7f7fff', '#7fff7f']
    
    for i, strategy in enumerate(strategies):
        plt.subplot(1, 3, i + 1)
        
        latencies = stats[strategy]['latencies']
        if not latencies:
            continue
            
        plt.hist(latencies, bins=30, alpha=0.7, color=colors[i], edgecolor='black')
        plt.title(f'{strategy.capitalize()} Batching\nLatency Distribution')
        plt.xlabel('Latency (steps)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_lat = np.mean(latencies)
        p95_lat = np.percentile(latencies, 95)
        plt.axvline(mean_lat, color='red', linestyle='--', label=f'Mean: {mean_lat:.1f}')
        plt.axvline(p95_lat, color='orange', linestyle='--', label=f'P95: {p95_lat:.1f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/latency_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_comparison_plot(stats, save_path='results'):
    """Create side-by-side comparison of key metrics"""
    
    strategies = ['naive', 'iterative', 'dynamic']
    colors = ['#ff7f7f', '#7f7fff', '#7fff7f']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Average Latency
    ax = axes[0, 0]
    avg_latencies = [np.mean(stats[s]['avg_latency']) for s in strategies]
    std_latencies = [np.std(stats[s]['avg_latency']) for s in strategies]
    
    bars = ax.bar(strategies, avg_latencies, yerr=std_latencies, 
                  color=colors, alpha=0.7, capsize=5)
    ax.set_title('Average Latency Comparison')
    ax.set_ylabel('Latency (steps)')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, avg_latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom')
    
    # 2. 95th Percentile Latency
    ax = axes[0, 1]
    p95_latencies = [np.mean(stats[s]['p95_latency']) for s in strategies if stats[s]['p95_latency']]
    p95_std = [np.std(stats[s]['p95_latency']) for s in strategies if stats[s]['p95_latency']]
    
    if p95_latencies:
        bars = ax.bar(strategies[:len(p95_latencies)], p95_latencies, yerr=p95_std,
                      color=colors[:len(p95_latencies)], alpha=0.7, capsize=5)
        ax.set_title('95th Percentile Latency')
        ax.set_ylabel('Latency (steps)')
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, p95_latencies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom')
    
    # 3. Requests Served
    ax = axes[1, 0]
    requests_served = [np.mean(stats[s]['requests_served']) for s in strategies]
    requests_std = [np.std(stats[s]['requests_served']) for s in strategies]
    
    bars = ax.bar(strategies, requests_served, yerr=requests_std,
                  color=colors, alpha=0.7, capsize=5)
    ax.set_title('Throughput (Requests Served)')
    ax.set_ylabel('Requests')
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, requests_served):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}', ha='center', va='bottom')
    
    # 4. Throughput Rate
    ax = axes[1, 1]
    throughputs = [np.mean(stats[s]['throughput']) for s in strategies if stats[s]['throughput']]
    throughput_std = [np.std(stats[s]['throughput']) for s in strategies if stats[s]['throughput']]
    
    if throughputs:
        bars = ax.bar(strategies[:len(throughputs)], throughputs, yerr=throughput_std,
                      color=colors[:len(throughputs)], alpha=0.7, capsize=5)
        ax.set_title('Throughput Rate')
        ax.set_ylabel('Requests/Step')
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_batch_analysis(batch_metrics, save_path='results'):
    """Create batch-specific analysis plots for dynamic batcher"""
    
    if not batch_metrics['dynamic']:
        print("No dynamic batch metrics available for analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract metrics across all runs
    avg_batch_sizes = [m['avg_batch_size'] for m in batch_metrics['dynamic'] if m]
    max_batch_sizes = [m['max_batch_size'] for m in batch_metrics['dynamic'] if m]
    avg_token_usage = [m['avg_token_usage'] for m in batch_metrics['dynamic'] if m]
    total_batches = [m['total_batches'] for m in batch_metrics['dynamic'] if m]
    
    # 1. Batch Size Distribution
    ax = axes[0, 0]
    ax.hist(avg_batch_sizes, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax.set_title('Average Batch Size Distribution')
    ax.set_xlabel('Average Batch Size')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(avg_batch_sizes), color='red', linestyle='--', 
               label=f'Mean: {np.mean(avg_batch_sizes):.1f}')
    ax.legend()
    
    # 2. Max Batch Size
    ax = axes[0, 1]
    ax.hist(max_batch_sizes, bins=15, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title('Maximum Batch Size Distribution')
    ax.set_xlabel('Max Batch Size')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(max_batch_sizes), color='red', linestyle='--',
               label=f'Mean: {np.mean(max_batch_sizes):.1f}')
    ax.legend()
    
    # 3. Token Usage
    ax = axes[1, 0]
    ax.hist(avg_token_usage, bins=15, alpha=0.7, color='orange', edgecolor='black')
    ax.set_title('Average Token Usage Distribution')
    ax.set_xlabel('Average Tokens per Batch')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(avg_token_usage), color='red', linestyle='--',
               label=f'Mean: {np.mean(avg_token_usage):.1f}')
    ax.legend()
    
    # 4. Total Batches
    ax = axes[1, 1]
    ax.hist(total_batches, bins=15, alpha=0.7, color='purple', edgecolor='black')
    ax.set_title('Total Batches Distribution')
    ax.set_xlabel('Total Batches')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(total_batches), color='red', linestyle='--',
               label=f'Mean: {np.mean(total_batches):.1f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/batch_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_performance_table(stats, save_path='results'):
    """Create a summary performance table"""
    
    strategies = ['naive', 'iterative', 'dynamic']
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    
    # Create table data
    table_data = []
    headers = ['Strategy', 'Avg Latency', 'P95 Latency', 'P99 Latency', 'Requests Served', 'Throughput']
    
    for strategy in strategies:
        if not stats[strategy]['latencies']:
            continue
            
        row = [strategy.capitalize()]
        
        # Average latency
        avg_lat = np.mean(stats[strategy]['avg_latency'])
        row.append(f"{avg_lat:.2f} ± {np.std(stats[strategy]['avg_latency']):.2f}")
        
        # P95 latency
        if stats[strategy]['p95_latency']:
            p95_lat = np.mean(stats[strategy]['p95_latency'])
            row.append(f"{p95_lat:.2f} ± {np.std(stats[strategy]['p95_latency']):.2f}")
        else:
            row.append("N/A")
        
        # P99 latency
        if stats[strategy]['p99_latency']:
            p99_lat = np.mean(stats[strategy]['p99_latency'])
            row.append(f"{p99_lat:.2f} ± {np.std(stats[strategy]['p99_latency']):.2f}")
        else:
            row.append("N/A")
        
        # Requests served
        req_served = np.mean(stats[strategy]['requests_served'])
        row.append(f"{req_served:.0f} ± {np.std(stats[strategy]['requests_served']):.0f}")
        
        # Throughput
        if stats[strategy]['throughput']:
            throughput = np.mean(stats[strategy]['throughput'])
            row.append(f"{throughput:.4f} ± {np.std(stats[strategy]['throughput']):.4f}")
        else:
            row.append("N/A")
        
        table_data.append(row)
    
    # Print table
    print(f"{'Strategy':<12} {'Avg Latency':<15} {'P95 Latency':<15} {'P99 Latency':<15} {'Requests':<12} {'Throughput':<15}")
    print("-" * 80)
    
    for row in table_data:
        print(f"{row[0]:<12} {row[1]:<15} {row[2]:<15} {row[3]:<15} {row[4]:<12} {row[5]:<15}")
    
    print("="*80)
    
    # Calculate improvements
    if len(table_data) >= 3:  # All three strategies available
        naive_avg = np.mean(stats['naive']['avg_latency'])
        iter_avg = np.mean(stats['iterative']['avg_latency'])
        dyn_avg = np.mean(stats['dynamic']['avg_latency'])
        
        print("\nIMPROVEMENTS:")
        print(f"Dynamic vs Naive:     {((naive_avg - dyn_avg) / naive_avg * 100):.1f}% latency reduction")
        print(f"Dynamic vs Iterative: {((iter_avg - dyn_avg) / iter_avg * 100):.1f}% latency reduction")
        
        if stats['dynamic']['throughput'] and stats['iterative']['throughput']:
            iter_throughput = np.mean(stats['iterative']['throughput'])
            dyn_throughput = np.mean(stats['dynamic']['throughput'])
            print(f"Dynamic vs Iterative: {((dyn_throughput - iter_throughput) / iter_throughput * 100):.1f}% throughput improvement")
    
    # Save to file
    with open(f'{save_path}/performance_summary.txt', 'w') as f:
        f.write("PERFORMANCE COMPARISON SUMMARY\n")
        f.write("="*50 + "\n\n")
        for row in table_data:
            f.write(f"{row[0]}: {row[1]} avg latency, {row[4]} requests\n")


def save_raw_data(all_results, batch_metrics, save_path='results'):
    """Save raw data for further analysis"""
    
    # Convert results to serializable format
    serializable_results = {}
    
    for strategy, runs in all_results.items():
        serializable_results[strategy] = []
        for run in runs:
            run_data = []
            for req in run:
                run_data.append({
                    'id': req.id,
                    'arrival_time': req.arrival_time,
                    'finish_time': req.finish_time,
                    'latency': req.finish_time - req.arrival_time
                })
            serializable_results[strategy].append(run_data)
    
    # Save results
    with open(f'{save_path}/simulation_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save batch metrics
    with open(f'{save_path}/batch_metrics.json', 'w') as f:
        json.dump(batch_metrics, f, indent=2)


def main():
    """Main visualization pipeline"""
    
    # Create results directory
    save_path = 'results'
    os.makedirs(save_path, exist_ok=True)
    
    print("Dynamic Batching Engine - Performance Visualization")
    print("="*60)
    
    # Run simulations
    all_results, batch_metrics = run_multiple_simulations(num_runs=5, sim_duration=1000)
    
    # Calculate statistics
    stats = calculate_statistics(all_results)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    create_latency_histogram(stats, save_path)
    create_comparison_plot(stats, save_path)
    create_batch_analysis(batch_metrics, save_path)
    create_performance_table(stats, save_path)
    
    # Save raw data
    save_raw_data(all_results, batch_metrics, save_path)
    
    print(f"\nAll results saved to '{save_path}/' directory")
    print("Generated files:")
    print("  - latency_histograms.png")
    print("  - comparison_metrics.png") 
    print("  - batch_analysis.png")
    print("  - performance_summary.txt")
    print("  - simulation_results.json")
    print("  - batch_metrics.json")


if __name__ == '__main__':
    main() 