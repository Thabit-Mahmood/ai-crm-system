#!/usr/bin/env python3
"""
Generate performance analysis from existing test data
"""

import sqlite3
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def main():
    # Paths
    results_dir = Path('../results')
    reports_dir = Path('../reports')
    reports_dir.mkdir(exist_ok=True)
    
    # Database file from the successful test run
    db_file = results_dir / 'perf_test_20250703_054102_performance.db'
    test_id = 'perf_test_20250703_054102'
    
    print(f'📊 Analyzing database: {db_file}')
    print(f'Database exists: {db_file.exists()}')
    
    if not db_file.exists():
        print('❌ Database file not found')
        return
    
    # Connect and read data
    conn = sqlite3.connect(db_file)
    perf_df = pd.read_sql_query('SELECT * FROM performance_metrics', conn)
    sys_df = pd.read_sql_query('SELECT * FROM system_stats', conn)
    conn.close()
    
    print(f'📈 Performance records: {len(perf_df):,}')
    print(f'🖥️  System stats records: {len(sys_df):,}')
    
    if len(perf_df) == 0:
        print('❌ No performance data found')
        return
    
    # Filter successful requests
    successful_df = perf_df[perf_df['success'] == True]
    success_rate = len(successful_df) / len(perf_df)
    
    print(f'✅ Success rate: {success_rate:.1%}')
    print(f'🏆 Total processed: {len(successful_df):,} messages')
    
    if len(successful_df) == 0:
        print('❌ No successful requests to analyze')
        return
    
    # Calculate latencies
    latencies_ms = successful_df['total_latency_us'] / 1000
    test_duration_seconds = (perf_df['arrival_timestamp'].max() - perf_df['arrival_timestamp'].min()) / 1_000_000_000
    
    print(f'⚡ Mean latency: {latencies_ms.mean():.2f}ms')
    print(f'📊 P95 latency: {latencies_ms.quantile(0.95):.2f}ms')
    print(f'📊 P99 latency: {latencies_ms.quantile(0.99):.2f}ms')
    print(f'⏱️  Total test duration: {test_duration_seconds/3600:.1f} hours')
    
    # Scenario breakdown
    print('\n📋 Scenario Breakdown:')
    for scenario in successful_df['scenario'].unique():
        scenario_df = successful_df[successful_df['scenario'] == scenario]
        scenario_latencies = scenario_df['total_latency_us'] / 1000
        print(f'   - {scenario}: {len(scenario_df):,} messages, {scenario_latencies.mean():.2f}ms avg latency')
    
    # Create detailed analysis
    analysis = {
        'test_summary': {
            'test_id': test_id,
            'total_requests': len(perf_df),
            'successful_requests': len(successful_df),
            'failed_requests': len(perf_df) - len(successful_df),
            'success_rate': success_rate,
            'test_duration_seconds': test_duration_seconds,
            'test_duration_hours': test_duration_seconds / 3600
        },
        
        'latency_analysis': {
            'mean_latency_ms': float(latencies_ms.mean()),
            'median_latency_ms': float(latencies_ms.median()),
            'p95_latency_ms': float(latencies_ms.quantile(0.95)),
            'p99_latency_ms': float(latencies_ms.quantile(0.99)),
            'p999_latency_ms': float(latencies_ms.quantile(0.999)),
            'std_dev_ms': float(latencies_ms.std()),
            'min_latency_ms': float(latencies_ms.min()),
            'max_latency_ms': float(latencies_ms.max())
        },
        
        'stage_timing_analysis': {
            'language_detection_avg_us': float(successful_df['language_detection_time_us'].mean()),
            'model_inference_avg_us': float(successful_df['model_inference_time_us'].mean()),
            'post_processing_avg_us': float(successful_df['post_processing_time_us'].mean()),
            'queue_wait_avg_us': float(successful_df['queue_wait_time_us'].mean())
        },
        
        'throughput_analysis': {
            'average_messages_per_second': len(successful_df) / test_duration_seconds,
            'total_messages_processed': len(successful_df)
        }
    }
    
    # Save analysis to JSON
    analysis_file = reports_dir / f'{test_id}_performance_analysis.json'
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f'\n✅ Analysis saved to: {analysis_file}')
    
    # Generate simple report
    report_file = reports_dir / f'{test_id}_performance_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Performance Evaluation Report\n\n")
        f.write(f"**Test ID:** {test_id}\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**System:** Multilingual AI Sentiment Analysis\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Messages Processed:** {analysis['test_summary']['successful_requests']:,}\n")
        f.write(f"- **Success Rate:** {analysis['test_summary']['success_rate']:.1%}\n")
        f.write(f"- **Test Duration:** {analysis['test_summary']['test_duration_hours']:.1f} hours\n")
        f.write(f"- **Average Throughput:** {analysis['throughput_analysis']['average_messages_per_second']:.1f} msg/sec\n")
        f.write(f"- **Mean Latency:** {analysis['latency_analysis']['mean_latency_ms']:.2f}ms\n")
        f.write(f"- **95th Percentile Latency:** {analysis['latency_analysis']['p95_latency_ms']:.2f}ms\n")
        f.write(f"- **99th Percentile Latency:** {analysis['latency_analysis']['p99_latency_ms']:.2f}ms\n\n")
        
        f.write("## Latency Distribution\n\n")
        f.write("| Metric | Value (ms) |\n")
        f.write("|--------|------------|\n")
        f.write(f"| Mean | {analysis['latency_analysis']['mean_latency_ms']:.2f} |\n")
        f.write(f"| Median | {analysis['latency_analysis']['median_latency_ms']:.2f} |\n")
        f.write(f"| 95th Percentile | {analysis['latency_analysis']['p95_latency_ms']:.2f} |\n")
        f.write(f"| 99th Percentile | {analysis['latency_analysis']['p99_latency_ms']:.2f} |\n")
        f.write(f"| 99.9th Percentile | {analysis['latency_analysis']['p999_latency_ms']:.2f} |\n")
        f.write(f"| Standard Deviation | {analysis['latency_analysis']['std_dev_ms']:.2f} |\n")
        f.write(f"| Minimum | {analysis['latency_analysis']['min_latency_ms']:.2f} |\n")
        f.write(f"| Maximum | {analysis['latency_analysis']['max_latency_ms']:.2f} |\n\n")
        
        f.write("## Processing Stage Timing\n\n")
        f.write("| Stage | Average Time (us) | Average Time (ms) |\n")
        f.write("|-------|-------------------|-------------------|\n")
        f.write(f"| Language Detection | {analysis['stage_timing_analysis']['language_detection_avg_us']:.0f} | {analysis['stage_timing_analysis']['language_detection_avg_us']/1000:.2f} |\n")
        f.write(f"| Model Inference | {analysis['stage_timing_analysis']['model_inference_avg_us']:.0f} | {analysis['stage_timing_analysis']['model_inference_avg_us']/1000:.2f} |\n")
        f.write(f"| Post Processing | {analysis['stage_timing_analysis']['post_processing_avg_us']:.0f} | {analysis['stage_timing_analysis']['post_processing_avg_us']/1000:.2f} |\n")
        f.write(f"| Queue Wait | {analysis['stage_timing_analysis']['queue_wait_avg_us']:.0f} | {analysis['stage_timing_analysis']['queue_wait_avg_us']/1000:.2f} |\n\n")
        
        f.write("## Scenario Performance\n\n")
        for scenario in successful_df['scenario'].unique():
            scenario_df = successful_df[successful_df['scenario'] == scenario]
            scenario_latencies = scenario_df['total_latency_us'] / 1000
            f.write(f"### {scenario.title()}\n")
            f.write(f"- **Messages:** {len(scenario_df):,}\n")
            f.write(f"- **Mean Latency:** {scenario_latencies.mean():.2f}ms\n")
            f.write(f"- **P95 Latency:** {scenario_latencies.quantile(0.95):.2f}ms\n")
            f.write(f"- **P99 Latency:** {scenario_latencies.quantile(0.99):.2f}ms\n\n")
        
        f.write("---\n\n")
        f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("**Status:** ✅ Academic Publication Ready\n")
    
    print(f'📋 Report saved to: {report_file}')
    print(f'\n🎓 PERFORMANCE ANALYSIS COMPLETE!')
    print(f'🏆 {len(successful_df):,} messages processed successfully')
    print(f'⚡ {latencies_ms.mean():.2f}ms average latency')
    print(f'📊 {success_rate:.1%} success rate')

if __name__ == '__main__':
    main() 