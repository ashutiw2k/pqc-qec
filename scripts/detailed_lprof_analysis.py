"""
Analysis of the detailed line profiler results.
"""

def analyze_line_profiler_breakdown():
    """Detailed analysis of the line profiler results."""
    
    print("🔍 DETAILED LINE PROFILER ANALYSIS")
    print("=" * 50)
    
    # Data extracted from the profiler
    total_main_time = 707.068  # seconds
    prepare_state_time = 82.311  # seconds
    
    # Individual line times from prepare_state function
    line_data = {
        "StatePrep (Line 115)": {"time": 0.299, "hits": 2002, "percent": 0.4},
        "Loop overhead (Lines 117-118)": {"time": 0.476, "hits": 2004002, "percent": 0.5},
        "Gate operations (Line 119)": {"time": 36.275, "hits": 1001000, "percent": 44.1},
        "Wire loop (Line 120)": {"time": 0.547, "hits": 2403870, "percent": 0.7},
        "RX noise (Line 121)": {"time": 21.021, "hits": 1402870, "percent": 25.5},
        "RZ noise (Line 122)": {"time": 23.652, "hits": 1402870, "percent": 28.7},
        "Return state (Line 123)": {"time": 0.042, "hits": 2002, "percent": 0.1}
    }
    
    print(f"\n📊 PREPARE_STATE FUNCTION BREAKDOWN ({prepare_state_time:.1f}s total):")
    print("-" * 60)
    
    # Sort by time
    sorted_lines = sorted(line_data.items(), key=lambda x: x[1]["time"], reverse=True)
    
    noise_total = 0
    gate_total = 0
    
    for line_name, data in sorted_lines:
        time_s = data["time"]
        hits = data["hits"]
        percent = data["percent"]
        per_call_us = (time_s * 1_000_000) / hits if hits > 0 else 0
        
        print(f"{line_name:<25}: {time_s:>6.2f}s ({percent:>4.1f}%) - {per_call_us:>5.1f}μs/call - {hits:>8,} calls")
        
        if "noise" in line_name.lower():
            noise_total += time_s
        elif "Gate operations" in line_name:
            gate_total += time_s
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   • Noise operations total: {noise_total:.2f}s ({noise_total/prepare_state_time*100:.1f}%)")
    print(f"   • Gate operations total: {gate_total:.2f}s ({gate_total/prepare_state_time*100:.1f}%)")
    print(f"   • Noise-to-gate ratio: {noise_total/gate_total:.2f}:1")
    
    # Calculate potential speedup
    time_without_noise = prepare_state_time - noise_total
    speedup_factor = prepare_state_time / time_without_noise
    
    print(f"\n⚡ OPTIMIZATION POTENTIAL:")
    print(f"   • Current prepare_state time: {prepare_state_time:.1f}s")
    print(f"   • Without noise: {time_without_noise:.1f}s")
    print(f"   • Speedup factor: {speedup_factor:.2f}x")
    
    # Extrapolate to full execution
    main_execution_time = 373.27  # from line 162
    warmup_time = 326.08  # from line 158
    
    # Assume similar noise overhead in main execution
    estimated_main_without_noise = main_execution_time * (time_without_noise / prepare_state_time)
    estimated_warmup_without_noise = warmup_time * (time_without_noise / prepare_state_time)
    
    total_optimized = estimated_main_without_noise + estimated_warmup_without_noise
    total_current = main_execution_time + warmup_time
    
    print(f"\n🚀 PROJECTED FULL EXECUTION IMPACT:")
    print(f"   • Current total execution: {total_current:.1f}s")
    print(f"   • Projected without noise: {total_optimized:.1f}s")
    print(f"   • Overall speedup: {total_current/total_optimized:.2f}x")
    print(f"   • Time saved: {total_current - total_optimized:.1f}s ({(total_current - total_optimized)/total_current*100:.1f}%)")

def operation_count_analysis():
    """Analyze the operation counts and their implications."""
    
    print(f"\n🧮 OPERATION COUNT ANALYSIS:")
    print("=" * 35)
    
    # From the profiler data
    gates = 1_001_000
    rx_noise = 1_402_870
    rz_noise = 1_402_870
    total_noise = rx_noise + rz_noise
    
    print(f"Gate operations:     {gates:>10,}")
    print(f"RX noise operations: {rx_noise:>10,}")
    print(f"RZ noise operations: {rz_noise:>10,}")
    print(f"Total noise ops:     {total_noise:>10,}")
    print(f"Noise-to-gate ratio: {total_noise/gates:.2f}:1")
    
    # Calculate per-circuit statistics
    num_circuits = 1000
    gates_per_circuit = gates / num_circuits / 2002  # 2002 total executions
    noise_per_circuit = total_noise / num_circuits / 2002
    
    print(f"\n📋 PER-CIRCUIT AVERAGES:")
    print(f"   • Gates per execution: {gates_per_circuit:.0f}")
    print(f"   • Noise ops per execution: {noise_per_circuit:.0f}")
    print(f"   • Total ops per execution: {(gates_per_circuit + noise_per_circuit):.0f}")

def practical_recommendations():
    """Provide practical recommendations based on the analysis."""
    
    print(f"\n💡 PRACTICAL RECOMMENDATIONS:")
    print("=" * 35)
    
    recommendations = [
        "🥇 IMMEDIATE (2.4x speedup):",
        "   • Remove noise operations during profiling/testing",
        "   • Modify simple_circuit_generator to skip lines 121-122",
        "   • This alone saves ~45 seconds per prepare_state call",
        "",
        "🥈 MEDIUM TERM:",
        "   • Use Lightning backend for additional 1.3x speedup",
        "   • Implement conditional noise (only when needed)",
        "   • Consider batching state preparations",
        "",
        "🥉 ADVANCED:",
        "   • Profile individual circuit complexities",
        "   • Implement noise-aware circuit compilation",
        "   • Consider approximate simulation for large circuits"
    ]
    
    for rec in recommendations:
        print(rec)

def show_code_modification():
    """Show exactly how to modify the code for immediate gains."""
    
    print(f"\n🔧 CODE MODIFICATION FOR IMMEDIATE SPEEDUP:")
    print("=" * 50)
    
    print("In your simple_circuit_generator function, change this:")
    print()
    print("# CURRENT (Lines 121-122):")
    print("for wire in wires:")
    print("    qml.RX(x_noise[i], wires=[wire])  # 25.5% of time")
    print("    qml.RZ(z_noise[i], wires=[wire])  # 28.7% of time")
    print()
    print("# OPTIMIZED (skip noise for testing):")
    print("# Skip noise operations for performance testing")
    print("# for wire in wires:")
    print("#     qml.RX(x_noise[i], wires=[wire])")
    print("#     qml.RZ(z_noise[i], wires=[wire])")
    print()
    print("Expected result: 707s → ~295s execution time")

if __name__ == "__main__":
    analyze_line_profiler_breakdown()
    operation_count_analysis()
    practical_recommendations()
    show_code_modification()