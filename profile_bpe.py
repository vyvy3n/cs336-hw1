import cProfile
import pstats
import io
import time
from tests.adapters import run_train_bpe_speed
from tests.common import FIXTURES_PATH

def profile_function():
    """Profile a single function call"""
    input_path = FIXTURES_PATH / "corpus.en"
    
    # Create a profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Run the function
    vocab, merges = run_train_bpe_speed(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    
    pr.disable()
    
    # Print stats sorted by cumulative time (most time-consuming functions first)
    print("=" * 60)
    print("TOP FUNCTIONS BY CUMULATIVE TIME:")
    print("=" * 60)
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(15)  # Top 15 functions
    print(s.getvalue())
    
    # Print stats sorted by internal time (functions that spend most time in themselves)
    print("=" * 60)
    print("TOP FUNCTIONS BY INTERNAL TIME:")
    print("=" * 60)
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('time')
    ps.print_stats(15)  # Top 15 functions
    print(s.getvalue())
    
    # Print stats sorted by number of calls
    print("=" * 60)
    print("TOP FUNCTIONS BY NUMBER OF CALLS:")
    print("=" * 60)
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('calls')
    ps.print_stats(15)  # Top 15 functions
    print(s.getvalue())

def profile_with_context():
    """Profile with more detailed context about what's happening"""
    input_path = FIXTURES_PATH / "corpus.en"
    
    pr = cProfile.Profile()
    pr.enable()
    
    vocab, merges = run_train_bpe_speed(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    
    pr.disable()
    
    # Get detailed stats
    stats = pstats.Stats(pr)
    
    print("=" * 60)
    print("DETAILED ANALYSIS:")
    print("=" * 60)
    
    # Find the most expensive operations
    print("\n1. Most expensive operations (cumulative time):")
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        if ct > 0.1:  # Only show functions taking more than 0.1 seconds
            filename, line_num, func_name = func
            print(f"  {func_name} in {filename}:{line_num} - {ct:.3f}s ({cc} calls)")
    
    print("\n2. Most called functions:")
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        if cc > 1000000:  # Only show functions called more than 1M times
            filename, line_num, func_name = func
            print(f"  {func_name} - {cc:,} calls ({tt:.3f}s total time)")

def profile_specific_lines():
    """Profile specific lines or functions you suspect are slow"""
    input_path = FIXTURES_PATH / "corpus.en"
    
    pr = cProfile.Profile()
    pr.enable()
    
    vocab, merges = run_train_bpe_speed(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    
    pr.disable()
    
    # Look for specific patterns
    stats = pstats.Stats(pr)
    
    print("=" * 60)
    print("LOOKING FOR SPECIFIC BOTTLENECKS:")
    print("=" * 60)
    
    # Look for expensive built-in operations
    expensive_builtins = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, line_num, func_name = func
        if filename == '<built-in>' and ct > 0.05:
            expensive_builtins.append((func_name, cc, ct))
    
    print("\n1. Expensive built-in operations:")
    for func_name, calls, time_taken in sorted(expensive_builtins, key=lambda x: x[2], reverse=True):
        print(f"  {func_name} - {calls:,} calls, {time_taken:.3f}s")
    
    # Look for expensive list/dict operations
    print("\n2. Expensive list/dict operations:")
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, line_num, func_name = func
        if ('list' in func_name or 'dict' in func_name or 'append' in func_name or 'get' in func_name) and ct > 0.05:
            print(f"  {func_name} - {cc:,} calls, {ct:.3f}s")

def save_profile_data():
    """Save profile data to a file for later analysis"""
    input_path = FIXTURES_PATH / "corpus.en"
    
    pr = cProfile.Profile()
    pr.enable()
    
    vocab, merges = run_train_bpe_speed(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    
    pr.disable()
    
    # Save to file
    pr.dump_stats('bpe_profile.stats')
    print("Profile data saved to 'bpe_profile.stats'")
    print("You can analyze it later with: python -m pstats bpe_profile.stats")

if __name__ == "__main__":
    print("PROFILING BPE TRAINING FUNCTION")
    print("=" * 60)
    
    # Run different profiling approaches
    profile_function()
    profile_with_context()
    profile_specific_lines()
    save_profile_data()
    
    print("\n" + "=" * 60)
    print("HOW TO INTERPRET THE RESULTS:")
    print("=" * 60)
    print("1. Cumulative time: Total time spent in function + all functions it calls")
    print("2. Internal time: Time spent only in the function itself (not in subfunctions)")
    print("3. Number of calls: How many times the function was called")
    print("4. Look for:")
    print("   - Functions with high cumulative time but low internal time = calling expensive subfunctions")
    print("   - Functions with high internal time = doing expensive work themselves")
    print("   - Functions called many times = potential for optimization")
    print("   - Built-in operations called millions of times = algorithmic inefficiency") 