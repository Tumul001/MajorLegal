"""
Generate evaluation_results.json
Simple wrapper to run evaluation and save JSON results
"""
import sys
import os

# Force UTF-8 for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Suppress emoji warnings
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Import and run
from run_real_evaluation import main

if __name__ == "__main__":
    try:
        report = main()
        print("\n" + "="*70)
        print("SUCCESS - Results saved to real_evaluation_report.json")
        print("="*70)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
