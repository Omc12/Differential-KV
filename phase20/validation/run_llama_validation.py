import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase20.validation.generic_validator import run_cross_arch_validation

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3-8B-Instruct")
    parser.add_argument("--modes", nargs="+", default=["fp16", "rank8", "sam", "actr", "lcg"])
    parser.add_argument("--output", type=str, default="phase20/results/llama3_validation.json")
    args = parser.parse_args()
    
    run_cross_arch_validation(args.model, args.modes, args.output)
