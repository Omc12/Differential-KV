import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase20_generic_validator import run_cross_arch_validation
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-llm-7b-base")
    parser.add_argument("--modes", nargs="+", default=["fp16", "rank8", "lcg"])
    parser.add_argument("--output", type=str, default="results/phase20/deepseek_validation.json")
    args = parser.parse_args()
    run_cross_arch_validation(args.model, args.modes, args.output)
