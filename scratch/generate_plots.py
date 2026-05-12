import torch
from experiments.infinite_horizon_reasoning import InfiniteHorizonReasoningSim
import os

# Create results directory
os.makedirs("results/phase25", exist_ok=True)

sim = InfiniteHorizonReasoningSim(max_steps=500)
print("Running simulation for plotting...")
result = sim.run_experiment(use_rcr=True)

# Generate plot
sim.dynamics.plot_dynamics("results/phase25/rcr_dynamics.png")
print("Plot saved to results/phase25/rcr_dynamics.png")
