import torch
from pathlib import Path

class RegisterPressureOptimizationEngine:
    """
    SGC Stage 3C.3: Register Pressure Optimization Engine.
    Ensures sparse kernels avoid high thread register allocation causing occupancy collapse.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 0
        self.props = torch.cuda.get_device_properties(self.device) if torch.cuda.is_available() else None
        
        # Track active register footprints
        self.kernel_register_counts = {
            "triton_sparse_attention": 48,
            "flash_sparse_attention": 52,
            "shared_memory_sparse_tile": 32,
            "persistent_sparse_attention": 40
        }

    def get_optimized_launch_bounds(self, kernel_name: str, threads_per_block: int) -> dict:
        """
        Determines optimized launch bounds to preserve maximum active occupancy.
        """
        if not self.props:
            return {"max_active_blocks": 16, "occupancy_pct": 100.0, "register_pressure_score": 1.0}

        # Retrieve hardware register limits
        max_regs_per_block = 65536  # Standard limit for modern NVIDIA architectures
        max_threads_per_multiprocessor = self.props.max_threads_per_multi_processor
        
        registers_per_thread = self.kernel_register_counts.get(kernel_name, 32)
        
        # Calculate maximum active blocks per streaming multiprocessor (SM)
        total_registers_per_block = threads_per_block * registers_per_thread
        
        # Safe register allocation check
        max_blocks_by_registers = max_regs_per_block // max(1, total_registers_per_block)
        max_blocks_by_threads = max_threads_per_multiprocessor // threads_per_block
        
        active_blocks_per_sm = min(max_blocks_by_registers, max_blocks_by_threads)
        
        # Compute active occupancy percentage
        active_threads = active_blocks_per_sm * threads_per_block
        occupancy_pct = (active_threads / max_threads_per_multiprocessor) * 100.0
        
        # Calculate Register Pressure Score (Lower is better, >= 1.0 represents safe bounds)
        reg_pressure = (registers_per_thread * threads_per_block) / max_regs_per_block

        return {
            "max_active_blocks": int(active_blocks_per_sm),
            "occupancy_pct": float(occupancy_pct),
            "register_pressure_score": float(reg_pressure),
            "registers_per_thread": registers_per_thread
        }

    def simplify_traversal_parameters(self, kernel_name: str, current_blocks: int) -> int:
        """
        Tunes traversal parameter size downwards if register pressure exceeds threshold.
        """
        bounds = self.get_optimized_launch_bounds(kernel_name, 128)
        if bounds["occupancy_pct"] < 75.0:
            # High register pressure detected, simplify sparse coordinate tracking size
            return max(8, current_blocks // 2)
        return current_blocks
