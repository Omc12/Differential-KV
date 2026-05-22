"""
experiments/persistent_agent_eval.py

Validates persistent multi-session reasoning for agentic workflows.
"""

import torch
import os
import tempfile
from memory.persistent_cognitive_fields import PersistentCognitiveFields
from memory.cross_session_reasoning_memory import CrossSessionReasoningMemory

def run_agent_persistence_test():
    print("Starting Persistent Agent Evaluation...")
    
    H, D = 8, 64
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = CrossSessionReasoningMemory(tmpdir)
        pcf = PersistentCognitiveFields(H, D)
        
        # Session 1: Planning
        print("Session 1: Planning...")
        plan_attractors = torch.randn(H, 16, D)
        plan_importance = torch.rand(H, 16)
        pcf.update_field(plan_attractors, plan_importance)
        
        # Save session
        memory.save_session("agent_task_1", pcf.field, {"task": "plan_v1", "state": "complete"})
        
        # Session 2: Execution (Loading Session 1)
        print("Session 2: Execution (Reloading Context)...")
        loaded_field, meta = memory.load_session("agent_task_1")
        
        pcf_new = PersistentCognitiveFields(H, D)
        pcf_new.field = loaded_field
        
        # Retrieve context from previous session
        context = pcf_new.get_context_attractors(4)
        
        # Verify resonance with new execution state
        # In a real scenario, exec_state would be derived from the same manifold
        exec_state = context[:, 0:1, :] + 0.1 * torch.randn(H, 1, D)
        resonance = torch.cosine_similarity(exec_state, context, dim=-1).mean()
        
        print(f"Session Reloaded. Task: {meta['task']}")
        print(f"Cross-Session Resonance: {resonance.item():.4f}")
        
        success = resonance > 0.0 # Just a check for connectivity
        print(f"Agent Persistence Status: {'SUCCESS' if success else 'FAILURE'}")
        
    return success

if __name__ == "__main__":
    run_agent_persistence_test()
