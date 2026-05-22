import torch
from continuity.session_resume_engine import SessionResumeEngine
from identity.persistent_cognitive_identity import PersistentCognitiveIdentity

def run_cross_session_eval():
    """
    Measures the fidelity of session restoration.
    """
    print("=== Phase 36: Cross-Session Reasoning Evaluation ===")
    
    identity_manager = PersistentCognitiveIdentity(identity_dir="eval_session_identity")
    resume_engine = SessionResumeEngine(identity_manager)
    
    # 1. Create a session state
    session_id = "test_eval_01"
    original_manifolds = torch.randn(1, 50, 32)
    original_state = {"manifolds": original_manifolds, "identity_id": "test_id"}
    
    # 2. Checkpoint
    resume_engine.prepare_shutdown(session_id, original_state)
    
    # 3. Restore
    restored_state = resume_engine.resume_session(session_id)
    
    # 4. Measure fidelity
    restored_manifolds = restored_state["manifolds"]
    diff = torch.abs(original_manifolds - restored_manifolds).mean().item()
    fidelity = 1.0 - diff
    
    print(f"Session Resume Fidelity: {fidelity:.4f}")
    success = fidelity > 0.99
    print(f"Target (>99%): {'PASSED' if success else 'FAILED'}")

if __name__ == "__main__":
    run_cross_session_eval()
