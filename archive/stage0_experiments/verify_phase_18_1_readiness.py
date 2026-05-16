import os
import sys
from models.checkpoint_integrity_verifier import CheckpointIntegrityVerifier
from models.tokenizer_consistency_lock import TokenizerConsistencyLock
from models.real_weight_manifest import RealWeightManifest
from models.checkpoint_hash_exporter import CheckpointHashExporter

def verify_readiness():
    print("="*60)
    print("PHASE 18.1 — REAL-MODEL READINESS AUDIT")
    print("="*60)

    # 1. Locate Checkpoint
    # We use the standard HF cache path as the target
    user_profile = os.environ.get("USERPROFILE")
    checkpoint_dir = os.path.join(user_profile, ".cache", "huggingface", "hub", "models--Qwen--Qwen2.5-7B-Instruct", "snapshots")
    
    if not os.path.exists(checkpoint_dir):
        print("[CRITICAL] Qwen2.5-7B snapshot directory not found.")
        return False
        
    snapshots = os.listdir(checkpoint_dir)
    if not snapshots:
        print("[CRITICAL] No snapshots found for Qwen2.5-7B.")
        return False
        
    target_dir = os.path.join(checkpoint_dir, snapshots[0])
    print(f"[INFO] Target Snapshot: {target_dir}")

    # 2. Verify Shards
    verifier = CheckpointIntegrityVerifier()
    manifest = verifier.generate_manifest(target_dir)
    
    # 3. Verify Tokenizer
    tokenizer_lock = TokenizerConsistencyLock()
    tok_passed, tok_msg = tokenizer_lock.verify(target_dir)
    
    # 4. Record Weights
    weight_manifest = RealWeightManifest()
    weights = weight_manifest.record_weights(target_dir)

    # 5. Check Completion
    is_complete = manifest.get("status") == "VERIFIED_PHYSICAL"
    # Check for .incomplete files in blobs (approximate)
    blob_dir = os.path.join(user_profile, ".cache", "huggingface", "hub", "models--Qwen--Qwen2.5-7B-Instruct", "blobs")
    incomplete = [f for f in os.listdir(blob_dir) if f.endswith(".incomplete")]
    
    print("\n--- AUDIT RESULTS ---")
    print(f"Tokenizer: {'[PASS]' if tok_passed else '[FAIL] (' + str(tok_msg) + ')'}")
    print(f"Weight Shards: {len(weights.get('weights', []))} detected")
    print(f"Incomplete Downloads: {len(incomplete)}")
    
    if len(incomplete) > 0 or not is_complete:
        print("\n[BLOCKER] Mandatory Rule: DO NOT benchmark until all shards are verified.")
        print(f"Status: INCOMPLETE (Missing/Incomplete shards: {len(incomplete)})")
        return False

    # 6. Export Final Hash Manifest
    exporter = CheckpointHashExporter()
    exporter.export(manifest)
    
    print("\n[SUCCESS] Phase 18.1 Readiness: VERIFIED. You may proceed to benchmarking.")
    return True

if __name__ == "__main__":
    verify_readiness()
