import torch
import json
import os
from typing import Dict, Any, Optional
from .manifold_fingerprint_engine import ManifoldFingerprintEngine
from .reasoning_signature_tracker import ReasoningSignatureTracker

class PersistentCognitiveIdentity:
    """
    Main manager for persistent cognitive identity (PCI).
    Maintains identity across sessions and regulates autonomous evolution.
    """
    def __init__(self, identity_dir: str = "identity_storage"):
        self.identity_dir = identity_dir
        os.makedirs(identity_dir, exist_ok=True)
        
        self.fp_engine = ManifoldFingerprintEngine()
        self.signature_tracker = ReasoningSignatureTracker()
        
        self.current_identity_id = None
        self.reference_fingerprint = None
        self.identity_metadata = {}

    def initialize_identity(self, manifolds: torch.Tensor, metadata: Optional[Dict] = None):
        """
        Creates a new cognitive identity or loads an existing one.
        """
        fingerprint = self.fp_engine.compute_geometric_fingerprint(manifolds)
        identity_id = self.fp_engine.generate_id_hash(fingerprint)
        
        self.current_identity_id = identity_id
        self.reference_fingerprint = fingerprint
        self.identity_metadata = metadata or {"created_at": "now", "version": "1.0"}
        
        self.save_identity()
        print(f"Initialized Persistent Identity: {identity_id}")

    def update_state(self, manifolds: torch.Tensor, metrics: Dict[str, float]):
        """
        Updates the identity state and tracks drift.
        """
        self.signature_tracker.update(metrics)
        
        if self.reference_fingerprint is not None:
            current_fp = self.fp_engine.compute_geometric_fingerprint(manifolds)
            drift = self.fp_engine.detect_identity_drift(current_fp, self.reference_fingerprint)
            
            # If drift is too high, we might need to update the reference (identity evolution)
            if drift < 0.8: # Example threshold
                print(f"Warning: Significant Identity Drift Detected ({drift:.4f})")
                
    def get_integrity_metrics(self) -> Dict[str, Any]:
        """
        Returns metrics about the stability and integrity of the identity.
        """
        signature = self.signature_tracker.compute_signature()
        return {
            "identity_id": self.current_identity_id,
            "signature": signature,
            "is_stable": signature.get("resonance_stability", 0) > 0.9
        }

    def save_identity(self):
        """
        Persists the identity to disk.
        """
        if self.current_identity_id is None:
            return
            
        path = os.path.join(self.identity_dir, f"{self.current_identity_id}.json")
        data = {
            "identity_id": self.current_identity_id,
            "metadata": self.identity_metadata,
            "fingerprint": self.reference_fingerprint.tolist() if self.reference_fingerprint is not None else None,
            "signature_history": self.signature_tracker.compute_signature()
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def load_identity(self, identity_id: str):
        """
        Loads an identity from disk.
        """
        path = os.path.join(self.identity_dir, f"{identity_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Identity {identity_id} not found.")
            
        with open(path, "r") as f:
            data = json.load(f)
            
        self.current_identity_id = data["identity_id"]
        self.identity_metadata = data["metadata"]
        if data["fingerprint"]:
            self.reference_fingerprint = torch.tensor(data["fingerprint"])
        print(f"Loaded Persistent Identity: {self.current_identity_id}")
