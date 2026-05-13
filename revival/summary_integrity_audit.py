import os
import torch

class SummaryIntegrityAudit:
    """
    Ensures that summaries do not contain hidden states or latent leakage.
    """

    def __init__(self):
        pass

    def audit_summary(self, summary_obj):
        """
        Check if the summary contains any non-integer or high-dimensional data
        that could be a hidden state.
        """
        if isinstance(summary_obj, torch.Tensor):
            if summary_obj.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                return False, "Summary contains floating point values (Potential Latent Leakage)."
        
        if isinstance(summary_obj, list):
            for item in summary_obj:
                if not isinstance(item, int):
                    return False, f"Summary contains non-integer token ID: {type(item)}"
        
        return True, "Summary integrity verified (Token-level only)."

    def scan_for_persistence_files(self):
        """Checks for any unauthorized persistence on disk."""
        forbidden_extensions = [".pt", ".pth", ".bin", ".safetensors"]
        revival_dir = "revival"
        
        found_files = []
        for root, dirs, files in os.walk(revival_dir):
            for file in files:
                if any(file.endswith(ext) for ext in forbidden_extensions):
                    found_files.append(os.path.join(root, file))
                    
        if found_files:
            return False, f"Forbidden persistence files found: {found_files}"
            
        return True, "No persistence leakage detected on disk."

if __name__ == "__main__":
    auditor = SummaryIntegrityAudit()
    valid, msg = auditor.audit_summary([1, 2, 3])
    print(f"Test 1 (Valid): {valid} - {msg}")
    
    valid, msg = auditor.audit_summary(torch.randn(1, 128))
    print(f"Test 2 (Invalid): {valid} - {msg}")
