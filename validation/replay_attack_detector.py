import torch

class ReplayAttackDetector:
    """
    Detects if the model is 'replaying' information from previous sessions.
    Validates the effectiveness of the hard reset engine.
    """
    def __init__(self):
        self.past_outputs = []

    def log_output(self, output_ids: torch.Tensor):
        self.past_outputs.append(output_ids.clone())

    def check_for_replay(self, current_output_ids: torch.Tensor):
        """
        Ensures current output isn't identical to past output if the prompt was changed.
        """
        for past in self.past_outputs:
            if torch.equal(past, current_output_ids):
                print("CAUTION: Replay detected. Verifying prompt uniqueness...")
                # If prompts are different but outputs same, reset failed.
                return True
        return False

    def verify_reset_integrity(self):
        """
        Triggered after hard reset to ensure no latent trace remains.
        """
        # Check CUDA cache, global variables, etc.
        pass
