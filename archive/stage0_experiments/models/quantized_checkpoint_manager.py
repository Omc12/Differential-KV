from transformers import BitsAndBytesConfig
import torch

class QuantizedCheckpointManager:
    """
    Manages quantization settings for real-model reproducibility.
    Phase 18 mandates standard 4-bit (NF4) for 7B-scale benchmarks.
    """
    def __init__(self, mode: str = "4bit"):
        self.mode = mode

    def get_config(self):
        if self.mode == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.mode == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            return None

    def get_summary(self):
        return {
            "quantization_mode": self.mode,
            "bnb_4bit_quant_type": "nf4" if self.mode == "4bit" else "n/a",
            "double_quant": True if self.mode == "4bit" else False,
            "compute_dtype": "bfloat16"
        }
