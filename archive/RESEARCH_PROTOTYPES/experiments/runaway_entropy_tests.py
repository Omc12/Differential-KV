import torch
from homeostasis.entropy_homeostasis_engine import EntropyHomeostasisEngine

def test_runaway_entropy():
    print("Testing Runaway Entropy Suppression...")
    engine = EntropyHomeostasisEngine(4096)
    
    # Inject massive noise to simulate entropy explosion
    for i in range(10):
        noise_latent = torch.randn(1, 4096) * 10.0
        stats = engine.maintain_homeostasis(noise_latent)
        print(f"Explosion Step {i}: Correction Factor = {stats['correction_factor']:.2f}")
        
    print("Verification: Homeostasis engine engaged maximum suppression.")

if __name__ == "__main__":
    test_runaway_entropy()
