import random

class NoisyContextInjector:
    """Injects noise and distractions into context to test robustness."""
    
    def __init__(self):
        self.noise_pool = [
            "Wait, I think I forgot to mention something else.",
            "By the way, have you seen the weather report today?",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "ERROR: Logger failed to initialize at 0x7FFF.",
            "SYSTEM NOTICE: Periodic maintenance scheduled for 02:00 UTC.",
            "Note: The previous instruction might be deprecated.",
            "Random fact: Honey never spoils.",
            "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "SELECT * FROM users WHERE status = 'active';",
            "TODO: Refactor the memory resolver to use trust calibration."
        ]

    def inject_noise(self, text, intensity=0.1):
        """Injects noise strings into the text at random positions."""
        lines = text.split(". ")
        num_noisy_lines = int(len(lines) * intensity)
        
        for _ in range(num_noisy_lines):
            pos = random.randint(0, len(lines))
            noise = random.choice(self.noise_pool)
            lines.insert(pos, noise)
            
        return ". ".join(lines)
