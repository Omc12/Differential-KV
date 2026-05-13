import hashlib
import json

class ConfigFingerprint:
    """
    Generates a unique fingerprint for a given configuration.
    Ensures that identical configs produce identical fingerprints.
    """
    def generate(self, config):
        # Sort keys to ensure deterministic hashing
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def verify(self, config, expected_fingerprint):
        return self.generate(config) == expected_fingerprint
