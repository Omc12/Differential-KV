class TokenAccountingVerifier:
    """
    Verifies that the number of tokens reported matches the actual tokens generated.
    Prevents "phantom tokens" from inflating TPS.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def verify_counts(self, text, token_ids):
        decoded_len = len(self.tokenizer.encode(text, add_special_tokens=False))
        reported_len = len(token_ids)
        
        if decoded_len != reported_len:
            return False, f"Token count mismatch: reported={reported_len}, decoded={decoded_len}"
        return True, "OK"
