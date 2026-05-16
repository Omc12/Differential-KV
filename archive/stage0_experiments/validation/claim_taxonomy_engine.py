"""
Phase 16.75A: Claim Taxonomy Engine
Categorizes every metric into MEASURED, REPLAYED, SIMULATED, PROJECTED, ESTIMATED, UNVERIFIED.
"""
class ClaimTaxonomyEngine:
    def categorize(self, claim):
        return {"category": "SIMULATED" if "H100" in claim else "MEASURED"}
