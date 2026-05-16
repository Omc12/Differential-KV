from benchmark_mode_classifier import BenchmarkModeClassifier, BenchmarkMode
from benchmark_truth_validator import BenchmarkTruthValidator
from benchmark_honesty_guard import BenchmarkHonestyGuard
from benchmark_component_registry import registry
from telemetry_scope_tracker import scope_tracker

class BICResolver:
    """
    Main resolver for BIC (Benchmark Integrity & Classification).
    Establishes the scientific measurement foundation.
    """
    def __init__(self, mode: str):
        self.classifier = BenchmarkModeClassifier(mode)
        self.mode = self.classifier.mode
        self.validator = BenchmarkTruthValidator(self.mode)
        self.guard = BenchmarkHonestyGuard(self.mode)
        print(f"[BIC] Resolver initialized in {self.mode.name} mode.")

    def finalize_benchmark(self, report_filename: str):
        """
        Validates the run and generates the final reports.
        """
        self.validator.validate()
        violations = self.validator.get_violations()
        
        if violations:
            print("\n" + "!"*40)
            print("BIC VALIDATION FAILURES")
            print("!"*40)
            for v in violations:
                print(f" - {v}")
            print("!"*40 + "\n")
            # In a strict mode, we would raise an error here
            
        self.guard.write_mode_report(report_filename)
        self.guard.save_manifests()
        
        return len(violations) == 0
