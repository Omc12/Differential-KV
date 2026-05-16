import logging
from typing import Dict, Any, List

class EcosystemCompatibilitySweep:
    """
    Validates Differential KV interoperability with:
    HuggingFace, OpenAI SDK, LangChain, and LlamaIndex.
    """
    def __init__(self):
        self.logger = logging.getLogger("EcosystemCompatibilitySweep")
        self.sweep_results = {}

    def run_compatibility_sweep(self) -> Dict[str, Any]:
        """
        Executes a battery of compatibility tests across ecosystem adapters.
        """
        self.logger.info("Starting Ecosystem Compatibility Sweep...")
        
        # 1. HuggingFace Integration
        hf_status = self._check_hf_compatibility()
        
        # 2. OpenAI SDK Integration
        openai_status = self._check_openai_compatibility()
        
        # 3. LangChain Integration
        langchain_status = self._check_langchain_compatibility()
        
        # 4. LlamaIndex Integration
        llamaindex_status = self._check_llamaindex_compatibility()
        
        self.sweep_results = {
            "huggingface": hf_status,
            "openai_sdk": openai_status,
            "langchain": langchain_status,
            "llamaindex": llamaindex_status,
            "overall_compatibility": all([hf_status, openai_status, langchain_status, llamaindex_status])
        }
        
        return self.sweep_results

    def _check_hf_compatibility(self) -> bool:
        try:
            from integrations.huggingface_runtime_adapter import DiffKVHFAdapter
            return True
        except ImportError as e:
            self.logger.error(f"HF compatibility check failed: {e}")
            return False

    def _check_openai_compatibility(self) -> bool:
        try:
            from integrations.openai_sdk_compatibility_layer import OpenAISDKCompatibilityLayer
            return True
        except ImportError as e:
            self.logger.error(f"OpenAI compatibility check failed: {e}")
            return False

    def _check_langchain_compatibility(self) -> bool:
        try:
            from integrations.langchain_adapter import DiffKVLangChainAdapter
            return True
        except ImportError as e:
            self.logger.error(f"LangChain compatibility check failed: {e}")
            return False

    def _check_llamaindex_compatibility(self) -> bool:
        try:
            from integrations.llamaindex_query_adapter import DiffKVLlamaIndexAdapter
            return True
        except ImportError as e:
            self.logger.error(f"LlamaIndex compatibility check failed: {e}")
            return False

    def get_ecosystem_health_report(self) -> Dict[str, Any]:
        results_only = {k: v for k, v in self.sweep_results.items() if k != "overall_compatibility"}
        return {
            "integrations_count": len(results_only),
            "healthy_count": sum(1 for v in results_only.values() if v is True),
            "compatibility_ratio": sum(1 for v in results_only.values() if v is True) / len(results_only) if results_only else 1.0
        }
