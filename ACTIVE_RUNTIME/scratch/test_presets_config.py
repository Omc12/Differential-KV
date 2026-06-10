import os
import torch
import unittest
import sys

# Ensure ACTIVE_RUNTIME is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from native_core.config import DiffKVConfig
from native_core.kv_runtime_manager import KVRuntimeManager
from native_core.sparse_decode.triton_fused_decode import _pytorch_vectorized_sparse_attn_decode
from runtime.native_block_pool import NativeBlockPool

class TestPresetsConfig(unittest.TestCase):
    def setUp(self):
        # Clean environment before each test
        for key in list(os.environ.keys()):
            if key.startswith("DIFFKV_") or key == "PYTORCH_MPS_HIGH_WATERMARK_RATIO":
                del os.environ[key]

    def test_preset_low(self):
        cfg = DiffKVConfig({"preset": "low"})
        self.assertEqual(cfg.preset, "low")
        self.assertFalse(cfg.decode_cache_enabled)
        self.assertEqual(cfg.decode_cache_max_tokens, 0)
        self.assertEqual(cfg.prefill_chunk_size, 256)
        self.assertEqual(cfg.srl_threshold, 30)
        self.assertFalse(cfg.async_svd)
        self.assertEqual(cfg.mps_watermark, 0.0)
        self.assertFalse(cfg.torch_compile)

    def test_preset_mid(self):
        cfg = DiffKVConfig({"preset": "mid"})
        self.assertEqual(cfg.preset, "mid")
        self.assertTrue(cfg.decode_cache_enabled)
        self.assertEqual(cfg.decode_cache_max_tokens, 4096)
        self.assertEqual(cfg.prefill_chunk_size, 512)
        self.assertEqual(cfg.srl_threshold, 50)
        self.assertTrue(cfg.async_svd)
        self.assertEqual(cfg.mps_watermark, 0.0)
        self.assertFalse(cfg.torch_compile)

    def test_preset_high(self):
        cfg = DiffKVConfig({"preset": "high"})
        self.assertEqual(cfg.preset, "high")
        self.assertTrue(cfg.decode_cache_enabled)
        self.assertEqual(cfg.decode_cache_max_tokens, 16384)
        self.assertEqual(cfg.prefill_chunk_size, 2048)
        self.assertEqual(cfg.srl_threshold, 100)
        self.assertTrue(cfg.async_svd)
        self.assertEqual(cfg.mps_watermark, 0.0)
        self.assertTrue(cfg.torch_compile)

    def test_individual_overrides(self):
        os.environ["DIFFKV_PRESET"] = "low"
        os.environ["DIFFKV_DECODE_CACHE_ENABLED"] = "true"
        os.environ["DIFFKV_SRL_THRESHOLD"] = "75"
        
        # Override via config dict
        cfg = DiffKVConfig({"prefill_chunk_size": 1024})
        self.assertEqual(cfg.preset, "low")
        self.assertTrue(cfg.decode_cache_enabled)      # env override
        self.assertEqual(cfg.srl_threshold, 75)        # env override
        self.assertEqual(cfg.prefill_chunk_size, 1024)  # dict override
        self.assertFalse(cfg.async_svd)                 # preset default

    def test_kv_runtime_manager_integration(self):
        mgr = KVRuntimeManager(
            num_layers=2, heads=4, head_dim=64,
            config={"preset": "low", "srl_threshold": 42}
        )
        self.assertEqual(mgr.config.preset, "low")
        self.assertEqual(mgr.config.srl_threshold, 42)
        self.assertFalse(mgr._async)
        self.assertEqual(mgr.native_pool.config.preset, "low")

    def test_decode_cache_bypass_and_cleanup(self):
        # Create a pool with low preset (cache disabled)
        pool = NativeBlockPool(
            max_blocks=16, num_kv_heads=2, head_dim=32, rank=4, max_seq_len=16,
            device="cpu", dtype=torch.float16, initial_blocks=2
        )
        pool.config = DiffKVConfig({"preset": "low"})

        q = torch.randn(1, 2, 1, 32, dtype=torch.float16)
        block_indices = torch.tensor([0], dtype=torch.int32)
        
        # Write some data to block 0
        pool.U[0] = torch.randn(16, 4, dtype=torch.float16)
        pool.V_K[0] = torch.randn(4, 2, 32, dtype=torch.float16)
        pool.V_V[0] = torch.randn(4, 2, 32, dtype=torch.float16)
        pool.anchors_K[0] = torch.randn(2, 32, dtype=torch.float16)
        pool.anchors_V[0] = torch.randn(2, 32, dtype=torch.float16)
        pool.scales[0] = 1.0
        pool.seq_lens[0] = 16

        decode_workspace = {
            "session1": {
                "gathered_kv": {
                    0: ("some_validation_key", torch.zeros(1))
                }
            }
        }

        # Run decode with cache disabled. It should execute but bypass cache and clear the stale keys.
        out = _pytorch_vectorized_sparse_attn_decode(
            q=q,
            block_indices=block_indices,
            pool=pool,
            dense_blocks=[],
            active_k=None,
            active_v=None,
            num_key_value_groups=1,
            R=4,
            S_MAX=16,
            session_id="session1",
            layer_idx=0,
            decode_workspace=decode_workspace,
            total_seq_len=20
        )
        
        # Verify that workspace cache keys were cleared
        self.assertNotIn("session1", decode_workspace)

    def test_rollback_clear_srl(self):
        mgr = KVRuntimeManager(
            num_layers=2, heads=4, head_dim=64,
            config={"preset": "mid"}
        )
        mgr._session_srl["session1"] = "mock_srl_state"
        
        # Rollback without clear_srl (should keep srl)
        mgr.rollback_session("session1", target_len=10, clear_srl=False)
        self.assertIn("session1", mgr._session_srl)
        
        # Rollback with clear_srl (should pop/clear srl)
        mgr.rollback_session("session1", target_len=10, clear_srl=True)
        self.assertNotIn("session1", mgr._session_srl)

if __name__ == "__main__":
    unittest.main()
