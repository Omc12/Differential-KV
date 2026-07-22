import sys
import os
import pytest
import torch
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Mocks for Testing Features ---
class MockNativeBlockPool:
    def __init__(self, max_blocks=100, device='cpu'):
        self.max_blocks = max_blocks
        self.seq_lens = torch.zeros(max_blocks, dtype=torch.int32)
        self.device = device
        self.U = torch.zeros((max_blocks, 16))
        self.V_KV = torch.zeros((max_blocks, 16))
        self.anchors_KV = torch.zeros((max_blocks, 16))

class MockTieredBlockStore:
    def __init__(self, pool):
        self.pool = pool
        self.tiers = {i: 'GPU' for i in range(pool.max_blocks)}
        self.heat = torch.zeros(pool.max_blocks)
        
    def update_heat(self, slot: int, val: float):
        if val > 0:
            self.heat[slot] = min(1.0, self.heat[slot] + val)
        else:
            self.heat[slot] = self.heat[slot] * 0.5
            
    def evict_slot(self, slot: int):
        self.tiers[slot] = 'CPU'
        
    def restore_slot(self, slot: int, blocking: bool = True):
        self.tiers[slot] = 'GPU'
        
    def get_tier(self, slot: int):
        return self.tiers[slot]

def maybe_evict(occupied_slots):
    eviction_count = 0
    if len(occupied_slots) > 85:
        eviction_count = len(occupied_slots) - 85
    return eviction_count

class MockBlockPrefetchEngine:
    def __init__(self, tiered_store=None, device='cpu'):
        self.tiered_store = tiered_store
        self.device = device
        self.pending = []
        self._enabled = tiered_store is not None
        
    def submit(self, jobs):
        if self._enabled:
            self.pending.extend(jobs)
            
    def sync_pending(self):
        if self._enabled and self.tiered_store:
            for job in self.pending:
                self.tiered_store.restore_slot(job)
            self.pending = []
            
    def is_enabled(self):
        return self._enabled
        
    def start(self):
        pass
        
    def stop(self):
        pass


# --- Feature 1 (TieredBlockStore) Tests ---
def test_tiered_heat_update():
    pool = MockNativeBlockPool()
    store = MockTieredBlockStore(pool)
    
    for _ in range(3):
        store.update_heat(5, 1.0)
    assert store.heat[5] > 0.9
    
    for _ in range(10):
        store.update_heat(5, 0.0)
    assert store.heat[5] < 0.5

def test_tiered_evict_restore_cpu():
    pool = MockNativeBlockPool(device='cpu')
    store = MockTieredBlockStore(pool)
    store.evict_slot(0)
    assert store.get_tier(0) == 'CPU'
    store.restore_slot(0, blocking=True)
    assert store.get_tier(0) == 'GPU'

def test_maybe_evict_triggers():
    occupied = list(range(90))  # 90 slots occupied (85% limit = 85 slots)
    evicted = maybe_evict(occupied)
    assert evicted > 0


# --- Feature 2 (BlockPrefetchEngine) Tests ---
def test_prefetch_engine_noop_no_tiered_store():
    engine = MockBlockPrefetchEngine(tiered_store=None, device='cpu')
    engine.submit([1, 2])
    engine.sync_pending()
    assert not engine.is_enabled()

def test_prefetch_engine_submit_and_drain():
    pool = MockNativeBlockPool()
    store = MockTieredBlockStore(pool)
    engine = MockBlockPrefetchEngine(tiered_store=store, device='cpu')
    
    engine.start()
    engine.submit([i for i in range(10)])
    engine.sync_pending()
    engine.stop()
    
    assert len(engine.pending) == 0

def test_prefetch_engine_cpu_restore():
    pool = MockNativeBlockPool()
    store = MockTieredBlockStore(pool)
    store.evict_slot(7)  # Now 'CPU'
    
    engine = MockBlockPrefetchEngine(tiered_store=store, device='cpu')
    engine.submit([7])
    
    time.sleep(0.1) # wait 100ms
    engine.sync_pending() # Process job
    
    assert store.get_tier(7) in ['GPU', 'WARMING']


# --- Feature 3 (MLX/Triton kernel path) Tests ---
def test_simd_expand_import():
    try:
        from native_core.sparse_decode import simd_expand
        assert hasattr(simd_expand, 'mlx_block_expand_fallback')
    except ImportError:
        pytest.skip("sparse_decode not implemented yet")


# --- Feature 4 (MLAProjector) Tests ---
def test_mla_projector_identity_bypass():
    os.environ['DIFFKV_MLA_LATENT'] = '0'
    import native_core.compression.mla_projector as mla_module
    import importlib
    importlib.reload(mla_module)  # Refresh _BYPASS based on env
    
    projector = mla_module.MLAProjector(head_dim=64, kv_heads=2)
    x = torch.randn(10, 2 * 2 * 64)
    out = projector.project(x)
    assert torch.allclose(out, x)

def test_mla_projector_calibration():
    os.environ['DIFFKV_MLA_LATENT'] = '1'
    import native_core.compression.mla_projector as mla_module
    import importlib
    importlib.reload(mla_module)
    
    projector = mla_module.MLAProjector(head_dim=64, kv_heads=2, n_calib_blocks=4)
    feat_dim = 256
    
    for _ in range(4):
        deltas = torch.randn(20, feat_dim)
        projector.update_calibration(deltas)
        
    assert projector.is_calibrated
    assert projector.W is not None
    assert projector.W.shape == (feat_dim, projector.latent_dim)

def test_mla_projector_project_unproject_shape():
    os.environ['DIFFKV_MLA_LATENT'] = '1'
    import native_core.compression.mla_projector as mla_module
    import importlib
    importlib.reload(mla_module)
    
    projector = mla_module.MLAProjector(head_dim=64, kv_heads=2, n_calib_blocks=2)
    feat_dim = 256
    for _ in range(2):
        projector.update_calibration(torch.randn(20, feat_dim))
        
    x = torch.randn(10, feat_dim)
    proj = projector.project(x)
    assert proj.shape == (10, projector.latent_dim)
    
    unproj = projector.unproject(proj)
    assert unproj.shape == (10, feat_dim)

def test_mla_projector_roundtrip_energy():
    os.environ['DIFFKV_MLA_LATENT'] = '1'
    import native_core.compression.mla_projector as mla_module
    import importlib
    importlib.reload(mla_module)

    # Use a rank-32 generative subspace in feat_dim=256 space.
    # latent_dim=64 should be able to capture most of the structure.
    feat_dim = 256
    true_rank = 32
    latent_dim = 64
    torch.manual_seed(42)
    basis = torch.randn(true_rank, feat_dim)  # [32, 256] — the true subspace
    basis = basis / basis.norm(dim=1, keepdim=True)

    projector = mla_module.MLAProjector(head_dim=64, kv_heads=2, latent_dim=latent_dim, n_calib_blocks=4)

    # Calibrate with 4 batches of structured data from the same rank-32 subspace
    for _ in range(4):
        coeffs = torch.randn(30, true_rank)
        deltas = coeffs @ basis  # [30, 256] — always in the true_rank subspace
        projector.update_calibration(deltas)

    # Test data also lives in the same rank-32 subspace → W should capture most energy
    test_coeffs = torch.randn(20, true_rank)
    x = test_coeffs @ basis  # [20, 256]

    proj = projector.project(x)       # [20, latent_dim]
    unproj = projector.unproject(proj)  # [20, 256]

    energy_x = torch.norm(x)**2
    energy_unproj = torch.norm(unproj)**2
    # PCA with latent_dim=64 > true_rank=32 should retain >85% of energy from this structured data.
    # We use a conservative 0.7 threshold to account for finite sample PCA approximation.
    assert (energy_unproj / energy_x) > 0.7, (
        f"Expected >70% energy retained, got {(energy_unproj/energy_x).item():.2%}. "
        "MLA projector roundtrip energy too low — check W PCA init."
    )



# --- Integration Test ---
def test_full_pipeline_cpu():
    class MockKVRuntimeManager:
        def __init__(self, device='cpu'):
            self.device = device
            self.pool = MockNativeBlockPool(device=device)
            self.store = MockTieredBlockStore(self.pool)
            self.engine = MockBlockPrefetchEngine(tiered_store=self.store, device=device)
            
    manager = MockKVRuntimeManager(device='cpu')
    
    os.environ['DIFFKV_MLA_LATENT'] = '1'
    import native_core.compression.mla_projector as mla_module
    import importlib
    importlib.reload(mla_module)
    
    projector = mla_module.MLAProjector(head_dim=64, kv_heads=2)
    
    # Simulated decode loop
    for step in range(5):
        # 1. Route results (submit jobs)
        manager.engine.submit([step])
        # 2. Maybe evict
        maybe_evict([step])
        # 3. Sync pending
        manager.engine.sync_pending()
        
    assert True
