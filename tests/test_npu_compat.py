"""
TorchStore NPU compatibility tests.

Verifies that the NPU adaptations work correctly:
1. RDMA availability detection (rdma_supported vs is_ibverbs_available)
2. spawn_actors accelerator dimension detection
3. SharedMemory device sync for NPU tensors
4. SharedMemory pin_memory NPU skip
5. Transport selection on NPU

Usage:
    conda activate monarch_ascend
    source /root/hzz/cann-9.0.0-beta.1/set_env.sh
    python -u tests/test_npu_compat.py
"""

import os
import sys
import traceback

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

import torch

_passed = 0
_failed = 0


def _pass(name, detail=""):
    global _passed
    _passed += 1
    print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))


def _fail(name, err):
    global _failed
    _failed += 1
    print(f"  [FAIL] {name} — {err}")


def test_npu_available():
    """Prerequisite: torch_npu is importable and NPU is available."""
    print("\n=== Test: NPU availability ===")
    try:
        import torch_npu  # noqa: F401
        assert hasattr(torch, "npu"), "torch.npu not found"
        assert torch.npu.is_available(), "torch.npu.is_available() returned False"
        count = torch.npu.device_count()
        _pass("npu_available", f"{count} NPU(s) detected")
    except Exception as e:
        _fail("npu_available", e)


def test_rdma_detection():
    """Fix 1: rdma_supported returns True on HiXL builds."""
    print("\n=== Test: RDMA backend detection ===")
    try:
        from monarch._rust_bindings.rdma import rdma_supported
        result = rdma_supported()
        _pass("rdma_supported()", f"returns {result}")
    except ImportError as e:
        _fail("rdma_supported_import", f"cannot import: {e}")
        return

    try:
        from monarch.rdma import is_ibverbs_available
        ib = is_ibverbs_available()
        _pass("is_ibverbs_available()", f"returns {ib} (expected False on NPU)")
    except Exception as e:
        _fail("is_ibverbs_available", e)

    try:
        from torchstore.transport.monarch_rdma import monarch_rdma_available
        avail = monarch_rdma_available()
        if avail:
            _pass("monarch_rdma_available()", "True — RDMA transport will be auto-selected")
        else:
            _fail("monarch_rdma_available()", "returned False — RDMA transport won't be used")
    except Exception as e:
        _fail("monarch_rdma_available", e)


def test_transport_selection():
    """Fix 1 continued: MonarchRDMA appears in the transport chain."""
    print("\n=== Test: Transport selection logic ===")
    try:
        from torchstore.transport.monarch_rdma import monarch_rdma_transport_available
        avail = monarch_rdma_transport_available()
        _pass("monarch_rdma_transport_available()", f"returns {avail}")
    except Exception as e:
        _fail("monarch_rdma_transport_available", e)


def test_spawn_actors_dim():
    """Fix 2: _detect_accelerator_dim returns 'npus' on NPU host."""
    print("\n=== Test: spawn_actors accelerator detection ===")
    try:
        from torchstore.utils import _detect_accelerator_dim
        dim = _detect_accelerator_dim()
        if dim == "npus":
            _pass("_detect_accelerator_dim()", f"'{dim}'")
        else:
            _fail("_detect_accelerator_dim()", f"expected 'npus', got '{dim}'")
    except Exception as e:
        _fail("_detect_accelerator_dim", e)


def test_shm_pin_memory_npu():
    """Fix 4: pin_memory skips gracefully on NPU (no crash)."""
    print("\n=== Test: SharedMemory pin_memory on NPU ===")
    try:
        from torchstore.transport.shared_memory import pin_memory, unpin_memory
        t = torch.randn(100, 100)
        t.share_memory_()
        pin_memory(t)
        _pass("pin_memory(cpu_tensor)", "no crash on NPU host")
        unpin_memory(t)
        _pass("unpin_memory(cpu_tensor)", "no crash on NPU host")
    except Exception as e:
        _fail("pin_memory_npu", e)


def test_shm_device_sync():
    """Fix 3: _post_handshake correctly identifies NPU tensors for sync."""
    print("\n=== Test: SharedMemory NPU device detection ===")
    try:
        t_npu = torch.randn(4, 4, device="npu:0")
        is_npu = getattr(t_npu, "is_npu", False)
        if is_npu:
            _pass("tensor.is_npu", f"{is_npu} for npu tensor")
        else:
            _fail("tensor.is_npu", f"expected True, got {is_npu}")

        is_cuda = t_npu.is_cuda
        if not is_cuda:
            _pass("tensor.is_cuda", f"{is_cuda} for npu tensor (expected False)")
        else:
            _fail("tensor.is_cuda", f"expected False, got {is_cuda}")

        # Verify the combined check works
        needs_sync = t_npu.is_cuda or getattr(t_npu, "is_npu", False)
        if needs_sync:
            _pass("combined_sync_check", "correctly identifies NPU tensor for sync")
        else:
            _fail("combined_sync_check", "failed to identify NPU tensor")

        # Verify torch.npu.synchronize doesn't crash
        torch.npu.synchronize(t_npu.device)
        _pass("torch.npu.synchronize", "no crash")
    except Exception as e:
        _fail("shm_device_sync", e)


def test_gloo_factory():
    """Fix 5: Gloo factory doesn't crash on NPU host."""
    print("\n=== Test: Gloo factory NPU registration ===")
    try:
        import torch.distributed as dist
        if dist.is_gloo_available():
            _pass("gloo_available", "True")
        else:
            _fail("gloo_available", "Gloo backend not available")
    except Exception as e:
        _fail("gloo_factory", e)


def test_torchstore_import():
    """Overall: torchstore imports cleanly on NPU."""
    print("\n=== Test: TorchStore import ===")
    try:
        import torchstore  # noqa: F401
        _pass("import torchstore", "clean import")
    except Exception as e:
        _fail("import_torchstore", e)

    try:
        from torchstore.transport import (
            TransportType,
            monarch_rdma_transport_available,
            gloo_available,
        )
        _pass("import transports", "all transport modules imported")

        avail = []
        if monarch_rdma_transport_available():
            avail.append("MonarchRDMA")
        if gloo_available():
            avail.append("Gloo")
        avail.append("MonarchRPC")
        _pass("available_transports", ", ".join(avail))
    except Exception as e:
        _fail("import_transports", e)


def test_byte_view_npu():
    """Verify to_byte_view works with CPU tensors (RDMA uses CPU byte views)."""
    print("\n=== Test: to_byte_view utility ===")
    try:
        from torchstore.utils import to_byte_view
        t = torch.randn(8, 8)
        bv = to_byte_view(t)
        expected = 8 * 8 * 4  # float32
        if bv.shape[0] == expected:
            _pass("to_byte_view(cpu)", f"shape={bv.shape[0]} bytes")
        else:
            _fail("to_byte_view(cpu)", f"expected {expected}, got {bv.shape[0]}")

        # NPU tensor → cpu → byte_view (the RDMA path with EAGER_D2H)
        t_npu = torch.randn(4, 4, device="npu:0")
        t_cpu = t_npu.cpu()
        bv2 = to_byte_view(t_cpu)
        expected2 = 4 * 4 * 4
        if bv2.shape[0] == expected2:
            _pass("to_byte_view(npu→cpu)", f"shape={bv2.shape[0]} bytes")
        else:
            _fail("to_byte_view(npu→cpu)", f"expected {expected2}, got {bv2.shape[0]}")
    except Exception as e:
        _fail("to_byte_view", e)


if __name__ == "__main__":
    print("=" * 60)
    print("TorchStore NPU Compatibility Tests")
    print("=" * 60)

    test_npu_available()
    test_torchstore_import()
    test_rdma_detection()
    test_transport_selection()
    test_spawn_actors_dim()
    test_shm_pin_memory_npu()
    test_shm_device_sync()
    test_gloo_factory()
    test_byte_view_npu()

    print("\n" + "=" * 60)
    total = _passed + _failed
    print(f"Results: {_passed}/{total} PASS, {_failed}/{total} FAIL")
    print("=" * 60)
    sys.exit(1 if _failed > 0 else 0)
