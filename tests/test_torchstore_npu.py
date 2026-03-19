"""
TorchStore NPU end-to-end tests — mirrors GPU test scenarios.

Covers:
  1. Basic put/get through Monarch actors (test_store.py mirror)
  2. Batch put/get with NPU tensors (test_store.py:test_batch_basic mirror)
  3. SharedMemory transport with NPU tensors (test_shared_memory.py GPU class mirror)
  4. Large tensor RDMA transfer (test_large_tensors.py mirror)
  5. Object put/get (test_store.py:test_objects mirror)
  6. Key existence & deletion (test_store.py mirror)

Usage:
    conda activate monarch_ascend
    source /root/hzz/cann-9.0.0-beta.1/set_env.sh
    ASCEND_RT_VISIBLE_DEVICES=0,1 python -u tests/test_torchstore_npu.py [--only GROUP]

    Groups: basic, batch, shm, large, object, keys, all (default)
"""

import argparse
import asyncio
import os
import socket
import sys
import time
import traceback

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("ERROR: torch_npu not available")
    sys.exit(1)

import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint, this_host
from torchstore.transport import TransportType
from torchstore.transport.monarch_rdma import monarch_rdma_transport_available
from torchstore.strategy import HostStrategy
from torchstore.utils import spawn_actors

_passed = 0
_failed = 0
_skipped = 0


def _header(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def _pass(name, detail=""):
    global _passed
    _passed += 1
    print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))


def _fail(name, err):
    global _failed
    _failed += 1
    print(f"  [FAIL] {name} — {err}")
    traceback.print_exc()


def _skip(name, reason):
    global _skipped
    _skipped += 1
    print(f"  [SKIP] {name} — {reason}")


# =====================================================================
# Test 1: Basic put/get (mirrors test_store.py:test_basic)
# =====================================================================

async def test_basic_put_get():
    """Basic put/get with 2 actor meshes through MonarchRPC transport."""
    _header("Test 1: Basic put/get")

    class PutGetActor(Actor):
        def __init__(self, world_size):
            self.world_size = world_size
            self.rank = current_rank().rank
            os.environ["LOCAL_RANK"] = str(self.rank)
            os.environ["HOSTNAME"] = socket.gethostname()

        @endpoint
        async def put(self):
            t = torch.tensor([self.rank + 1] * 10)
            await ts.put(f"basic_key_{self.rank}", t)

        @endpoint
        async def get(self, rank_offset=0):
            other_rank = (self.rank + rank_offset) % self.world_size
            return await ts.get(f"basic_key_{other_rank}")

    try:
        await ts.initialize(
            num_storage_volumes=2,
            strategy=ts.LocalRankStrategy(TransportType.MonarchRPC),
        )
        mesh_put = await spawn_actors(2, PutGetActor, "basic_put", world_size=2)
        mesh_get = await spawn_actors(2, PutGetActor, "basic_get", world_size=2)

        await mesh_put.put.call()
        _pass("put", "2 tensors stored")

        results = await mesh_get.get.call()
        for pt, val in results:
            expected = torch.tensor([pt.rank + 1] * 10)
            assert torch.equal(expected, val), f"{expected} != {val}"
        _pass("get", "cross-mesh retrieval OK")

        results = await mesh_get.get.call(1)
        for pt, val in results:
            other_rank = (pt.rank + 1) % 2
            expected = torch.tensor([other_rank + 1] * 10)
            assert torch.equal(expected, val)
        _pass("get_cross_rank", "rank offset retrieval OK")
    except Exception as e:
        _fail("basic_put_get", e)
    finally:
        await ts.shutdown()


# =====================================================================
# Test 2: Batch put/get with NPU tensors (mirrors test_store.py:test_batch_basic)
# =====================================================================

async def test_batch_npu():
    """Batch put/get with NPU-originated tensors."""
    _header("Test 2: Batch put/get with NPU tensors")

    class BatchActor(Actor):
        def __init__(self, world_size):
            self.world_size = world_size
            self.rank = current_rank().rank
            os.environ["LOCAL_RANK"] = str(self.rank)
            os.environ["HOSTNAME"] = socket.gethostname()

        def _make_batch(self, prefix, offset, device):
            batch = {
                f"{prefix}_t1_{self.rank}": torch.tensor(
                    [self.rank + 1 + offset] * 10, device=device
                ),
                f"{prefix}_t2_{self.rank}": torch.tensor(
                    [self.rank + 100 + offset] * 5, device=device
                ),
                f"{prefix}_obj_{self.rank}": {"rank": self.rank, "offset": offset},
            }
            if device == "npu":
                batch[f"{prefix}_cpu_{self.rank}"] = torch.tensor(
                    [self.rank + 200 + offset] * 8, device="cpu"
                )
            return batch

        @endpoint
        async def put_batch(self, prefix, offset, device):
            await ts.put_batch(self._make_batch(prefix, offset, device))

        @endpoint
        async def get_and_verify(self, prefix, offset, device):
            expected = self._make_batch(prefix, offset, device)
            results = await ts.get_batch(list(expected.keys()))
            for key, expected_val in expected.items():
                actual = results[key]
                if isinstance(expected_val, torch.Tensor):
                    assert torch.equal(actual, expected_val.cpu()), f"{key} mismatch"
                else:
                    assert actual == expected_val, f"{key} mismatch"
            return True

    try:
        await ts.initialize(
            num_storage_volumes=2,
            strategy=ts.LocalRankStrategy(TransportType.MonarchRPC),
        )
        actors = await spawn_actors(2, BatchActor, "batch_npu", world_size=2)

        for device in ["cpu", "npu"]:
            prefix = f"batch_{device}"
            await actors.put_batch.call(prefix, 0, device)
            results = await actors.get_and_verify.call(prefix, 0, device)
            for _, ok in results:
                assert ok
            _pass(f"batch_{device}", f"put+get cycle OK")

            await actors.put_batch.call(prefix, 1000, device)
            results = await actors.get_and_verify.call(prefix, 1000, device)
            for _, ok in results:
                assert ok
            _pass(f"batch_{device}_overwrite", "overwrite+verify OK")
    except Exception as e:
        _fail("batch_npu", e)
    finally:
        await ts.shutdown()


# =====================================================================
# Test 3: SharedMemory with NPU tensor (mirrors test_shared_memory.py GPU class)
# =====================================================================

async def test_shm_npu():
    """SharedMemory transport: NPU tensor → shared memory copy."""
    _header("Test 3: SharedMemory NPU tensor handling")
    try:
        from torchstore.transport.shared_memory import (
            allocate_shared_tensor,
            SharedMemoryDescriptor,
            pin_memory,
            unpin_memory,
        )

        npu_tensor = torch.randn(50, 50, device="npu:0")
        cpu_copy = npu_tensor.cpu()

        shm_tensor = allocate_shared_tensor(npu_tensor.shape, npu_tensor.dtype)
        _pass("alloc_shm", f"shared tensor shape={shm_tensor.shape}")

        descriptor = SharedMemoryDescriptor.from_tensor(shm_tensor)
        assert descriptor is not None
        _pass("shm_descriptor", "descriptor created from shared tensor")

        shm_tensor.copy_(cpu_copy)
        assert torch.allclose(shm_tensor, cpu_copy, atol=1e-6)
        _pass("npu_to_shm_copy", "NPU→CPU→SharedMemory data correct")

        pin_memory(shm_tensor)
        _pass("pin_memory", "no crash on NPU host (skipped internally)")

        unpin_memory(shm_tensor)
        _pass("unpin_memory", "no crash on NPU host")

        entry = descriptor.attach()
        attached_tensor = entry.get_tensor()
        assert torch.allclose(attached_tensor, cpu_copy, atol=1e-6)
        _pass("shm_attach_read", "attached tensor matches original")

    except Exception as e:
        _fail("shm_npu", e)


# =====================================================================
# Test 4: Large tensor through RDMA (mirrors test_large_tensors.py)
# =====================================================================

async def test_large_tensor_rdma():
    """Large tensor put/get through default transport."""
    _header("Test 4: Large tensor transfer")

    class LargeTensorActor(Actor):
        def __init__(self):
            pass

        @endpoint
        async def put_and_get(self):
            sizes = [(1024, 1024), (1024, 2048)]
            for i, shape in enumerate(sizes):
                t = torch.randn(shape, dtype=torch.float32)
                size_mb = t.numel() * 4 / (1024 * 1024)
                await ts.put(f"large_{i}", t)

            results = []
            for i, shape in enumerate(sizes):
                got = await ts.get(f"large_{i}")
                assert got.shape == torch.Size(shape), f"shape mismatch: {got.shape}"
                results.append(got.numel() * 4 / (1024 * 1024))
            return results

    try:
        await ts.initialize()
        actor = await spawn_actors(1, LargeTensorActor, "large_tensor")

        t0 = time.perf_counter()
        sizes_mb = await actor.put_and_get.call_one()
        elapsed = time.perf_counter() - t0
        _pass("large_put_get", f"{[f'{s:.0f}MB' for s in sizes_mb]} in {elapsed:.2f}s")
    except Exception as e:
        _fail("large_tensor", e)
    finally:
        await ts.shutdown()


# =====================================================================
# Test 5: Object put/get (mirrors test_store.py:test_objects)
# =====================================================================

async def test_object_put_get():
    """Put/get non-tensor objects through TorchStore."""
    _header("Test 5: Object put/get")

    class ObjectActor(Actor):
        def __init__(self, world_size):
            self.world_size = world_size
            self.rank = current_rank().rank
            os.environ["LOCAL_RANK"] = str(self.rank)

        @endpoint
        async def put(self, obj):
            await ts.put(f"obj_{self.rank}", obj)

        @endpoint
        async def get(self, rank_offset=0):
            other_rank = (self.rank + rank_offset) % self.world_size
            return await ts.get(f"obj_{other_rank}")

    try:
        await ts.initialize(
            num_storage_volumes=2,
            strategy=ts.LocalRankStrategy(TransportType.MonarchRPC),
        )
        actors = await spawn_actors(2, ObjectActor, "obj_actors", world_size=2)

        for idx in range(2):
            actor = actors.slice(npus=idx)
            await actor.put.call({"data": [idx, idx * 10], "name": f"rank_{idx}"})
        _pass("object_put", "2 objects stored")

        results = await actors.get.call()
        for pt, val in results:
            expected = {"data": [pt.rank, pt.rank * 10], "name": f"rank_{pt.rank}"}
            assert val == expected, f"{val} != {expected}"
        _pass("object_get", "objects retrieved correctly")
    except Exception as e:
        _fail("object_put_get", e)
    finally:
        await ts.shutdown()


# =====================================================================
# Test 6: Key existence & deletion (mirrors test_store.py)
# =====================================================================

async def test_keys_and_delete():
    """Test exists() and delete() APIs."""
    _header("Test 6: Key exists & delete")

    class KeyActor(Actor):
        def __init__(self):
            self.rank = current_rank().rank
            os.environ["LOCAL_RANK"] = str(self.rank)

        @endpoint
        async def test_lifecycle(self):
            key = f"lifecycle_{self.rank}"
            tensor = torch.tensor([1, 2, 3, 4, 5])

            assert not await ts.exists(key), "key should not exist yet"
            await ts.put(key, tensor)
            assert await ts.exists(key), "key should exist after put"

            retrieved = await ts.get(key)
            assert torch.equal(tensor, retrieved)

            await ts.delete(key)
            assert not await ts.exists(key), "key should not exist after delete"

            return "ok"

    try:
        await ts.initialize(
            num_storage_volumes=1,
            strategy=ts.LocalRankStrategy(TransportType.MonarchRPC),
        )
        actors = await spawn_actors(1, KeyActor, "key_actors")

        result = await actors.test_lifecycle.call_one()
        assert result == "ok"
        _pass("key_lifecycle", "put → exists → get → delete → !exists")
    except Exception as e:
        _fail("keys_and_delete", e)
    finally:
        await ts.shutdown()


# =====================================================================
# Test 7: State dict put/get (mirrors test_state_dict.py)
# =====================================================================

async def test_state_dict():
    """Put/get a model state_dict through TorchStore."""
    _header("Test 7: State dict put/get")

    class TrainerActor(Actor):
        def __init__(self):
            self.rank = current_rank().rank
            os.environ["LOCAL_RANK"] = str(self.rank)

        @endpoint
        async def do_test(self):
            import torch.nn as nn
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            for _ in range(3):
                optimizer.zero_grad()
                loss = model(torch.randn(4, 10)).sum()
                loss.backward()
                optimizer.step()

            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            await ts.put_state_dict(state_dict, "checkpoint_v0")

            fetched = await ts.get_state_dict("checkpoint_v0")
            return state_dict, fetched

    try:
        await ts.initialize(
            num_storage_volumes=1,
            strategy=ts.LocalRankStrategy(TransportType.MonarchRPC),
        )
        trainer = await spawn_actors(1, TrainerActor, "trainer")
        original, fetched = await trainer.do_test.call_one()

        from torch.distributed.checkpoint._nested_dict import flatten_state_dict
        flat_orig, _ = flatten_state_dict(original)
        flat_fetched, _ = flatten_state_dict(fetched)
        assert len(flat_orig) == len(flat_fetched)
        for key in flat_orig:
            assert key in flat_fetched, f"missing key: {key}"
            if isinstance(flat_orig[key], torch.Tensor):
                assert torch.equal(flat_orig[key], flat_fetched[key]), f"mismatch: {key}"
            else:
                assert flat_orig[key] == flat_fetched[key], f"mismatch: {key}"
        _pass("state_dict_roundtrip", f"{len(flat_orig)} keys matched")
    except Exception as e:
        _fail("state_dict", e)
    finally:
        await ts.shutdown()


# =====================================================================
# Test 8: DirectWeightSync with mock RDMA (mirrors test_direct_weight_sync.py)
# =====================================================================

async def test_direct_weight_sync():
    """DirectWeightSync correctness with MockRDMABuffer (no real RDMA needed)."""
    _header("Test 8: DirectWeightSync (mock RDMA)")
    try:
        from torchstore.direct_weight_sync import (
            DirectWeightSyncDest,
            RDMAWeightHandle,
        )
        from torchstore.transport.types import TensorSlice
        from torchstore.utils import to_byte_view

        class MockRDMABuffer:
            def __init__(self, source_bytes):
                self._source = source_bytes
            async def read_into(self, dest_byte_view):
                dest_byte_view.copy_(self._source)
            async def drop(self):
                pass

        # Test exact match (zero-copy path)
        original = torch.arange(256, dtype=torch.float32).reshape(16, 16)
        ts_slice = TensorSlice(
            offsets=(0, 0), coordinates=(0,),
            global_shape=(16, 16), local_shape=(16, 16), mesh_shape=(1,),
        )
        buf = MockRDMABuffer(to_byte_view(original.contiguous()))
        handle = RDMAWeightHandle(rdma_buffer=buf, tensor_slice=ts_slice, source_rank=0)

        dest = torch.zeros_like(original)
        sync = DirectWeightSyncDest()
        await sync.pull({"weight": [handle]}, {"weight": dest})
        assert torch.equal(dest, original)
        _pass("exact_match", "zero-copy path OK")

        # Test resharding (2 shards → full tensor)
        shard0 = original[:8, :].contiguous()
        shard1 = original[8:, :].contiguous()
        handles = []
        for i, (shard, off) in enumerate([(shard0, 0), (shard1, 8)]):
            s = TensorSlice(
                offsets=(off, 0), coordinates=(i,),
                global_shape=(16, 16), local_shape=(8, 16), mesh_shape=(2,),
            )
            handles.append(RDMAWeightHandle(
                rdma_buffer=MockRDMABuffer(to_byte_view(shard)),
                tensor_slice=s, source_rank=i,
            ))
        dest2 = torch.zeros_like(original)
        sync2 = DirectWeightSyncDest()
        await sync2.pull({"weight": handles}, {"weight": dest2})
        assert torch.equal(dest2, original)
        _pass("resharding_2shard", "2-shard reassembly OK")

        # Test multiple params
        w1 = torch.randn(10, 10)
        w2 = torch.randn(5, 5)
        def make_handle(t):
            s = TensorSlice(
                offsets=tuple(0 for _ in t.shape), coordinates=(0,),
                global_shape=tuple(t.shape), local_shape=tuple(t.shape), mesh_shape=(1,),
            )
            return RDMAWeightHandle(
                rdma_buffer=MockRDMABuffer(to_byte_view(t.contiguous())),
                tensor_slice=s, source_rank=0,
            )
        dest_sd = {"w1": torch.zeros_like(w1), "w2": torch.zeros_like(w2)}
        sync3 = DirectWeightSyncDest()
        await sync3.pull({"w1": [make_handle(w1)], "w2": [make_handle(w2)]}, dest_sd)
        assert torch.equal(dest_sd["w1"], w1) and torch.equal(dest_sd["w2"], w2)
        _pass("multi_param", "2 params synced correctly")

    except Exception as e:
        _fail("direct_weight_sync", e)


# =====================================================================
# Test 9: Gloo transport (verify alternative transport on NPU)
# =====================================================================

async def test_gloo_transport():
    """Put/get using Gloo transport explicitly."""
    _header("Test 9: Gloo transport")

    import torch.distributed as dist
    if not dist.is_gloo_available():
        _skip("gloo_transport", "Gloo not available")
        return

    class GlooActor(Actor):
        def __init__(self):
            self.rank = current_rank().rank
            os.environ["LOCAL_RANK"] = str(self.rank)
            os.environ["HOSTNAME"] = socket.gethostname()

        @endpoint
        async def put_get(self):
            key = "gloo_test"
            t = torch.randn(32, 32)
            await ts.put(key, t)
            got = await ts.get(key)
            assert torch.equal(t, got), "Gloo put/get mismatch"
            return True

    try:
        await ts.initialize(
            num_storage_volumes=1,
            strategy=ts.LocalRankStrategy(TransportType.Gloo),
        )
        actor = await spawn_actors(1, GlooActor, "gloo_actor")
        ok = await actor.put_get.call_one()
        assert ok
        _pass("gloo_put_get", "Gloo transport put/get OK")
    except Exception as e:
        _fail("gloo_transport", e)
    finally:
        await ts.shutdown()


# =====================================================================
# Test 10: NPU tensor D2H in storage (verify device tensor handling)
# =====================================================================

async def test_npu_tensor_store():
    """Store tensors computed on NPU, verify CPU-converted roundtrip through store.

    Note: torchstore serializes via torch.save which doesn't handle raw NPU tensors.
    The correct pattern is .cpu() before put — this test verifies that workflow.
    """
    _header("Test 10: NPU tensor store & retrieve")

    class NPUStoreActor(Actor):
        def __init__(self):
            self.rank = current_rank().rank
            os.environ["LOCAL_RANK"] = str(self.rank)

        @endpoint
        async def test_npu_tensor(self):
            import torch_npu  # noqa: F401
            npu_t = torch.randn(64, 64, device="npu:0")
            cpu_ref = npu_t.cpu().clone()

            await ts.put("npu_tensor_cpu", cpu_ref)
            got = await ts.get("npu_tensor_cpu")

            assert got.device.type == "cpu", f"expected CPU, got {got.device}"
            assert torch.equal(got, cpu_ref), "data mismatch after roundtrip"

            npu_restored = got.to("npu:0")
            assert torch.equal(npu_restored, npu_t), "NPU restore mismatch"
            return True

    try:
        await ts.initialize(
            num_storage_volumes=1,
            strategy=ts.LocalRankStrategy(TransportType.MonarchRPC),
        )
        actor = await spawn_actors(1, NPUStoreActor, "npu_store")
        ok = await actor.test_npu_tensor.call_one()
        assert ok
        _pass("npu_tensor_store", "NPU→CPU→store→CPU→NPU roundtrip correct")
    except Exception as e:
        _fail("npu_tensor_store", e)
    finally:
        await ts.shutdown()


# =====================================================================
# Test 11: Keys prefix query (mirrors test_keys.py)
# =====================================================================

async def test_keys_prefix():
    """Key prefix query functionality."""
    _header("Test 11: Keys prefix query")

    class KeysActor(Actor):
        def __init__(self):
            pass

        @endpoint
        async def test(self):
            await ts.put("v0.weight", torch.tensor([1.0]))
            await ts.put("v0.bias", torch.tensor([2.0]))
            await ts.put("v0.weight.grad", torch.tensor([3.0]))
            await ts.put("v1.weight", torch.tensor([4.0]))

            all_keys = await ts.keys()
            assert len(all_keys) == 4, f"expected 4 keys, got {len(all_keys)}"

            v0_keys = await ts.keys("v0")
            assert len(v0_keys) == 3, f"expected 3 v0 keys, got {len(v0_keys)}"
            for k in v0_keys:
                assert k.startswith("v0"), f"unexpected key: {k}"

            v1_keys = await ts.keys("v1")
            assert len(v1_keys) == 1
            return True

    try:
        await ts.initialize()
        actor = await spawn_actors(1, KeysActor, "keys_actor")
        ok = await actor.test.call_one()
        assert ok
        _pass("keys_prefix", "prefix query OK (4 total, 3 v0.*, 1 v1.*)")
    except Exception as e:
        _fail("keys_prefix", e)
    finally:
        await ts.shutdown()


# =====================================================================
# Test 12: MonarchRDMA transport (HiXL RoCE — real RDMA)
# =====================================================================

def _npu_bootstrap(dev_id: int, roce: bool):
    """Build a bootstrap function that pins a worker to a specific NPU
    and configures HiXL engine ID + optional RoCE."""
    def _bootstrap():
        os.environ["MONARCH_NPU_DEVICE"] = str(dev_id)
        if roce:
            os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
        else:
            os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)
        import torch
        import torch_npu  # noqa: F401
        torch.npu.set_device(dev_id)
        from monarch._src.rdma.hixl_transfer import compute_engine_id
        eid = compute_engine_id()
        os.environ["MONARCH_PYTHON_HIXL_ENGINE_ID"] = eid
    return _bootstrap


async def test_rdma_roce():
    """MonarchRDMA transport via HiXL RoCE: Producer on NPU 0, Consumer on NPU 1.
    Uses HCCL_INTRA_ROCE_ENABLE=1 to force RoCE fabric."""
    _header("Test 12: MonarchRDMA (HiXL RoCE)")

    if torch.npu.device_count() < 2:
        _skip("rdma_roce", "need >= 2 NPUs")
        return

    if not monarch_rdma_transport_available():
        _skip("rdma_roce", "Monarch RDMA not available")
        return

    from monarch.rdma import RDMABuffer

    class RDMAProducer(Actor):
        def __init__(self):
            self.tensor = torch.arange(256, dtype=torch.float32, device="npu").reshape(16, 16)
            torch.npu.synchronize()
            self.buf = None

        @endpoint
        async def get_handle(self) -> RDMABuffer:
            if self.buf is None:
                byte_view = self.tensor.view(torch.uint8).flatten()
                self.buf = RDMABuffer(byte_view)
            return self.buf

        @endpoint
        async def get_sum(self) -> float:
            return self.tensor.sum().cpu().item()

    class RDMAConsumer(Actor):
        @endpoint
        async def write_ones(self, remote: RDMABuffer) -> float:
            local = torch.ones(16, 16, dtype=torch.float32, device="npu")
            torch.npu.synchronize()
            await remote.write_from(local.view(torch.uint8).flatten(), timeout=30)
            return local.sum().cpu().item()

        @endpoint
        async def read_remote(self, remote: RDMABuffer) -> float:
            local = torch.zeros(16, 16, dtype=torch.float32, device="npu")
            torch.npu.synchronize()
            await remote.read_into(local.view(torch.uint8).flatten(), timeout=30)
            torch.npu.synchronize()
            return local.sum().cpu().item()

    try:
        host = this_host()
        prod_mesh = host.spawn_procs(per_host={"gpus": 1}, bootstrap=_npu_bootstrap(0, roce=True))
        cons_mesh = host.spawn_procs(per_host={"gpus": 1}, bootstrap=_npu_bootstrap(1, roce=True))

        producer = prod_mesh.spawn("rdma_producer", RDMAProducer)
        consumer = cons_mesh.spawn("rdma_consumer", RDMAConsumer)

        handle = await producer.get_handle.call_one()
        orig_sum = await producer.get_sum.call_one()
        _pass("rdma_roce_handle", f"handle created, producer sum={orig_sum}")

        await asyncio.sleep(2)

        wrote = await consumer.write_ones.call_one(handle)
        after_write = await producer.get_sum.call_one()
        assert abs(after_write - 256.0) < 1e-3, f"write mismatch: expected 256, got {after_write}"
        _pass("rdma_roce_write", f"write_from OK (producer sum: {orig_sum} → {after_write})")

        pulled = await consumer.read_remote.call_one(handle)
        assert abs(pulled - after_write) < 1e-3, f"read mismatch: expected {after_write}, got {pulled}"
        _pass("rdma_roce_read", f"read_into OK (consumer got sum={pulled})")

    except Exception as e:
        _fail("rdma_roce", e)


# =====================================================================
# Test 13: MonarchRDMA transport (HiXL HCCS — intra-node D2D)
# =====================================================================

async def test_hccs():
    """MonarchRDMA transport via HiXL HCCS: Producer on NPU 0, Consumer on NPU 1.
    HCCL_INTRA_ROCE_ENABLE is NOT set, so HiXL defaults to HCCS for D2D."""
    _header("Test 13: MonarchRDMA (HiXL HCCS)")

    if torch.npu.device_count() < 2:
        _skip("hccs", "need >= 2 NPUs")
        return

    if not monarch_rdma_transport_available():
        _skip("hccs", "Monarch RDMA not available")
        return

    from monarch.rdma import RDMABuffer

    class HCCSProducer(Actor):
        def __init__(self):
            self.tensor = torch.arange(64, dtype=torch.float32, device="npu").reshape(8, 8)
            torch.npu.synchronize()
            self.buf = None

        @endpoint
        async def get_handle(self) -> RDMABuffer:
            if self.buf is None:
                byte_view = self.tensor.view(torch.uint8).flatten()
                self.buf = RDMABuffer(byte_view)
            return self.buf

        @endpoint
        async def get_sum(self) -> float:
            return self.tensor.sum().cpu().item()

    class HCCSConsumer(Actor):
        @endpoint
        async def write_twos(self, remote: RDMABuffer) -> float:
            local = torch.full((8, 8), 2.0, dtype=torch.float32, device="npu")
            torch.npu.synchronize()
            await remote.write_from(local.view(torch.uint8).flatten(), timeout=30)
            return local.sum().cpu().item()

        @endpoint
        async def read_remote(self, remote: RDMABuffer) -> float:
            local = torch.zeros(8, 8, dtype=torch.float32, device="npu")
            torch.npu.synchronize()
            await remote.read_into(local.view(torch.uint8).flatten(), timeout=30)
            torch.npu.synchronize()
            return local.sum().cpu().item()

    try:
        host = this_host()
        prod_mesh = host.spawn_procs(per_host={"gpus": 1}, bootstrap=_npu_bootstrap(0, roce=False))
        cons_mesh = host.spawn_procs(per_host={"gpus": 1}, bootstrap=_npu_bootstrap(1, roce=False))

        producer = prod_mesh.spawn("hccs_producer", HCCSProducer)
        consumer = cons_mesh.spawn("hccs_consumer", HCCSConsumer)

        handle = await producer.get_handle.call_one()
        orig_sum = await producer.get_sum.call_one()
        _pass("hccs_handle", f"handle created, producer sum={orig_sum}")

        await asyncio.sleep(2)

        wrote = await consumer.write_twos.call_one(handle)
        after_write = await producer.get_sum.call_one()
        assert abs(after_write - 128.0) < 1e-3, f"write mismatch: expected 128 (8*8*2), got {after_write}"
        _pass("hccs_write", f"write_from OK (producer sum: {orig_sum} → {after_write})")

        pulled = await consumer.read_remote.call_one(handle)
        assert abs(pulled - after_write) < 1e-3, f"read mismatch: expected {after_write}, got {pulled}"
        _pass("hccs_read", f"read_into OK (consumer got sum={pulled})")

    except Exception as e:
        _fail("hccs", e)


# =====================================================================
# Test 14: TorchStore E2E via MonarchRDMA transport
# =====================================================================

async def test_torchstore_rdma_transport():
    """TorchStore-style RDMA lifecycle with NPU tensors across two devices.

    IMPORTANT NPU LIMITATION:
      HiXL HCCS requires 2MB-aligned memory. CPU tensors from malloc() are NOT
      2MB-aligned, so TorchStore's default MONARCH_RDMA_EAGER_D2H=1 path (which
      converts to CPU before RDMABuffer) will fail with HCCS.
      Workarounds:  (a) keep tensors on NPU for RDMA
                    (b) use alloc_aligned_tensor() for CPU buffers
                    (c) set TORCHSTORE_MONARCH_RDMA_EAGER_D2H=0

    This test verifies the NPU-resident tensor RDMA lifecycle that TorchStore's
    MonarchRDMATransportBuffer would use with EAGER_D2H disabled.
    """
    _header("Test 14: TorchStore RDMA lifecycle (NPU-resident)")

    if torch.npu.device_count() < 2:
        _skip("ts_rdma", "need >= 2 NPUs")
        return
    if not monarch_rdma_transport_available():
        _skip("ts_rdma", "Monarch RDMA not available")
        return

    from monarch.rdma import RDMABuffer
    from torchstore.utils import to_byte_view

    class TSRDMASource(Actor):
        """Simulates put-side: single 2MB-aligned buffer, shared across operations.
        NPU allocators reuse pages, so only the FIRST allocation per process
        is guaranteed 2MB-aligned. We create one large buffer in __init__."""
        def __init__(self):
            self.tensor = torch.arange(512 * 512, dtype=torch.float32, device="npu").reshape(512, 512)
            torch.npu.synchronize()
            self.buf = None

        @endpoint
        async def expose(self) -> tuple:
            if self.buf is None:
                self.buf = RDMABuffer(self.tensor.view(torch.uint8).flatten())
            return self.buf, self.tensor.sum().cpu().item()

        @endpoint
        async def get_sum(self) -> float:
            torch.npu.synchronize()
            return self.tensor.sum().cpu().item()

    class TSRDMASink(Actor):
        """Simulates get-side: reads/writes RDMABuffer across devices."""
        @endpoint
        async def read_all(self, remote: RDMABuffer) -> float:
            local = torch.zeros(512, 512, dtype=torch.float32, device="npu")
            torch.npu.synchronize()
            await remote.read_into(local.view(torch.uint8).flatten(), timeout=30)
            torch.npu.synchronize()
            return local.sum().cpu().item()

        @endpoint
        async def write_pattern(self, remote: RDMABuffer) -> bool:
            payload = torch.ones(512, 512, dtype=torch.float32, device="npu")
            torch.npu.synchronize()
            await remote.write_from(payload.view(torch.uint8).flatten(), timeout=30)
            return True

    try:
        host = this_host()
        src_mesh = host.spawn_procs(per_host={"gpus": 1}, bootstrap=_npu_bootstrap(0, roce=True))
        sink_mesh = host.spawn_procs(per_host={"gpus": 1}, bootstrap=_npu_bootstrap(1, roce=True))

        source = src_mesh.spawn("ts_source", TSRDMASource)
        sink = sink_mesh.spawn("ts_sink", TSRDMASink)

        await asyncio.sleep(2)

        # 1) Read 1MB tensor across devices (source → sink)
        buf, expected = await source.expose.call_one()
        got = await sink.read_all.call_one(buf)
        assert abs(got - expected) < 1.0, f"read: expected {expected:.0f}, got {got:.0f}"
        _pass("ts_rdma_read", f"1MB RDMA read OK (sum≈{got:.0f})")

        # 2) Write across devices (sink → source) + verify
        ok = await sink.write_pattern.call_one(buf)
        assert ok
        written = await source.get_sum.call_one()
        expected_written = 512.0 * 512.0  # all ones
        assert abs(written - expected_written) < 1e-2, f"write: expected {expected_written}, got {written}"
        _pass("ts_rdma_write", f"1MB RDMA write OK (sum={written:.0f})")

        # 3) Verify read again after write (data should now be all-ones)
        got2 = await sink.read_all.call_one(buf)
        assert abs(got2 - expected_written) < 1e-2, f"re-read: expected {expected_written}, got {got2}"
        _pass("ts_rdma_reread", f"post-write re-read OK (sum={got2:.0f})")

    except Exception as e:
        _fail("ts_rdma", e)


# =====================================================================
# Main
# =====================================================================

ALL_TESTS = {
    "basic": ("Basic put/get", test_basic_put_get),
    "batch": ("Batch put/get (NPU tensors)", test_batch_npu),
    "shm": ("SharedMemory NPU", test_shm_npu),
    "large": ("Large tensor transfer", test_large_tensor_rdma),
    "object": ("Object put/get", test_object_put_get),
    "keys": ("Key exists & delete", test_keys_and_delete),
    "state_dict": ("State dict roundtrip", test_state_dict),
    "dws": ("DirectWeightSync (mock)", test_direct_weight_sync),
    "gloo": ("Gloo transport", test_gloo_transport),
    "npu_store": ("NPU tensor store", test_npu_tensor_store),
    "keys_prefix": ("Keys prefix query", test_keys_prefix),
    "rdma": ("MonarchRDMA RoCE", test_rdma_roce),
    "hccs": ("MonarchRDMA HCCS", test_hccs),
    "ts_rdma": ("TorchStore via RDMA", test_torchstore_rdma_transport),
}


async def main():
    parser = argparse.ArgumentParser(description="TorchStore NPU E2E Tests")
    parser.add_argument(
        "--only",
        choices=list(ALL_TESTS.keys()) + ["all"],
        default="all",
        help="Run specific test group",
    )
    args = parser.parse_args()

    if not torch.npu.is_available():
        print("ERROR: NPU not available")
        sys.exit(1)

    print(f"NPU count: {torch.npu.device_count()}")
    print(f"RDMA available: {monarch_rdma_transport_available()}")

    tests = ALL_TESTS if args.only == "all" else {args.only: ALL_TESTS[args.only]}

    for name, (desc, func) in tests.items():
        try:
            await func()
        except Exception as e:
            _fail(f"{name}_unexpected", e)

    print(f"\n{'='*60}")
    total = _passed + _failed + _skipped
    print(f"Results: {_passed} PASS / {_failed} FAIL / {_skipped} SKIP (total {total})")
    print(f"{'='*60}")
    return 1 if _failed > 0 else 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
