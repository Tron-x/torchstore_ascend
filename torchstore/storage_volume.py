# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
from logging import getLogger
from typing import Any

import torch
from monarch.actor import Actor, endpoint

from torchstore.transport.buffers import TransportBuffer, TransportContext
from torchstore.transport.types import Request, TensorSlice
from torchstore.utils import get_slice_intersection, spawn_actors

logger = getLogger(__name__)


FULL_TENSOR = "full_tensor"


class StorageVolume(Actor):
    """The remote logic for storage. Recieves remote put/get requests and handles them via the storage abstraction"""

    actor_name: str = "StorageVolumes"

    def __init__(
        self,
        id_func,
    ) -> None:
        self.store: StorageImpl = InMemoryStore()
        self.volume_id: str = id_func()
        # Optional collective-broadcast process group. Installed lazily
        # via ``init_bcast_group`` when a consumer (e.g. forge's
        # ``CollectiveBroadcastBackend``) wants this volume to reflect
        # its locally-stored tensors out to a set of remote peers over
        # HCCL/NCCL instead of them each doing their own ``ts.get``.
        # The volume still accepts regular ``put``/``get`` traffic on
        # its MonarchRDMA transport side; the bcast group sits on a
        # separate ``torch.distributed`` backend (HCCL on Ascend NPU,
        # NCCL on CUDA).  See
        # ``forge/docs/weight_sync.md §7.4`` for the design rationale.
        self._bcast_pg = None  # type: ignore[var-annotated]

    @classmethod
    async def spawn(
        cls,
        num_volumes: int,
        mesh,
        *init_args: Any,
        **init_kwargs: Any,
    ) -> "StorageVolume":
        actors = await spawn_actors(
            num_volumes, cls, cls.actor_name, mesh, *init_args, **init_kwargs
        )

        return actors

    @endpoint
    async def get_id(self) -> tuple[str, str]:
        hostname = os.environ.get("HOSTNAME", socket.gethostname())
        return (self.volume_id, hostname)

    @endpoint
    async def handshake(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> list[Any]:
        return await self.store.handshake(transport_buffer, requests)

    @endpoint
    async def put(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> None:
        await self.store.put(transport_buffer, requests)

    @endpoint
    async def get(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> TransportBuffer:
        return await self.store.get(transport_buffer, requests)

    @endpoint
    async def get_meta(
        self,
        requests: list[Request],
    ) -> list[tuple[torch.Size, torch.dtype] | str]:
        return await self.store.get_meta(requests)

    @endpoint
    async def delete(self, key: str) -> None:
        await self.store.delete(key)

    @endpoint
    async def reset(self) -> None:
        self.store.reset()

    # ------------------------------------------------------------------
    # Collective-broadcast reflector endpoints
    # ------------------------------------------------------------------
    #
    # The three endpoints below let an external consumer (forge's
    # ``CollectiveBroadcastBackend``) turn a storage volume into a
    # broadcast source: after the trainer has ``ts.put``-ed a tensor
    # into this volume's ``InMemoryStore``, the consumer calls
    # ``init_bcast_group`` to rendezvous this volume with the target
    # peers (e.g. inference TP workers), then ``bcast_tensor(key)`` to
    # broadcast the locally-stored tensor over HCCL/NCCL to those
    # peers in a single collective operation.
    #
    # This is strictly additive -- volumes that never call
    # ``init_bcast_group`` behave exactly as before.  Volumes that do
    # still accept regular ``put`` / ``get`` traffic concurrently with
    # any bcast work, because the two transports live on separate
    # backends (torchstore's MonarchRDMA vs stock torch.distributed
    # HCCL/NCCL).
    @endpoint
    async def init_bcast_group(
        self,
        master_addr: str,
        master_port: int,
        world_size: int,
        rank: int,
        backend: str = "hccl",
        group_name: str = "torchstore_bcast_reflector",
        timeout_s: int = 180,
    ) -> dict:
        """Rendezvous this volume into a ``torch.distributed`` process
        group so later calls to :meth:`bcast_tensor` broadcast from
        this volume to the other ranks in the group.

        Only one bcast group per volume is supported; calling twice is
        a no-op on the second call.
        """
        import datetime

        import torch.distributed as dist

        if self._bcast_pg is not None:
            return {
                "already_initialized": True,
                "rank": rank,
                "world_size": world_size,
            }

        # We use the default-PG init path (instead of
        # ``_new_process_group_helper`` gymnastics) because storage
        # volumes are plain Monarch actors that don't have a pre-existing
        # default PG.  First volume to hit this wins the default slot;
        # subsequent groups in the same proc would need ``new_group``,
        # which we'll add if the design ever needs multiple bcast
        # groups per volume.
        # Both legs of the bcast group (storage vol here, TP workers in
        # forge's WorkerWrapper) must use the SAME ``group_name`` so
        # torch.distributed's PrefixStore scoping puts their
        # ``hcclUniqueId`` rendezvous keys under a matching prefix.
        # The worker side already uses a custom (non-default) PG because
        # vLLM has a default PG; for symmetry we do the same on the
        # storage side so the prefix is consistent across both ranks.
        # We vendor ``init_custom_process_group`` from AReaL (not
        # torchstore's normal dep) only if available; otherwise fall
        # back to default init_process_group, which works as long as no
        # other PG exists in the proc.
        try:
            from areal.engine.core.distributed import init_custom_process_group

            self._bcast_pg = init_custom_process_group(
                backend=backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
                timeout=datetime.timedelta(seconds=timeout_s),
            )
        except ImportError:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_addr}:{master_port}",
                    world_size=world_size,
                    rank=rank,
                    timeout=datetime.timedelta(seconds=timeout_s),
                )
            self._bcast_pg = dist.group.WORLD

        # Run a tiny all_reduce so the HCCL / NCCL communicator is
        # actually constructed here, not only on the first real bcast.
        # Without this the worker-side rank-0 lookup races the first
        # real collective and can time out waiting for the
        # ``hcclUniqueId`` to appear in the TCPStore rendezvous --
        # torch.distributed only eagerly builds the communicator on
        # the first collective call on each rank.
        import torch

        # Storage volumes have exactly one visible NPU (they're
        # bootstrapped with ASCEND_RT_VISIBLE_DEVICES masking), so
        # logical device 0 always refers to "our NPU".  For CUDA
        # volumes the same single-visible-device invariant holds.
        if backend == "hccl" and hasattr(torch, "npu") and torch.npu.is_available():
            device = torch.device("npu", 0)
        elif backend == "nccl" and torch.cuda.is_available():
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")
        probe = torch.ones(8, dtype=torch.float32, device=device)
        dist.all_reduce(probe, group=self._bcast_pg)
        if device.type == "npu":
            torch.npu.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        return {
            "rank": rank,
            "world_size": world_size,
            "probe_sum": float(probe.sum().item()),
        }

    @endpoint
    async def bcast_tensor(self, key: str) -> dict:
        """Broadcast the tensor stored under ``key`` on this volume to
        every peer in the bcast group.  Must be called after
        :meth:`init_bcast_group` and must be matched by a
        ``dist.broadcast`` call on every peer rank with ``src=<this
        volume's rank>``.
        """
        import time

        import torch
        import torch.distributed as dist

        if self._bcast_pg is None:
            raise RuntimeError(
                "bcast_tensor called before init_bcast_group; "
                "the bcast group must be set up first"
            )
        # ``InMemoryStore`` keys either a Tensor (regular case), a
        # ``{"obj": ...}`` wrapper (ts.put on non-tensor), or a
        # coord-indexed dict (DTensor shards).  We only support plain
        # tensor values here -- forge's flat-buffer path stores exactly
        # that shape.  Error clearly if the caller asks us to bcast
        # something we can't.
        store = self.store
        if not hasattr(store, "kv"):
            raise RuntimeError(
                "bcast_tensor only supports InMemoryStore-backed volumes"
            )
        val = store.kv.get(key)
        if val is None:
            raise KeyError(
                f"key '{key}' not found on volume {self.volume_id}; "
                f"caller should ensure ts.put landed before bcast"
            )
        if not isinstance(val, torch.Tensor):
            raise TypeError(
                f"key '{key}' maps to {type(val).__name__}, not Tensor; "
                "bcast_tensor cannot broadcast non-tensor storage entries"
            )

        # NOTE: ``dist.get_rank(pg)`` hits a default-PG check on
        # torch_npu's build even when called with an explicit group
        # handle (the wrapper short-circuits to ``_get_default_group``).
        # Since volumes in this design always occupy rank 0 of the
        # bcast group (set by the caller in ``init_bcast_group``), it's
        # sufficient -- and safer -- to broadcast from ``src=0``.  If
        # this ever needs to be the non-0 rank, track the rank in an
        # instance variable set during ``init_bcast_group``.
        t0 = time.perf_counter()
        dist.broadcast(val, src=0, group=self._bcast_pg)
        if val.is_cuda or (hasattr(torch, "npu") and val.device.type == "npu"):
            # Synchronize so the caller's timing reflects completion,
            # not just queueing onto the stream.
            if val.device.type == "npu":
                torch.npu.synchronize()
            else:
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return {
            "bytes": int(val.numel() * val.element_size()),
            "bcast_s": elapsed,
            # Volumes are always rank 0 in this design (see note on the
            # broadcast call above); avoiding ``dist.get_rank(pg)`` here
            # sidesteps torch_npu's default-PG check.
            "src_rank": 0,
        }

    @endpoint
    async def shutdown_bcast_group(self) -> dict:
        import torch.distributed as dist

        if self._bcast_pg is None:
            return {"ok": True, "was_initialized": False}
        # Destroying the default PG invalidates future default-PG
        # collectives in this proc.  That's fine -- storage volumes
        # don't use torch.distributed for anything else.
        if dist.is_initialized():
            dist.destroy_process_group()
        self._bcast_pg = None
        return {"ok": True, "was_initialized": True}


class StorageImpl:
    """Abstract base class for storage implementations."""

    def __init__(self) -> None:
        self.transport_context = TransportContext()

    async def put(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> None:
        """Store data in the storage backend."""
        raise NotImplementedError()

    async def get(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> TransportBuffer:
        """Retrieve data from the storage backend."""
        raise NotImplementedError()

    async def get_meta(
        self, requests: list[Request]
    ) -> list[tuple[torch.Size, torch.dtype] | str]:
        """Get metadata about stored data."""
        raise NotImplementedError()

    async def delete(self, key: str) -> None:
        """Delete data from the storage backend."""
        raise NotImplementedError()

    async def handshake(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> list[Any]:
        raise NotImplementedError()


class InMemoryStore(StorageImpl):
    """Local in memory storage."""

    def __init__(self) -> None:
        self.kv: dict[str, Any] = {}
        super().__init__()

    async def handshake(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> list[Any]:
        pairs = [(request, self._extract_existing(request)) for request in requests]
        return await transport_buffer.recv_handshake(self.transport_context, pairs)

    def _extract_existing(self, request: "Request") -> torch.Tensor | None:
        """Extract existing tensor from storage for in-place update.

        Looks up the key in kv storage and extracts the tensor if it exists.
        Only asserts on type mismatches between existing data and incoming request.

        Args:
            request: The incoming put request

        Returns:
            The existing tensor if found, None otherwise.

        Raises:
            AssertionError: If there's a type mismatch between existing data and request.
        """
        current_object = self.kv.get(request.key, None)

        if current_object is None:
            return None

        if isinstance(current_object, torch.Tensor):
            # Regular tensor - request must also be a regular tensor (no tensor_slice)
            assert (
                request.tensor_slice is None
            ), "Existing data is a regular tensor but incoming request has tensor_slice (DTensor)"
            return current_object

        if isinstance(current_object, dict):
            if "obj" in current_object:
                # Object dict - request must also be an object
                assert (
                    request.is_object
                ), "Existing data is an object but request.is_object is False"
                return None

            # DTensor shard dict - incoming request must also be a DTensor
            assert (
                request.tensor_slice is not None
            ), "Existing data is DTensor shards but incoming request has no tensor_slice"
            # Look up by coordinates
            shard = current_object.get(request.tensor_slice.coordinates)
            if shard is not None and "tensor" in shard:
                return shard["tensor"]
            # Coordinates don't match - new shard, return None to allocate new
            return None

        raise AssertionError(f"Unexpected current_object type: {type(current_object)}")

    def _handle_dtensor(
        self, key: str, tensor_slice: TensorSlice, tensor: torch.Tensor
    ) -> None:
        if key not in self.kv:
            self.kv[key] = {}

        self.kv[key][tensor_slice.coordinates] = {
            "slice": tensor_slice,
            "tensor": tensor,
        }

    def _extract_slice_from_tensor(
        self, tensor: torch.Tensor, tensor_slice: TensorSlice
    ) -> torch.Tensor:
        """Extract a slice from a full tensor.

        Args:
            tensor: The full stored tensor.
            tensor_slice: The slice specification to extract.

        Returns:
            The extracted tensor slice.
        """
        indices = []
        for dim in range(len(tensor_slice.global_shape)):
            start = tensor_slice.offsets[dim]
            end = start + tensor_slice.local_shape[dim]
            indices.append(slice(start, end))
        return tensor[tuple(indices)]

    def _get_sharded_tensor(self, request: Request) -> torch.Tensor | None:
        """
        Searches stored shards and returns one which completely contains the requested tensor slice

        Args:
            request: Request object containing the tensor_slice specification

        Returns:
            The extracted tensor slice if found completely within a stored shard,
            None otherwise.
        """
        for shard in self.kv[request.key].values():
            stored_slice = shard["slice"]
            stored_tensor = shard["tensor"]

            intersection_slice = get_slice_intersection(
                stored_slice, request.tensor_slice
            )

            # We don't want to visit the shard where requested tensor slice is not completely contained
            # in the stored tensor slice.
            if (
                intersection_slice is None
                or intersection_slice.local_shape != request.tensor_slice.local_shape
                or intersection_slice.offsets != request.tensor_slice.offsets
            ):
                continue

            # Extract the intersection from the stored tensor
            indices = []
            for dim in range(len(stored_slice.global_shape)):
                start = intersection_slice.offsets[dim] - stored_slice.offsets[dim]
                indices.append(
                    slice(
                        start,
                        start + intersection_slice.local_shape[dim],
                    )
                )
            extracted_tensor = stored_tensor[tuple(indices)]

            if extracted_tensor is not None:
                return extracted_tensor

    async def put(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> None:
        # Extract existing tensor for potential in-place update
        entries_with_current_obj = [
            (request, self._extract_existing(request)) for request in requests
        ]

        # fetch from remote
        results = await transport_buffer.handle_put_request(
            self.transport_context, entries_with_current_obj
        )

        # store locally
        for request, result in zip(requests, results, strict=True):
            self._store(request, result)

    def _store(self, request: "Request", data: Any) -> None:
        """Store data in kv, wrapping objects and handling DTensor shards."""
        key = request.key
        if request.is_object:
            self.kv[key] = {"obj": data}
            return

        if request.tensor_slice is not None:
            # tensor is actually part of a DTensor
            self._handle_dtensor(key, request.tensor_slice, data)
            return

        self.kv[key] = data

    async def get(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> TransportBuffer:
        data_entries = []
        for request in requests:
            if request.key not in self.kv:
                raise KeyError(
                    f"Key '{request.key}' not found. {list(self.kv.keys())=}"
                )
            data_entries.append((request, self._get_data(request)))
        await transport_buffer.handle_get_request(self.transport_context, data_entries)
        return transport_buffer

    def _get_data(self, request):
        key = request.key
        val = self.kv[key]
        if isinstance(val, dict) and "obj" in val:
            return val["obj"]

        # Full tensor stored - return it (possibly sliced)
        if isinstance(val, torch.Tensor):
            if request.tensor_slice is None:
                return val
            # User wants a slice of the full tensor - extract it
            return self._extract_slice_from_tensor(val, request.tensor_slice)

        # Must be sharded tensor dict - delegate to _get_sharded_tensor
        if request.tensor_slice is None:
            # TODO: currently, it seems we only support requested a subsection of a shard.
            # this needs to be made more general, such that we can request any region of the stored tensor
            # (with the most useful case really being even to fetch all shards at once, so we can recreate them locally)
            raise RuntimeError(
                f"Key '{key}' contains sharded tensor but no tensor_slice was requested"
            )

        extracted_tensor = self._get_sharded_tensor(request)

        if extracted_tensor is not None:
            return extracted_tensor

        raise RuntimeError(
            f"Tensor slice {request.tensor_slice} not found in any stored shards for {key}"
        )

    async def get_meta(
        self,
        requests: list[Request],
    ) -> list[tuple[torch.Size, torch.dtype] | str]:
        return [self._get_meta(request) for request in requests]

    def _get_meta(self, request: Request) -> tuple[torch.Size, torch.dtype] | str:
        key = request.key
        if key not in self.kv:
            raise KeyError(f"Key '{key}' not found. {list(self.kv.keys())=}")

        stored_object = self.kv[key]
        if isinstance(stored_object, torch.Tensor):
            return stored_object.shape, stored_object.dtype

        assert isinstance(stored_object, dict)
        if "obj" in stored_object:
            return "obj"

        if "tensor" in stored_object:
            return stored_object["tensor"].shape, stored_object["tensor"].dtype

        if request.tensor_slice is not None:
            extracted_tensor = self._get_sharded_tensor(request)
            if extracted_tensor is not None:
                return extracted_tensor.shape, extracted_tensor.dtype

            raise KeyError(
                f"Could not find shard slice with {request.tensor_slice=}  Slices:{stored_object}"
            )

        raise RuntimeError(
            f"Unknown type for {key} type={type(stored_object)} {stored_object=}"
        )

    async def delete(self, key: str) -> None:
        if key not in self.kv:
            raise KeyError(f"Key '{key}' not found. {list(self.kv.keys())=}")
        del self.kv[key]

    def reset(self) -> None:
        self.kv = {}
        self.transport_context.clear()
