# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import threading
from typing import Any, Optional, TYPE_CHECKING

import torch

try:
    from monarch.rdma import RDMABuffer

    try:
        # Preferred: rdma_supported detects ibverbs, HiXL, and TCP fallback
        from monarch._rust_bindings.rdma import rdma_supported as _rdma_backend_available
    except ImportError:
        try:
            # monarch >= 0.4.0
            from monarch.rdma import is_ibverbs_available as _rdma_backend_available
        except ImportError:
            # monarch < 0.4.0
            from monarch.rdma import is_rdma_available as _rdma_backend_available
except ImportError:
    _rdma_backend_available = lambda: False

    def RDMABuffer(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "RDMABuffer is not available. This environment was likely not built with rdma support."
        )


def monarch_rdma_available() -> bool:
    """Check if any Monarch RDMA backend is available (ibverbs, HiXL, or TCP fallback)."""
    return _rdma_backend_available()


from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.types import Request
from torchstore.utils import to_byte_view

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext

# For some reason, monarch sometimes doesn't like gpu tensors, so we convert to cpu. We noticed this in larger
# models, like qwen3-30BA3B
MONARCH_RDMA_EAGER_D2H = os.environ.get("TORCHSTORE_MONARCH_RDMA_EAGER_D2H", "1") == "1"

# HiXL / Monarch RDMABuffer.read_into/write_from take ``timeout`` — which the
# Python API documents as seconds but the Rust side (today) passes straight
# to HiXL as milliseconds. The default of 3 times out large transfers
# (e.g. 64 MB @ 20 GB/s needs ~3 ms on the wire alone). Give ourselves a much
# bigger envelope until the upstream bug is fixed.
_RDMA_TIMEOUT = int(os.environ.get("TORCHSTORE_MONARCH_RDMA_TIMEOUT", "60000"))


# Storage-side allocation device override.
#
# On NPU + HiXL RoCE, the RDMA transport requires 2 MB-aligned device memory
# on both sides of the transfer. torchstore defaults to allocating the
# destination / get-metadata tensor on CPU, which (a) is rarely 2 MB-aligned
# and (b) is not reachable by HiXL at all in RoCE mode, so the transfer hangs.
# Setting this env var to a real device (e.g. "npu:0") makes the storage side
# allocate on that device via alloc_aligned_tensor(), which is what HiXL
# requires. The value is interpreted by torch.device().
_STORAGE_DEVICE = os.environ.get("TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE", "")

# Staging pool size.  HiXL (CANN 9.x) rejects more than one registered
# memory region per engine pair (ret=503900).  To work around this we
# pre-register a single large NPU buffer via RDMABuffer on first use and
# allocate every subsequent transport buffer from inside it; the patched
# monarch Rust backend (``register_mem_if_needed`` with range-containment
# checks) then aliases those sub-slices to the pool registration and
# avoids a second ``hixl_register_mem`` call.  Set to 0 to disable.
_POOL_BYTES = int(os.environ.get("TORCHSTORE_MONARCH_RDMA_POOL_MB", "4096")) * 1024 * 1024
_POOL_ALIGN = 2 * 1024 * 1024  # HiXL HCCS alignment.


class _MonarchRDMAStagingPool:
    """Process-global staging pool for Monarch RDMA sub-allocations.

    The pool owns one 2 MB-aligned NPU ``uint8`` buffer and one
    long-lived ``RDMABuffer`` pinning it (the single "real" HiXL
    ``register_mem``).  Callers request aligned sub-slices via
    :meth:`alloc`, get tensor views via :meth:`view_tensor`, and return
    slots via :meth:`free`.  When a sub-slice is later wrapped in a
    fresh ``RDMABuffer``/used as a ``read_into``/``write_from`` dst, the
    patched Rust backend recognises it as contained in the pool's
    registration and aliases it instead of issuing a second HiXL
    ``register_mem``.

    Implementation is a simple segregated free-list with
    forward-coalescing on :meth:`free`; adequate for the typical
    torchstore traffic pattern (symmetric put/get of similarly-sized
    chunks).
    """

    def __init__(self, size_bytes: int, device: str = "npu:0") -> None:
        from monarch._src.rdma.xdma import alloc_aligned_tensor

        self._size = size_bytes
        self._device = device
        self._tensor, _ = alloc_aligned_tensor(
            (size_bytes,), dtype=torch.uint8, device=device
        )
        # Pin the registration.  This is the single real HiXL
        # ``register_mem`` call for the whole pool.
        self._pin_buffer = RDMABuffer(self._tensor)
        # Free list of (offset, size) entries, kept sorted by offset.
        self._free: list[tuple[int, int]] = [(0, size_bytes)]
        # Outstanding allocations: offset -> size, for size-lookup on free.
        self._live: dict[int, int] = {}
        self._lock = threading.Lock()
        logging.info(
            "[torchstore-monarch-rdma] staging pool: %.1f GiB on %s, base=%#x",
            size_bytes / (1024 ** 3), device, self._tensor.data_ptr(),
        )

    @property
    def size(self) -> int:
        return self._size

    @property
    def base_addr(self) -> int:
        return self._tensor.data_ptr()

    def alloc(self, nbytes: int, align: int = _POOL_ALIGN) -> int:
        """Allocate ``nbytes`` with ``align``-byte alignment, return offset."""
        with self._lock:
            for i, (off, sz) in enumerate(self._free):
                aligned_off = (off + align - 1) & ~(align - 1)
                pad = aligned_off - off
                if sz >= pad + nbytes:
                    # Take aligned chunk; keep any splinters.
                    leftover_after = sz - pad - nbytes
                    del self._free[i]
                    if leftover_after > 0:
                        self._free.insert(
                            i, (aligned_off + nbytes, leftover_after)
                        )
                    if pad > 0:
                        self._free.insert(i, (off, pad))
                    self._live[aligned_off] = nbytes
                    return aligned_off
            free_total = sum(s for _, s in self._free)
            raise RuntimeError(
                f"MonarchRDMAStagingPool: cannot allocate {nbytes} bytes "
                f"(pool size={self._size}, free total={free_total}, "
                f"set TORCHSTORE_MONARCH_RDMA_POOL_MB higher if needed)"
            )

    def free(self, offset: int) -> None:
        """Return a slot to the pool and coalesce with neighbours."""
        with self._lock:
            sz = self._live.pop(offset, None)
            if sz is None:
                return  # double-free or never-allocated; ignore.
            self._free.append((offset, sz))
            self._free.sort(key=lambda p: p[0])
            merged: list[tuple[int, int]] = []
            for off, s in self._free:
                if merged and merged[-1][0] + merged[-1][1] == off:
                    merged[-1] = (merged[-1][0], merged[-1][1] + s)
                else:
                    merged.append((off, s))
            self._free = merged

    def view_tensor(
        self, offset: int, shape: torch.Size | tuple, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return a typed tensor view into the pool at ``offset``."""
        numel = 1
        for d in shape:
            numel *= int(d)
        elem_size = torch.empty((), dtype=dtype).element_size()
        nbytes = numel * elem_size
        # uint8 byte slice → re-view as target dtype → reshape.
        byte_slice = self._tensor.narrow(0, offset, nbytes)
        return byte_slice.view(dtype).view(shape)


_GLOBAL_POOL: Optional[_MonarchRDMAStagingPool] = None
_POOL_LOCK = threading.Lock()


def _addr_in_pool(addr: int, pool: "_MonarchRDMAStagingPool") -> bool:
    base = pool.base_addr
    return base <= addr < base + pool.size


def _get_pool() -> Optional[_MonarchRDMAStagingPool]:
    """Lazily initialise the staging pool.  Returns ``None`` when the
    pool is disabled via env var or when ``_STORAGE_DEVICE`` is unset.
    """
    global _GLOBAL_POOL
    if _POOL_BYTES <= 0 or not _STORAGE_DEVICE:
        return None
    with _POOL_LOCK:
        if _GLOBAL_POOL is None:
            try:
                _GLOBAL_POOL = _MonarchRDMAStagingPool(
                    _POOL_BYTES, device=_STORAGE_DEVICE
                )
            except Exception:
                logging.exception(
                    "[torchstore-monarch-rdma] staging pool init failed; "
                    "falling back to per-transfer allocations"
                )
                _GLOBAL_POOL = None
        return _GLOBAL_POOL


def _empty_for_rdma(shape, dtype):
    """Allocate an RDMA-friendly tensor for the storage side.

    Falls back to plain ``torch.empty`` on CPU for the default behaviour so that
    GPU / vanilla paths are unchanged.
    """
    if not _STORAGE_DEVICE:
        return torch.empty(shape, dtype=dtype, device=torch.device("cpu"))

    pool = _get_pool()
    if pool is not None:
        numel = 1
        for d in shape:
            numel *= int(d)
        elem_size = torch.empty((), dtype=dtype).element_size()
        nbytes = numel * elem_size
        try:
            offset = pool.alloc(nbytes)
            # ``view_tensor`` returns a tensor view whose ``data_ptr()``
            # equals ``pool.base_addr + offset``; the TransportBuffer's
            # :meth:`drop` method computes the offset from that and
            # returns the slot to the pool.
            return pool.view_tensor(offset, shape, dtype)
        except Exception:
            logging.exception(
                "[torchstore-monarch-rdma] staging pool alloc failed; "
                "falling back to alloc_aligned_tensor"
            )
    try:
        from monarch._src.rdma.xdma import alloc_aligned_tensor

        tensor, _raw = alloc_aligned_tensor(shape, dtype=dtype, device=_STORAGE_DEVICE)
        return tensor
    except Exception:
        logging.exception(
            "alloc_aligned_tensor on %s failed, falling back to CPU", _STORAGE_DEVICE
        )
        return torch.empty(shape, dtype=dtype, device=torch.device("cpu"))


def monarch_rdma_transport_available() -> bool:
    """Check if Monarch RDMA transport is available for use.

    Returns True if:
    - TORCHSTORE_RDMA_ENABLED environment variable is set to "1" (default)
    - The monarch RDMA library is available and functional
    """
    rdma_enabled = os.environ.get("TORCHSTORE_RDMA_ENABLED", "1") == "1"
    return rdma_enabled and monarch_rdma_available()


class MonarchRDMATransportBuffer(TransportBuffer):
    def __init__(self, storage_volume_ref: "StorageVolumeRef"):
        super().__init__(storage_volume_ref)

        self.rdma_buffer: Any | None = None
        self.byte_view: torch.Tensor | None = None
        self.shape: torch.Size | None = None
        self.dtype: torch.dtype | None = None
        self.is_object: bool = False

    async def _pre_put_hook(self, requests: list[Request]) -> None:
        """Hook to perform any pre-put operations on the buffer."""
        assert len(requests) == 1
        request = requests[0]

        if request.is_object:
            return
        self.allocate(request.tensor_val)

    async def _pre_get_hook(self, requests: list[Request]) -> None:
        """Hook to perform any pre-get operations on the buffer."""
        assert len(requests) == 1
        request = requests[0]

        # keep request for later
        self.request = request

        # rdma buffer requires we have a pre-existing memory space locally
        # if the user has not provided a local tensor, we need to first
        # identify and allocate ahead of time
        meta = None
        if request.tensor_val is None:
            meta = (
                await self.storage_volume_ref.volume.get_meta.call_one(
                    [request.meta_only()]
                )
            )[0]
            if isinstance(meta, str) or meta is None:
                return  # objects don't get handled

            # if we are fetching a tensor slice, the local shape is already known
            if request.tensor_slice is not None:
                meta = (request.tensor_slice.local_shape, *meta[1:])

        self.allocate(meta or request.tensor_val)

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        assert len(entries) == 1
        request, current_object = entries[0]

        if request.is_object:
            self.is_object = True
            return [request.objects]

        # current_object is now the extracted tensor (or None)
        tensor = current_object

        if tensor is None:
            # happens when we haven't seen this tensor / dtensor before
            tensor = _empty_for_rdma(self.shape, self.dtype)

        self._assert_valid_tensor(tensor, self.dtype, self.shape)

        byte_view = to_byte_view(tensor)
        await self.rdma_buffer.read_into(byte_view, timeout=_RDMA_TIMEOUT)

        return [tensor]

    async def handle_get_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> None:
        assert len(entries) == 1
        _, data = entries[0]
        if not isinstance(data, torch.Tensor):
            self.is_object = True
            self.objects = data
            return

        tensor = data

        self._assert_valid_tensor(
            tensor, self.dtype, self.shape, must_be_contiguous=False
        )
        assert self.rdma_buffer is not None

        # Write directly from tensor byte view (no chunking)
        byte_view = to_byte_view(tensor)
        await self.rdma_buffer.write_from(byte_view, timeout=_RDMA_TIMEOUT)

    async def _handle_storage_volume_response(
        self, requests: list[Request], transport_buffer: TransportBuffer
    ) -> list[Any]:
        if transport_buffer.is_object:
            return [transport_buffer.objects]

        # If user provided an inplace tensor but we had to make a contiguous copy
        # during allocate(), the RDMA data landed in self.tensor (the copy), not
        # in the user's original tensor. We need to copy the data back.
        if self.request.tensor_val is not None:
            if self.request.tensor_val.data_ptr() != self.tensor.data_ptr():
                self.request.tensor_val.copy_(self.tensor)
            return [self.request.tensor_val]

        # self.byte_view already points to the byte view of self.tensor
        # so the data is already in self.tensor after the RDMA write completes
        return [self.tensor]

    async def drop(self) -> None:
        """Explicitly clean up RDMA buffers to prevent kernel memory leak.

        When RDMA buffers are created, they register memory regions with the RDMA
        hardware which pins pages in kernel memory. Without explicit cleanup, these
        pages remain pinned even after the Python objects are garbage collected,
        leading to a memory leak that manifests as unbounded Inactive(anon) growth.
        """
        if self.rdma_buffer is None:
            return
        try:
            await self.rdma_buffer.drop()
        except Exception as e:
            logging.warning(f"Failed to drop RDMA buffer during cleanup: {e}")
        self.rdma_buffer = None
        self.byte_view = None
        # If our ``self.tensor`` is a view into the staging pool, return
        # its slot.  We detect pool ownership by checking whether the
        # tensor's data_ptr falls inside the pool range.
        pool = _GLOBAL_POOL
        tensor = getattr(self, "tensor", None)
        if pool is not None and tensor is not None:
            try:
                base = pool.base_addr
                addr = tensor.data_ptr()
                if base <= addr < base + pool.size:
                    pool.free(addr - base)
            except Exception:
                logging.debug(
                    "[torchstore-monarch-rdma] pool.free attempt failed",
                    exc_info=True,
                )

    def __getstate__(self) -> dict[str, Any]:
        # Any time that we serialize the transport buffer, the idea is
        # that tensors will be transported via tensor_enginer.RDMABuffer, so it makes
        # no sense to hold this reference when we are serializing
        state = self.__dict__.copy()
        state["byte_view"] = None
        state["tensor"] = None
        state["request"] = None
        return state

    def allocate(self, tensor_like: torch.Tensor | tuple) -> None:
        """Allocates internal buffers based on either an existing tensor
        or a Tuple of (shape, dtype)
        """
        logging.debug("Allocating rdma buffer")

        if isinstance(tensor_like, tuple):
            # Happens only on get if we don't have an inplace tensor.
            # In that case, we know the size of the tensor from fetching metadata
            tensor = _empty_for_rdma(tensor_like[0], tensor_like[1])
        else:
            assert isinstance(tensor_like, torch.Tensor)

            # note: .contiguous will return a copy if this tensor is not contiguous
            # this usually shows up during resharding cases
            tensor = tensor_like.contiguous()

            # monarch sometimes really doesn't like gpu tensors, so we convert to cpu
            # this makes things way slower, and hopefully will be fixed in the future
            if MONARCH_RDMA_EAGER_D2H:
                tensor = tensor.cpu()

            # On HiXL (CANN 9.x) at most one registered memory region is
            # allowed per engine pair.  If the caller's tensor lives
            # outside the staging pool's range, wrapping it directly in
            # RDMABuffer would issue a second ``hixl_register_mem``
            # and subsequent ``TransferSync`` calls would fail with
            # ret=503900.  Copy such tensors into a pool slot so the
            # wrapped address is aliased (no new HiXL region).  User
            # tensors already inside the pool (e.g. from a previous
            # :meth:`allocate` call or from application-level staging)
            # are left alone to avoid a needless extra copy.
            pool = _get_pool()
            if (
                pool is not None
                and tensor.device.type != "cpu"
                and not _addr_in_pool(tensor.data_ptr(), pool)
            ):
                try:
                    nbytes = tensor.numel() * tensor.element_size()
                    offset = pool.alloc(nbytes)
                    staged = pool.view_tensor(offset, tensor.shape, tensor.dtype)
                    staged.copy_(tensor)
                    tensor = staged
                except Exception:
                    logging.exception(
                        "[torchstore-monarch-rdma] pool staging of source "
                        "tensor failed; falling back to direct registration "
                        "(may fail with HiXL ret=503900)"
                    )

        self.tensor = tensor

        # store tensor meta
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.dim = tensor.dim()

        self._assert_valid_tensor(tensor, self.dtype, self.shape)

        byte_view = to_byte_view(tensor)
        self.byte_view = byte_view
        self.rdma_buffer = RDMABuffer(byte_view)
