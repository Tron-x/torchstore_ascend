# TorchStore Ascend NPU 安装指南

本文档描述如何在华为 Ascend NPU 环境上安装和测试 TorchStore NPU 适配版本。

## 1. 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| 硬件 | Ascend 910B / 910C | 需要 ≥ 2 张 NPU 卡用于 RDMA/HCCS 测试 |
| OS | Linux aarch64 | 已验证: openEuler / Ubuntu |
| CANN | 9.0.0-beta1 或更高 | 含 HiXL 单边通信库 |
| Python | 3.10 ~ 3.11 | 推荐 3.11 |
| PyTorch | 2.7.x (CANN 版) | 由 torch_npu 提供 |
| torch_npu | 与 PyTorch 版本匹配 | 如 torch_npu 2.7.1 对应 PyTorch 2.7.1 |
| Monarch | 源码编译 (ascend 分支) | 含 HiXL Rust 后端 |

## 2. 安装步骤

### 2.1 创建 Conda 环境

```bash
conda create -n monarch_ascend python=3.11 -y
conda activate monarch_ascend
```

### 2.2 安装 CANN 工具包

按照华为官方文档安装 CANN 9.0.0 或更高版本。安装完成后确认 `set_env.sh` 路径：

```bash
# 示例路径，根据实际安装位置修改
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 或
source /path/to/cann-9.0.0/set_env.sh
```

验证 CANN 安装：

```bash
npu-smi info
```

### 2.3 安装 PyTorch + torch_npu

```bash
# 安装与 CANN 版本匹配的 PyTorch 和 torch_npu
pip install torch==2.7.1 torch_npu==2.7.1
```

验证：

```bash
python -c "import torch; import torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
# 预期输出: True 8  (数量取决于你的机器)
```

### 2.4 编译安装 Monarch (含 HiXL 后端)

Monarch 需要从源码编译以支持 Ascend HiXL RDMA 后端：

```bash
git clone https://github.com/monarch-project/monarch.git  # 或你的 fork
cd monarch
git checkout ascend/actor-plan  # NPU 适配分支

# 编译 Rust 组件 + Python 绑定
# 确保已 source CANN set_env.sh
pip install -e . -v
```

验证 Monarch 和 RDMA 后端：

```bash
python -c "
from monarch._rust_bindings.rdma import rdma_supported
print('RDMA supported:', rdma_supported())
from monarch.actor import this_host
print('Monarch actor OK')
"
```

### 2.5 安装 TorchStore NPU 适配版

```bash
git clone https://github.com/Tron-x/torchstore_ascend.git
cd torchstore_ascend

# 使用 --no-deps 避免与已安装的 torch/monarch 版本冲突
# (pyproject.toml 声明的是 GPU 版依赖 torch==2.9.0, torchmonarch==0.2.0)
pip install pygtrie portpicker
pip install -e . --no-deps
```

验证安装：

```bash
python -c "
import torchstore as ts
from torchstore.transport import TransportType
from torchstore.transport.monarch_rdma import monarch_rdma_transport_available
print('TorchStore import OK')
print('MonarchRDMA available:', monarch_rdma_transport_available())
print('Transport types:', [t.name for t in TransportType])
"
```

## 3. 环境变量配置

每次使用前需设置的环境变量：

```bash
# CANN 运行时
source /path/to/cann/set_env.sh

# 指定可见 NPU 设备 (根据需要调整)
export ASCEND_RT_VISIBLE_DEVICES=0,1

# Monarch 通信
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
```

### 可选环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HCCL_INTRA_ROCE_ENABLE` | 未设置 | 设为 `1` 强制 HiXL 使用 RoCE 而非 HCCS |
| `TORCHSTORE_RDMA_ENABLED` | `1` | 设为 `0` 禁用 RDMA 传输 |
| `TORCHSTORE_MONARCH_RDMA_EAGER_D2H` | `1` | 设为 `0` 让 RDMA 直接操作 NPU 内存（需 2MB 对齐）|
| `TORCHSTORE_SHARED_MEMORY_ENABLED` | `1` | 设为 `0` 禁用共享内存传输 |
| `MONARCH_NPU_DEVICE` | - | 在 actor bootstrap 中指定 NPU 设备 ID |
| `MONARCH_PYTHON_HIXL_ENGINE_ID` | - | HiXL 引擎 ID，由 `compute_engine_id()` 生成 |

## 4. 运行测试

### 4.1 快速兼容性检查

```bash
cd torchstore_ascend
ASCEND_RT_VISIBLE_DEVICES=0,1 python -u tests/test_npu_compat.py
```

### 4.2 运行单个测试组

```bash
# 可选组: basic, batch, shm, large, object, keys, state_dict, dws, gloo,
#         npu_store, keys_prefix, rdma, hccs, ts_rdma
ASCEND_RT_VISIBLE_DEVICES=0,1 python -u tests/test_torchstore_npu.py --only basic
```

### 4.3 运行全部 14 组测试

```bash
# 方式 1: 使用运行脚本（推荐，自动隔离每组测试）
bash tests/run_all_npu.sh

# 方式 2: 一次性运行
ASCEND_RT_VISIBLE_DEVICES=0,1 python -u tests/test_torchstore_npu.py --only all
```

### 4.4 测试组说明

| 组名 | 传输层 | 说明 |
|------|--------|------|
| `basic` | MonarchRPC | 基础 put/get |
| `batch` | MonarchRPC | 批量 NPU 张量存取 |
| `shm` | POSIX 共享内存 | SharedMemory 与 NPU 张量交互 |
| `large` | 自动选择 | 大张量 (4MB/8MB) 传输 |
| `object` | MonarchRPC | Python 对象序列化 |
| `keys` | MonarchRPC | Key 生命周期 (exists/delete) |
| `state_dict` | MonarchRPC | 模型 state_dict 完整往返 |
| `dws` | Mock RDMA | DirectWeightSync 权重同步 |
| `gloo` | Gloo (TCP) | Gloo 传输层 |
| `npu_store` | MonarchRPC | NPU↔CPU 张量存取往返 |
| `keys_prefix` | 自动选择 | Key 前缀查询 |
| `rdma` | **HiXL RoCE** | NPU 跨设备 RDMA (RoCE) |
| `hccs` | **HiXL HCCS** | NPU 片间互联 (D2D) |
| `ts_rdma` | **HiXL RDMA** | TorchStore RDMA 完整生命周期 |

预期结果: **14 GROUP PASS / 0 GROUP FAIL**

## 5. 已知限制

### 5.1 HiXL HCCS 2MB 对齐要求

HiXL 的 HCCS 传输要求内存 2MB 对齐。CPU 上 `malloc` 分配的内存不满足此条件，因此：

- TorchStore 默认的 `MONARCH_RDMA_EAGER_D2H=1`（先转 CPU 再建 RDMABuffer）在 **HCCS 模式下会失败**
- **解决方案**：
  - 保持张量在 NPU 上（设置 `TORCHSTORE_MONARCH_RDMA_EAGER_D2H=0`）
  - 或强制使用 RoCE（设置 `HCCL_INTRA_ROCE_ENABLE=1`）
  - 或使用 `alloc_aligned_tensor()` 分配 2MB 对齐的 CPU 内存

### 5.2 HiXL 不支持自连接

HiXL 不支持同一设备连接自己（与 ibverbs 不同）。RDMABuffer 的 `read_into` / `write_from` 必须跨不同进程/设备使用。

### 5.3 SharedMemory pin_memory

NPU 环境下 `torch_npu` 没有暴露类似 `cudaHostRegister` 的公开 API，因此 SharedMemory 的 `pin_memory` 在 NPU 上自动跳过。这不影响正确性，但可能影响同主机传输性能。

### 5.4 torch.save 序列化警告

直接对 NPU 张量执行 `torch.save` 会产生警告。建议在存入 TorchStore 前先 `.cpu()` 转换。

### 5.5 PyTorch 版本兼容

`torch.distributed.tensor._utils` 中的 `_compute_local_shape_and_global_offset` 在不同 PyTorch 版本间有 API 变更。本适配已通过 `try/except` 兼容 2.7.x 和 2.9+ 版本。

## 6. NPU 适配修改清单

| 文件 | 修改内容 |
|------|---------|
| `torchstore/transport/monarch_rdma.py` | `is_ibverbs_available` → `rdma_supported`，支持 HiXL 检测 |
| `torchstore/utils.py` | 新增 `_detect_accelerator_dim()`，自动选择 `npus` 或 `gpus` |
| `torchstore/transport/shared_memory.py` | NPU 设备同步 + pin_memory 跳过 |
| `torchstore/transport/gloo.py` | 注册 NPU 设备到 Gloo 后端 |
| `torchstore/transport/types.py` | PyTorch 版本兼容 import |

## 7. 传输优先级链

NPU 适配后，TorchStore 的传输自动选择优先级为：

```
SharedMemory（同主机）→ MonarchRDMA（HiXL 单边通信）→ Gloo（TCP）→ MonarchRPC
```

---

*文档版本: 2026-03-19，基于 ascend/npu-adaptation 分支*
