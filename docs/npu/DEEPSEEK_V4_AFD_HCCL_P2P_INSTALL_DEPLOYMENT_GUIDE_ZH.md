# DeepSeek-V4 AFD HCCL P2P 安装部署指南

## 0. 交付包结构和安装入口

收到的 `dsv4-afd-hccl-install-delivery-*.zip` 是外层交付包，用于把指导书、
实际安装包和校验文件放在一起。解压后的结构如下：

```text
dsv4-afd-hccl-install-delivery-*/
├── INSTALL_DEPLOYMENT_GUIDE_ZH.md
├── PACKAGE_README_ZH.md
├── dsv4-afd-hccl-manual-install-slim-*.tar.gz
├── dsv4-afd-hccl-manual-install-slim-*.tar.gz.sha256
└── SHA256SUMS
```

其中 `dsv4-afd-hccl-manual-install-slim-*.tar.gz` 是目标机真正要解开的安装
脚本包。文件名中的 `*` 只是构建时间戳通配符，不代表还需要寻找其他文件。
该包使用 `tar.gz` 是为了保留 Linux 目录结构和脚本执行权限；它包含：

- `bin/`：环境检查、源码下载、依赖安装、构建、验收、启停和请求脚本；
- `config.env`：目标机部署参数；
- `manifest/`：固定版本、AFD MTP M1 补丁和 SHA256；
- `README_ZH.md`：脚本包使用说明。

`slim` 表示轻量包：不包含 vLLM、vLLM-Ascend、afd-plugin 完整源码，也不
包含模型和 Python wheel。执行安装时，`bin/02_prepare_sources.sh` 会下载固定
源码并校验版本。

目标机从 ZIP 开始的推荐流程如下：

```bash
unzip dsv4-afd-hccl-install-delivery-*.zip
cd dsv4-afd-hccl-install-delivery-*

# 校验 ZIP 内全部交付文件和实际安装包。
sha256sum -c SHA256SUMS
sha256sum -c dsv4-afd-hccl-manual-install-slim-*.tar.gz.sha256

# 解开实际安装脚本包。
tar -xzf dsv4-afd-hccl-manual-install-slim-*.tar.gz
cd dsv4-afd-hccl-manual-install-slim-*

# 按目标机修改路径、网卡、SoC、Git/pip 镜像等参数。
vi config.env

# 完整安装从配置打印和环境门禁开始，门禁失败不会继续安装。
bash bin/install_all.sh
```

`install_all.sh` 会运行配置打印和 preflight，然后依次完成源码准备、
venv 创建、Python 依赖安装、vLLM/vLLM-Ascend/afd-plugin 安装及验收。需要逐步
排障时，按脚本包 `README_ZH.md` 中的分步命令执行。

## 1. 适用范围

本文给出 `P2pHcclAFDConnector` 的单机 Atlas A3 安装、启动和验收方法，
主路径严格复现以下已验证基线：

| 组件 | 固定版本 |
| --- | --- |
| Python | 3.12，验证环境为 3.12.9 |
| CANN / NNAL | 9.0.1 |
| torch | 2.10.0 |
| torch-npu | 2.10.0.post2 |
| vLLM | `releases/v0.23.0`，`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann`，`3da28f9414583d2d0b672a8f06d1fae142404bda` |
| afd-plugin | tag `dsv4-afd-v023-hccl-mtp-m1-v1` |
| transformers | 5.5.4 |
| numpy | 2.2.6 |
| 硬件 | 16 NPU Atlas A3，验证机型 SoC 为 `ascend910_9362` |

推荐部署为 Attention NPU 0-7、FFN NPU 8-15，即 A8F8、DP8/TP1/EP8。
本文覆盖 eager/U1、eager/U2、等量 A/F 下的 `FULL_DECODE_ONLY` Graph/U1，
以及等量 A8F8 的 eager/U1 + MTP。MTP 首版只支持 1 个 MTP layer、
`method=mtp`、`num_speculative_tokens=1`。Graph + MTP、U2 + MTP、非等量 +
MTP、更多 speculative token、PD、sequence parallel 和 Attention-side gate
不在当前支持范围内。

本文不适用于 afd-plugin 主 README 当前的 vLLM 0.26 默认栈。不要把 0.26的 vLLM 或 vLLM-Ascend 快照混入本指南的 0.23 Graph/U1 环境。

## 2. 组件关系

三层 Python 组件按以下方向依赖：

```text
vLLM                 通用推理框架、API、调度和采样
  ^
  |
vLLM-Ascend          Ascend platform、NPU worker、DSV4、ACL Graph 和 NPU 算子
  ^
  |
afd-plugin           Attention/FFN 角色、AFD worker 和 P2pHcclAFDConnector

torch-npu -> CANN -> HCCL -> Atlas NPU
```

`P2pHcclAFDConnector` 在 afd-plugin 内，不是一个独立的 pip 包。它通过`torch.distributed.send/recv` 使用 torch-npu 注册的 HCCL backend。不要为此额外执行 `pip install hccl`。

HCCL 路径不依赖 afd-plugin 的 CAMP2P A2E/E2A 自定义算子，但 DSV4 模型仍依赖 vLLM-Ascend 自己的 `custom_transformer` 算子。因此运行时需要 source vLLM-Ascend 的 custom-op 环境，不能 source afd-plugin 的 CAMP2P 环境。

## 3. 系统和 CANN 前置条件

### 3.1 系统工具

需要 Linux aarch64、GCC/G++ 8 以上、C++17、CMake 3.26 以上、Ninja、Git、NUMA 开发库和足够的 `/dev/shm`。根据操作系统安装工具，例如：

```bash
# Ubuntu
sudo apt-get update
sudo apt-get install -y gcc g++ cmake ninja-build libnuma-dev git curl jq iproute2

# openEuler
sudo yum install -y gcc gcc-c++ cmake ninja-build numactl-devel git curl jq iproute
```

### 3.2 Driver、Firmware、CANN 和 NNAL

先安装与 CANN 9.0.1 匹配的 Ascend Driver/Firmware，再安装以下 CANN 组件：

- Toolkit 9.0.1；
- 与 SoC 匹配的 kernels/ops 包；
- NNAL/ATB 9.0.1，运行 DSV4 时需要 `libatb.so`。

安装介质和命令因发行版及硬件 SKU 不同，应以对应 CANN 9.0.1 发布包为准。
安装后在一个干净 shell 中检查：

```bash
export CANN_ROOT=/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1

source "${CANN_ROOT}/set_env.sh"
if [[ -f "${CANN_ROOT}/nnal/atb/set_env.sh" ]]; then
  source "${CANN_ROOT}/nnal/atb/set_env.sh"
fi

"${CANN_ROOT}/query_pkg_version.sh" | sed -n '1,20p'
npu-smi info
npu-smi info -t board -i 0 -c 0
```

A8F8 要求 16 张 NPU 可见且没有遗留推理进程。验证机 `NPU Name` 为 `9362`，对应构建参数 `SOC_VERSION=ascend910_9362`。其他 A3 SKU 必须使用实际 SoC，不能直接照搬该值。

不要在同一个 shell 中先后 source CANN 9.0.1 和 9.1.0。若以下命令还看到其他 CANN 版本，重新打开干净 shell：

```bash
env | sort | rg '^(ASCEND|CANN|PATH|PYTHONPATH|LD_LIBRARY_PATH)='
```

## 4. 获取固定源码

以下目录与仓库内现有激活、检查和 recipe 脚本一致。对已有工作区先执行`git status --short`，有本地改动时不要强制切换提交。

使用轻量安装包时，推荐在修改 `config.env` 后直接执行
`bash bin/02_prepare_sources.sh`，由脚本完成以下下载、补丁和 tree 校验。只有
需要完全手工安装时才执行本节命令。手工方式必须先设置已解开的安装包目录：

```bash
export INSTALL_BUNDLE_ROOT=/path/to/dsv4-afd-hccl-manual-install-slim-YYYYmmdd_HHMMSS
export CODE_ROOT=/mnt/workspace/code
mkdir -p "${CODE_ROOT}"

git clone https://github.com/vllm-project/vllm.git \
  "${CODE_ROOT}/vllm-release-v0.23.0"
git -C "${CODE_ROOT}/vllm-release-v0.23.0" checkout --detach \
  0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665

git clone https://github.com/vllm-project/vllm-ascend.git \
  "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann"
git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" checkout --detach \
  3da28f9414583d2d0b672a8f06d1fae142404bda
git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" \
  submodule update --init --recursive

git clone https://github.com/wenhow/afd-plugin.git \
  "${CODE_ROOT}/afd-plugin"
git -C "${CODE_ROOT}/afd-plugin" checkout --detach \
  d7aeb9b7554803931e42bf405623f212030ed60f

(
  cd "${INSTALL_BUNDLE_ROOT}"
  sha256sum -c manifest/SHA256SUMS
)
git -C "${CODE_ROOT}/afd-plugin" apply --index \
  "${INSTALL_BUNDLE_ROOT}/manifest/afd-plugin-mtp-m1.patch"

test "$(git -C "${CODE_ROOT}/afd-plugin" write-tree)" = \
  8f2dfdb1533353d424ccfd78d66d8647df37ac85
```

MTP M1 tag 尚未发布到远端，因此不能直接按 tag checkout。上述已发布基础提交
加包内补丁会重建与 `dsv4-afd-v023-hccl-mtp-m1-v1` 完全相同的源码 tree。

核对三个固定点：

```bash
git -C "${CODE_ROOT}/vllm-release-v0.23.0" rev-parse HEAD
git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" rev-parse HEAD
git -C "${CODE_ROOT}/afd-plugin" rev-parse HEAD
git -C "${CODE_ROOT}/afd-plugin" write-tree
```

预期依次为 vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、
vLLM-Ascend `3da28f9414583d2d0b672a8f06d1fae142404bda`、afd-plugin 基础提交
`d7aeb9b7554803931e42bf405623f212030ed60f` 和 MTP M1 目标 tree
`8f2dfdb1533353d424ccfd78d66d8647df37ac85`。

## 5. 创建 Python 环境

### 5.1 创建 venv

验证基线使用 Python 3.12.9：

```bash
export VENV_ROOT=/mnt/workspace/code/.venvs/afd-v023-vllm-cann
python3.12 -m venv "${VENV_ROOT}"
source "${VENV_ROOT}/bin/activate"

python -m pip install --upgrade \
  pip "setuptools>=64" "setuptools-scm>=8" wheel \
  "cmake>=3.26" ninja pybind11
```

此后所有 `python` 和 `pip` 命令都必须来自该 venv：

```bash
command -v python
python --version
```

### 5.2 安装 NPU Python 运行时

torch、torch-npu 与 CANN 必须成套。华为源地址可按部署网络替换为内部镜像：

```bash
python -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  torch-npu==2.10.0.post2 triton-ascend==3.2.1

python -m pip install \
  -r "${CODE_ROOT}/vllm-release-v0.23.0/requirements/common.txt"
```

目标 `rfc/vllm_cann` 提交的 `requirements.txt` 仍写着`torch-npu==2.10.0`，而 Graph/U1 实际验证和运行检查要求
`2.10.0.post2`。安装该 requirements 后必须恢复验证版本：

```bash
python -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  -r "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann/requirements.txt"

python -m pip install --upgrade --force-reinstall --no-deps \
  torch-npu==2.10.0.post2 transformers==5.5.4 numpy==2.2.6
```

这个固定栈存在两个已知的包元数据偏差：vLLM-Ascend 源码元数据要求`torch-npu==2.10.0`，triton-ascend 3.2.1 元数据要求 numpy 1.26.4；已验证运行时分别使用 2.10.0.post2 和 2.2.6。因此 `pip check` 可能报告这两项，最终门禁应以第 7 节的实际 import、版本和 NPU 检查为准。不要在环境验证后执行无版本约束的 `pip install -U`。

## 6. 安装 vLLM、vLLM-Ascend 和 afd-plugin

安装顺序固定为 vLLM、vLLM-Ascend、afd-plugin。

### 6.1 安装 vLLM 0.23

Ascend 使用 `empty` target，避免构建 CUDA/ROCm 扩展：

```bash
cd "${CODE_ROOT}/vllm-release-v0.23.0"
VLLM_TARGET_DEVICE=empty \
  python -m pip install --no-build-isolation --no-deps --editable .
```

预期版本为 `0.23.0+empty`。

### 6.2 安装 vLLM-Ascend

构建前必须已经 source CANN 9.0.1，并使用与设备匹配的 SoC：

```bash
export SOC_VERSION=ascend910_9362
cd "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann"
python -m pip install -v --no-build-isolation --no-deps --editable .
```

构建成功后必须存在以下文件：

```bash
export VLLM_ASCEND_OPS_ENV="${CODE_ROOT}/vllm-ascend-rfc-vllm-cann/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
test -f "${VLLM_ASCEND_OPS_ENV}"
```

### 6.3 安装 HCCL Connector

HCCL-only 部署不构建 afd-plugin 的 CAMP2P 自定义算子：

```bash
cd "${CODE_ROOT}/afd-plugin"
AFD_BUILD_ASCEND_OPS=0 \
  python -m pip install -v --no-build-isolation --no-deps --editable .
```

如果同一个环境还要运行 `CAMP2pAFDConnector`，应改用
`AFD_BUILD_ASCEND_OPS=1` 重新安装，并配置对应 custom-op 环境。但运行
`P2pHcclAFDConnector` 时仍不要 source
`afd_plugin/_cann_ops_custom/vendors/afd-plugin/bin/set_env.bash`。

## 7. 激活和安装验收

### 7.1 使用固定激活脚本

```bash
cd "${CODE_ROOT}/afd-plugin"
export DSV4_CANN_ROOT="${CANN_ROOT}"
export DSV4_RUNTIME_VENV="${VENV_ROOT}"
export DSV4_VLLM_ROOT="${CODE_ROOT}/vllm-release-v0.23.0"
export DSV4_VLLM_ASCEND_ROOT="${CODE_ROOT}/vllm-ascend-rfc-vllm-cann"
source tools/dsv4/activate_v023_vllm_cann_runtime.sh
```

该脚本还会设置：

```text
VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd
```

仓库脚本按验证机的 `/opt/buildtools/python-3.12.9` 清理基础 PATH。若部署机
的 Python 安装位置不同，应先按本机路径调整 `tools/dsv4/activate_runtime.sh`，
不能保留一个不存在的 Python 或动态库目录。

NNAL 的 `set_env.sh` 会通过 `import torch` 探测 C++ ABI，首次 source 可能需要
数十秒。先等待命令返回，再执行后续检查，不要在探测过程中重复 source。

### 7.2 HCCL-only 运行检查

```bash
python - <<'PY'
from importlib.metadata import version

import torch
import torch_npu
import vllm
import vllm_ascend  # noqa: F401

from afd_plugin.connectors.npu.p2p_hccl import P2pHcclAFDConnector

assert torch.__version__.startswith("2.10.0")
assert version("torch-npu") == "2.10.0.post2"
assert vllm.__version__.startswith("0.23.0")
assert version("vllm-ascend").endswith("g3da28f941")
assert version("transformers") == "5.5.4"
assert version("numpy") == "2.2.6"
assert P2pHcclAFDConnector.__name__ == "P2pHcclAFDConnector"
assert torch.npu.is_available()
assert torch.npu.device_count() == 16

print("DSV4_AFD_HCCL_RUNTIME_OK")
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("vllm", vllm.__version__)
print("vllm_ascend", version("vllm-ascend"))
PY
```

现有 `tools/dsv4/check_v023_vllm_cann_runtime.sh` 还会检查 afd-plugin 的
CAMP2P 自定义算子，因此它只适合 `AFD_BUILD_ASCEND_OPS=1` 的完整构建。
HCCL-only 安装使用上面的检查，不要因该脚本的 CAMP2P 检查失败误判 HCCL。

## 8. 部署前配置

### 8.1 模型和网络

确认模型目录至少包含 config、tokenizer 和所有 safetensors/index 文件：

```bash
export MODEL_PATH=/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp
test -f "${MODEL_PATH}/config.json"
find "${MODEL_PATH}" -maxdepth 1 -type f | sort | sed -n '1,40p'
```

选择 HCCL/Gloo 通信网卡，并将 `HCCL_IF_IP` 设置为该网卡的真实 IPv4 地址：

```bash
export GLOO_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
ip -o -4 addr show dev "${HCCL_SOCKET_IFNAME}"
export HCCL_IF_IP=YOUR_LOCAL_NPU_IP
```

`YOUR_LOCAL_NPU_IP` 必须替换为当前节点实际通信 IP。单机 A8F8 的 AFD
rendezvous 可以使用 `127.0.0.1`。

### 8.2 端口和 NPU

默认端口如下：

| 用途 | 端口 |
| --- | ---: |
| Attention API | 8910 |
| FFN 启动进程 | 8911 |
| AFD rendezvous | 29761 |
| Attention HCCL base | 51000 |
| FFN HCCL base | 52000 |

部署前确认端口空闲，且 NPU0-15 没有其他任务：

```bash
ss -ltnp | rg ':(8910|8911|29761|51000|52000)\b' || true
npu-smi info
df -h /dev/shm
```

## 9. 启动 A8F8 Graph/U1

在同一个 shell 中完成激活和变量设置。FFN 与 Attention 必须前后紧邻地
启动，不能等待 FFN HTTP ready 后再启动 Attention，因为双方会在 AFD/HCCL
初始化阶段互相等待。

```bash
cd "${CODE_ROOT}/afd-plugin"
source tools/dsv4/activate_v023_vllm_cann_runtime.sh

export MODEL_PATH=/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp
export HCCL_IF_IP=YOUR_LOCAL_NPU_IP
export GLOO_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export AFD_HOST=127.0.0.1
export AFD_PORT=29761
export ATTENTION_RANKS=8
export FFN_RANKS=8
export ATTENTION_DEVICES=0,1,2,3,4,5,6,7
export FFN_DEVICES=8,9,10,11,12,13,14,15
export EXECUTION_MODE=full-decode-only
export U_BATCHES=1

mkdir -p /mnt/workspace/logs/dsv4-afd-hccl

bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_ffn.sh \
  > /mnt/workspace/logs/dsv4-afd-hccl/ffn.log 2>&1 &
ffn_pid=$!

sleep 2

bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_attention.sh \
  > /mnt/workspace/logs/dsv4-afd-hccl/attention.log 2>&1 &
attention_pid=$!
```

脚本会自动：

- 为两个角色选择 `P2pHcclAFDConnector`；
- 只 source vLLM-Ascend 的 `custom_transformer` 环境；
- Attention 使用 NPU0-7，FFN 使用 NPU8-15；
- 设置 DP8/TP1/EP8、W8A8 Ascend quantization；
- Graph 模式使用 `FULL_DECODE_ONLY`，capture size 为 1/2/4/8。

模型加载、编译和 Graph capture 可能持续数分钟。进程存活不等于服务 ready。

## 10. Readiness 和请求验证

只向 Attention API 发请求。FFN 是 connector-driven 后台角色，不要把 FFN
端口当成业务健康接口。

```bash
curl -fsS --max-time 10 http://127.0.0.1:8910/health

rg -o 'AFD FFN EngineCore started; workers run connector loop' \
  /mnt/workspace/logs/dsv4-afd-hccl/ffn.log | wc -l

rg 'enable_npugraph_ex|Graph capturing finished|Replaying aclgraph' \
  /mnt/workspace/logs/dsv4-afd-hccl/attention.log
```

A8F8 下 FFN ready marker 应出现 8 次。Graph 模式还应看到 8 个 Attention
rank 完成 capture，并在请求阶段看到 ACL Graph replay。

发送一个 OpenAI Completions 请求：

```bash
curl -fsS http://127.0.0.1:8910/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4-afd",
    "prompt": "Please explain why deterministic validation matters.",
    "max_tokens": 32,
    "temperature": 0
  }'
```

同时检查两侧日志中没有 fatal marker：

```bash
rg -n \
  'EngineCore encountered a fatal error|AFD NPU FFN worker loop failed|Communication_Error|507015|Traceback' \
  /mnt/workspace/logs/dsv4-afd-hccl/attention.log \
  /mnt/workspace/logs/dsv4-afd-hccl/ffn.log
```

## 11. 自动化功能验收

已有 golden 文件时，优先使用 recipe runner。它会检查端口、启动两个角色、
等待 Attention API、执行串行 golden 和 batch 验证、按 Attention 后 FFN 的
顺序退出，并检查 fatal 日志与 NPU 清理：

```bash
cd "${CODE_ROOT}/afd-plugin"
source tools/dsv4/activate_v023_vllm_cann_runtime.sh
export HCCL_IF_IP=YOUR_LOCAL_NPU_IP
export GLOO_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0

python recipe/npu/P2pHcclAFDConnector/deepseek_v4/run_validation.py \
  --execution-mode full-decode-only \
  --u-batches 1 \
  --golden /mnt/workspace/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json \
  --cycles 1 \
  --idle-seconds 0 \
  --rounds 3 \
  --batch-sizes 1 8 32 \
  --output-dir /mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u1_install_gate
```

最终检查 `validation_summary.json` 中的 `passed`，并确认 Attention/FFN
返回码均为 0、NPU cleanup 通过。安装 smoke 不能替代 golden token 验收。

在加载大模型前，也可以先做 A1F1、U1 的 HCCL 组件 round-trip：

```bash
mkdir -p /mnt/workspace/validation/dsv4_hccl_component_smoke
python tools/dsv4/validate_hccl_p2p_roundtrip.py \
  --attention-devices 0 \
  --ffn-devices 8 \
  --stages 1 \
  --steps 2 \
  --port 29841 \
  --output /mnt/workspace/validation/dsv4_hccl_component_smoke/summary.json
```

## 12. eager/U1 和 eager/U2

eager/U1 只需把两个角色的环境改成：

```bash
export EXECUTION_MODE=eager
export U_BATCHES=1
```

等量 A8F8 的 eager/U1 + MTP 在此基础上增加：

```bash
export ENABLE_MTP=1
export MTP_NUM_SPECULATIVE_TOKENS=1
```

也可以直接运行自动化功能门禁：

```bash
python recipe/npu/P2pHcclAFDConnector/deepseek_v4/run_validation.py \
  --connector P2pHcclAFDConnector \
  --execution-mode eager \
  --u-batches 1 \
  --enable-mtp \
  --mtp-num-speculative-tokens 1 \
  --golden /mnt/workspace/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json \
  --cycles 1 --idle-seconds 0 --rounds 3 --batch-sizes 1 8 32 \
  --output-dir /mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m1_install_gate
```

当前 MTP draft 使用学习式 gate。Attention 先发送包含各 DP token count 的固定
header，再发送 post-HC `[T,4096]` BF16 hidden；FFN 返回同 shape 的 MoE output。
MTP phase 不发送 input IDs，connector 会拒绝错误的 pre-HC `[T,4,4096]` 输入。

eager/U2 使用：

```bash
export EXECUTION_MODE=eager
export U_BATCHES=2
export DBO_DECODE_TOKEN_THRESHOLD=2
export DBO_PREFILL_TOKEN_THRESHOLD=12
```

Graph 只能与 U1 组合。若设置 `EXECUTION_MODE=full-decode-only` 且
`U_BATCHES=2`，recipe 会直接拒绝启动。

MTP M1 只能与 eager/U1 和 A8F8 等量拓扑组合。Graph + MTP、U2 + MTP、
非等量 + MTP 或 `MTP_NUM_SPECULATIVE_TOKENS` 大于 1 会在启动前 fail-fast。

eager 支持 `A = k x F` 的整数倍 connector 拓扑，但 FFN 的
`max_num_batched_tokens` 至少要是 Attention 值的 `k` 倍。A3 上 A8F4 的完整
DeepSeek-V4 FFN EP4 已确认受 64 GiB HBM 容量限制，这不是 HCCL connector
故障。生产首选仍是已完整验证的 A8F8。

## 13. 正常停止

必须先停止 Attention，再停止 FFN，让 connector 按协议关闭：

```bash
kill -TERM "${attention_pid}"
wait "${attention_pid}"

kill -TERM "${ffn_pid}"
wait "${ffn_pid}"

npu-smi info
ss -ltnp | rg ':(8910|8911|29761|51000|52000)\b' || true
```

正式部署应由 systemd、Supervisor 或容器编排分别管理两个角色，并保留相同
的启动并发关系和停止顺序。不要用模糊的全局 `pkill` 清理服务。

## 14. 常见故障

| 现象 | 优先检查 |
| --- | --- |
| import 时找不到 `.so` 或出现 ABI 错误 | CANN、torch、torch-npu 是否成套；是否混入 9.1.0 路径 |
| `libatb.so` 找不到 | NNAL/ATB 是否安装并 source 对应 `set_env.sh` |
| `P2pHcclAFDConnector` 未注册 | afd-plugin 是否安装；`VLLM_PLUGINS` 是否包含 `afd` |
| vLLM-Ascend custom op 不存在 | 是否初始化 submodule；是否在正确 CANN/SoC 下重新构建 vLLM-Ascend |
| HCCL bind/connect 错误 | `HCCL_IF_IP`、两个 socket interface、AFD port 和 base port 是否一致且空闲 |
| FFN 启动后一直等待 | Attention 是否在 2 秒后并发启动；双方 AFD host/port/rank 数是否一致 |
| Attention health 正常但请求 hang | FFN 8 个 rank 是否都进入 connector loop；两侧 HCCL 日志是否报错 |
| Graph 首次请求失败 | 是否使用固定 tag、torch-npu 2.10.0.post2、Graph/U1 和等量 A/F |
| MTP 配置启动即拒绝 | 是否为 HCCL connector、A8F8、eager/U1、1 个 MTP layer 和 1 个 speculative token |
| MTP FFN 收到 shape 错误 | 远端边界必须是 post-HC `[T,4096]`；不要发送 target hidden buffer 的三维 view |
| 启动数分钟仍未 ready | 查看模型加载、编译和 capture 进度；不要只看父进程存活 |
| 重启报端口占用或显存未释放 | 先按角色 PID 正常停止，核对端口和 `npu-smi info`，不要叠加启动第二套服务 |

固定 vLLM 0.23 的 FFN API launcher 在计划内 SIGTERM 后可能于
`[shutdown] MPClient: complete` 之后打印 `KeyboardInterrupt: terminated` 和
`ERR99999`。只有两侧返回码为 0、请求阶段没有 fatal marker 且 NPU cleanup 通过
时，才将它归类为已知 shutdown 噪声；其他位置的 traceback 仍按失败处理。

## 15. 生产交付检查表

- 三个源码提交/tag 与第 1 节一致；
- CANN 环境中没有其他版本路径；
- `torch.npu.device_count() == 16`；
- Attention/FFN 使用不重叠的 0-7 和 8-15；
- HCCL/Gloo 网卡和本机 IP 已按部署机修改；
- 业务请求只进入 Attention API；
- 8 个 FFN rank 均进入 connector loop；
- Graph/U1 的 capture/replay 证据完整；
- 使用 MTP 时确认 eager/U1、proposal/acceptance 日志和 MTP phase 证据完整；
- golden、batch、fatal-log、正常退出和 NPU cleanup 门禁通过；
- 日志、版本、启动环境和验收产物已归档。

功能范围和正式验证证据见
[`DEEPSEEK_V4_AFD_HCCL_P2P_GRAPH_U1_VALIDATION_REPORT_ZH.md`](DEEPSEEK_V4_AFD_HCCL_P2P_GRAPH_U1_VALIDATION_REPORT_ZH.md)
和
[`DEEPSEEK_V4_AFD_HCCL_P2P_MTP_M1_VALIDATION_REPORT_ZH.md`](DEEPSEEK_V4_AFD_HCCL_P2P_MTP_M1_VALIDATION_REPORT_ZH.md)。

## 16. 可移植安装脚本包

用于其他 A3 环境手工安装的脚本和打包器位于：

```text
tools/dsv4/hccl_manual_install/
```

默认生成轻量 transfer archive。包内只包含脚本、固定版本清单和 afd-plugin
MTP M1 补丁；目标机重新下载 vLLM、vLLM-Ascend（含递归 submodule）和
afd-plugin 基础源码：

```bash
bash tools/dsv4/hccl_manual_install/build_bundle.sh /mnt/workspace/artifacts
```

生成物名称包含 `slim`，同时提供 `.tar.gz.sha256`。源码、模型和 Python wheel
均不进入轻量包。目标机可联网安装；若必须携带源码，可设置
`INCLUDE_SOURCES=1` 生成名称包含 `with-sources` 的完整包。Python wheel 可在
构建时设置 `INCLUDE_WHEELHOUSE=/path/to/wheels` 加入。详细步骤见
[`hccl_manual_install/README_ZH.md`](../../tools/dsv4/hccl_manual_install/README_ZH.md)。

正式包固定使用 `dsv4-afd-v023-hccl-mtp-m1-v1`。由于该 MTP M1 tag 尚未发布
到远端，轻量包从已发布提交 `d7aeb9b7554803931e42bf405623f212030ed60f`
下载 afd-plugin，再应用包内补丁。打包和目标机安装都会校验最终 Git tree、
精确 commit 以及 SHA256。
