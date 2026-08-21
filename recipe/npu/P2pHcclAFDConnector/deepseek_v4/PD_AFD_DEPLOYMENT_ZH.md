# DeepSeek-V4 两节点 Mooncake PD × AFD 部署

本方案面向两台 16 卡 910C 机器。节点 P 的 16 卡运行原生 Prefill，节点 D
的前 8 卡运行 Decode Attention、后 8 卡运行 Decode FFN：

```text
OpenAI Client -> Proxy :8000
                    |-> Node P: Prefill DP4/TP4, NPU 0-15 :8100
                    |       | MooncakeHybridConnector 传输请求 KV
                    |       ` Mooncake KV Pool 保存和复用 KV
                    `-> Node D: Decode Attention DP8/TP1, NPU 0-7 :8200
                                      <-> P2pHcclAFDConnector
                                Decode FFN DP8/TP1/EP8, NPU 8-15

Mooncake Master :50088 -> 管理两端 AscendStoreConnector 的 KV Pool 元数据与租约
```

`MultiConnector` 的第一个子连接器必须是 `MooncakeHybridConnector`，负责一次
PD 请求的直接 KV 传输；第二个必须是 `AscendStoreConnector`，以 Mooncake
Store 为后端管理可复用 KV。Decode FFN 没有 KV Cache，不得配置
`--kv-transfer-config`。

当前正确性基线固定为 P2pHccl、eager、U1、无 MTP。Graph、U2、MTP 需要在
该基线完成 Batch 1/8/32 验证后分别适配。

## 1. 固定软件版本

两台机器必须使用相同模型和软件栈：vLLM 0.23.0、对应的 vLLM-Ascend
`f042ad888`、CANN 9.0.1、torch-npu 2.10.0.post2，以及 Mooncake NPU
0.3.11.post1。

在两台节点相同的 Python 环境安装 Mooncake：

```bash
python -m pip install mooncake-transfer-engine-npu==0.3.11.post1 \
  --extra-index-url https://mirrors.aliyun.com/pypi/web/simple
python -c 'import mooncake; print(mooncake.__file__)'
command -v mooncake_master
```

放通至少以下端口：

| 端口 | 节点 | 用途 |
| --- | --- | --- |
| 50088 | P | Mooncake Master |
| 8000、8100 | P | Proxy、Prefill API |
| 8200、29761 | D | Decode API、AFD rendezvous |
| 36000 起 | P | Prefill Mooncake handshake |
| 36200 起 | D | Decode Mooncake handshake |
| 42000 起 | P | Prefill HCCL |
| 44000、46000 起 | D | Attention/FFN HCCL |
| 20000-35999 | 两端 | 16 卡 AscendDirectTransport 保留范围 |

Mooncake 的 `kv_port` 不能放在 16 卡 AscendDirectTransport 使用的
20000-35999 范围内，因此本方案使用 36000 和 36200。

## 2. 准备公共配置

在两台机器分别执行：

```bash
cd /mnt/workspace/code/afd-plugin
cp recipe/npu/P2pHcclAFDConnector/deepseek_v4/two_node_16npu.env.example \
  /mnt/workspace/two_node_16npu.env
vim /mnt/workspace/two_node_16npu.env
export DEPLOY_ENV_FILE=/mnt/workspace/two_node_16npu.env
```

必须修改 `PREFILL_IP`、`DECODE_IP`、`NETWORK_INTERFACE` 和 `MODEL_PATH`。
配置文件使用 `export`，所以脚本启动的所有子进程会继承这些值。

## 3. 按依赖顺序启动

### 3.1 节点 P：Mooncake Master

第一个终端：

```bash
cd /mnt/workspace/code/afd-plugin
export DEPLOY_ENV_FILE=/mnt/workspace/two_node_16npu.env
bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/start_mooncake_master.sh \
  2>&1 | tee /mnt/workspace/mooncake-master.log
```

脚本会生成 `/tmp/afd-mooncake/mooncake.json`。默认每个 Mooncake 客户端
注册 1GB segment，Master 的 KV lease TTL 为 120 秒，client TTL 为 120 秒。
先以该保守配置完成正确性验证，再根据主机内存和命中率增大
`MOONCAKE_GLOBAL_SEGMENT_SIZE`。

### 3.2 节点 D：AFD Decode

第二个终端在节点 D 执行：

```bash
cd /mnt/workspace/code/afd-plugin
export DEPLOY_ENV_FILE=/mnt/workspace/two_node_16npu.env
bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/start_decode.sh
```

该脚本先启动 FFN daemon，再启动 Attention consumer，避免两边都以阻塞
方式等待对端。日志位于 `/tmp/afd-pd-decode/ffn.log` 和
`/tmp/afd-pd-decode/attention.log`。任一角色退出时，脚本会终止另一个角色，
避免残留 NPU 进程。

### 3.3 节点 P：Prefill

第三个终端在节点 P 执行：

```bash
cd /mnt/workspace/code/afd-plugin
export DEPLOY_ENV_FILE=/mnt/workspace/two_node_16npu.env
bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/pd_prefill.sh \
  2>&1 | tee /mnt/workspace/prefill.log
```

Prefill 默认使用 `DP4 × TP4 = 16 NPU`。它是原生 vLLM-Ascend 服务，不加载
AFD 插件。

### 3.4 节点 P：请求代理

所有计算服务健康后启动：

```bash
cd /mnt/workspace/code/afd-plugin
export DEPLOY_ENV_FILE=/mnt/workspace/two_node_16npu.env
bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/start_proxy.sh \
  2>&1 | tee /mnt/workspace/pd-afd-proxy.log
```

代理先向 Prefill 发送 `do_remote_decode=true` 请求，取得
`kv_transfer_params`，再把原始请求和这些参数发送到 AFD Decode。

## 4. 健康检查与请求

在节点 P 执行：

```bash
export DEPLOY_ENV_FILE=/mnt/workspace/two_node_16npu.env
bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/check_two_node_service.sh

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"dsv4-afd",
    "messages":[{"role":"user","content":"你好"}],
    "temperature":0,
    "max_tokens":32
  }'
```

日志验收必须同时满足：Prefill 有 Mooncake KV send/Store put，Decode Attention
有 KV load/Store get，FFN 没有 Mooncake 初始化，并且 Attention 与 FFN 的
P2pHccl 收发次数匹配。

## 5. Batch 1/8/32 正确性门禁

固定 prompt、seed、temperature=0 和最大输出长度，先用未分离原生模型生成
golden token，再分别以并发 1、8、32 请求代理。每一档必须满足：

- HTTP 请求全部成功，输出 token 与 golden 逐 token 一致；
- 无 `507015`、HCCL timeout、Mooncake lease expired 或 FFN worker fatal；
- 请求结束后 KV lease 能释放，Mooncake Store 占用不持续单调增长；
- 停服后 32 张 NPU 均无残留 vLLM 进程。

排查责任边界时设置 `PD_KV_MODE=direct` 可保留 Mooncake P2P、关闭 Store
管理；设置 `DECODE_STANDALONE_AF=1` 可关闭整个 PD KV 路径，只验证 AFD
Attention/FFN 链路。

停止顺序为 Proxy、Prefill、Decode、Mooncake Master。调试重启时建议 Master
与 vLLM 客户端一起重启，避免旧 segment 元数据影响下一轮。
