# PD × AFD Graph/U2 实验

本分支为 Decode 侧 Attention/FFN 增加 `FULL_DECODE_ONLY + U2` 实验入口。
P 节点仍负责 Prefill，因此继续使用原有 eager 启动方式；图捕获只作用于 Decode
阶段。当前模式固定使用 `P2pHcclAFDConnector`，不支持 MTP。

服务器本地的模型、IP、网卡、端口等配置仍放在私有环境文件中：

```bash
export DEPLOY_ENV_FILE=/path/to/script/two_node_16npu.env
```

先按原方案启动 Mooncake Master 和 P 节点 Prefill。在 D 节点分别启动 FFN 和
Decode Attention（两个终端或两个后台进程）：

```bash
cd /path/to/afd-plugin
bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_ffn_graph_u2.sh
```

```bash
cd /path/to/afd-plugin
bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/pd_decode_attention_graph_u2.sh
```

Decode Attention 健康检查通过后，再启动原有 PD proxy 并发送请求。首次覆盖某个
shape 时会发生图捕获，耗时可能高于后续回放。

验证时至少检查以下证据：

- 启动参数不包含 `--enforce-eager`，并包含
  `"cudagraph_mode":"FULL_DECODE_ONLY"` 与 `--enable-dbo`。
- Attention 日志出现 ACL Graph 捕获/回放信息，FFN 日志出现
  `AFD NPU FFN captured ACL graph` 或 `replaying ACL graph`。
- C32 请求全部完成，结果与 eager/U2 基线一致。
- Profile 中同时存在 Stage 0、Stage 1，并可观察两个 Stage 的 HCCL 通信与计算重叠。

这是实验性功能。合并为生产默认值前，必须在目标 910C、CANN、torch-npu、
vLLM-Ascend 固定版本上完成正确性、稳定性和性能对照测试。
