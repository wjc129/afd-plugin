# Connector Package Layout

AFD connector implementations are grouped by backend:

- `gpu/`: GPU-only connector implementations. `P2pNcclAFDConnector` is implemented by
  `afd_plugin.connectors.gpu.p2p`.
- `npu/`: NPU-only connector implementations. `CAMP2pAFDConnector` is implemented
  by `afd_plugin.connectors.npu.camp2p`, and `CAMAsyncAFDConnector` is implemented
  by `afd_plugin.connectors.npu.async_cam`.

The vLLM 0.23 source port includes GPU `P2pNcclAFDConnector` and NPU
`CAMP2pAFDConnector` and `CAMAsyncAFDConnector` code paths, but still requires
matching hardware E2E validation. The CAM async PCP8 recipe is retained for
v0.19.1rc1 and must be used with the `release/v0.19.1rc1` branch.

Shared connector contracts, metadata containers, factory registration, and
backend-neutral helpers stay in `afd_plugin.connectors`.
