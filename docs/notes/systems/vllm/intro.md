# vLLM 代码库

这里作为 vLLM 相关笔记的总入口，从整体架构出发，链接到各个模块拆解与优化专题。

## 模块拆解

- [Entrypoint](模块/Entrypoint.md)：`LLM` / API server 入口与请求生命周期。
- [Engine](模块/Engine.md)：调度核心、执行 pipeline 与资源管理。
- [Scheduler](模块/Scheduler.md)：请求排队策略、chunk/prefill 调度逻辑。
- [KV Cache Manager](模块/KV Cache Manager.md)：KV cache 生命周期、分页与回收。
- [Evictor](模块/Evictor.md)：eviction 策略、冷热数据与显存回收。
- [Worker](模块/Worker.md)：worker 角色、与 engine 的通信协议。
- [Model Executor](模块/Model_executer.md)：模型前向执行、batch 合并与张量布局。
- [Modeling](模块/Modeling.md)：模型定义、权重加载与与 HF/原生接口的对接。
- [Attention backend](模块/Attention backend.md)：不同 Attention kernel / backend 的抽象与选择。

## 周边与优化

- [Distributed Inference](周边&优化/Distributed Inference.md)：多机多卡推理、分布式拓扑与通信开销。
- [KV Cache 管理专题](kv-cache.md)：从抽象接口到 PagedAttention slot 分配与显存碎片治理。
- （预留）Speculative decoding / Chunked prefill / Cascade inference / Prefix caching 等优化可以在这里继续拆分成单独文件后挂链接。
