vllm/v1/worker

里面有各种 hardware 适配
硬件软件结合的交错点

核心：
vllm/v1/worker/worker_base.py 抽象

vllm/v1/worker/gpu_worker.py
```
class Worker(WorkerBase):
```

worker 初始化一系列变量和环境，给 model_executer