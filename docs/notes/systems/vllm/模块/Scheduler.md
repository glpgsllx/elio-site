vllm/v1/core/sched/scheduler.py

core文件夹可以理解为实现 vllm paper 的地方

经过*一次* inference 的过程叫做一个 step （生成一个 token）

Scheduler：每个inference放哪个request

所有request打包成一个大包，一起跑(continuous batching)

