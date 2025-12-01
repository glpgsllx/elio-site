- Why distributed inference?
- Typed of distributed inference: TP / EP / PP
- PD Disaggregation

# Why?
1. 模型单卡放不下
2. 推理的不同阶段都能充分利用硬件资源
	1. prefil：计算密集，IO要求不高
	2. decode：内存密集型。batch size 大了才效率高

## 传统的分布式方法
## Tensor parallel
把模型同一层的weights拆成好几份，分到不同的硬件上各自计算，然后 all-reduce 重新汇总起来。
模型的每一层会被切成几块（比如四分之一、八分之一），每个 worker/GPU 只持有自己那一片参数，并对完整输入做自己的那部分计算；等所有分片都算完后，再把这些局部结果拼接起来，得到最终的层输出。

vllm/distributed/parallel_state.py
```
_TP: GroupCoordinator | None = None
def get_tp_group() -> GroupCoordinator:
	assert _TP is not None, "tensor model parallel group is not initialized"

	return _TP
```

```
# message queue broadcaster is only used in tensor model parallel group

_TP = init_model_parallel_group(
    group_ranks,
	get_world_group().local_rank,
	backend,
	use_message_queue_broadcaster=True,
	group_name="tp",
)
```


## Pipeline parallel 
把**不同层**分布到不同 GPU 上串联起来。
```
class GroupCoordinator:
	rank: int # global rank
	ranks: list[int] # global ranks in the group
	world_size: int # size of the group
	# difference between `local_rank` and `rank_in_group`:
	# if we have a group of size 4 across two nodes:
	# Process | Node | Rank | Local Rank | Rank in Group
	# 0 | 0 | 0 | 0 | 0
	# 1 | 0 | 1 | 1 | 1
	# 2 | 1 | 2 | 0 | 2
	# 3 | 1 | 3 | 1 | 3
	local_rank: int # local rank used to assign devices
	rank_in_group: int # rank inside the group
	cpu_group: ProcessGroup # group for CPU communication
	device_group: ProcessGroup # group for device communication
	# device communicator (if use_device_communicator=True)
	device_communicator: DeviceCommunicatorBase | None
	mq_broadcaster: Any | None # shared memory broadcaster
```

比如 TP=2, PP=4, 则rank为0-7
ranks：TP的，每个ranks长度为2；PP的，每个ranks长度为4。
local_rank：给每个rank制定gpu，指定在哪一个GPU上allocate。（受CUDA_VISIBLE_DEVICES影响）
通信有两种渠道
- CPU communication，慢，但是可控性更好。
- 基于 device 的，能利用上硬件特性。
	- **NV Link**：GPU和GPU之间的 direct communication。物理上的一个连接；
	- **Infinity Bind**: 也是一个硬件。Between nodes
	- **RDMA**: Remote direct memory access. 可以避免经过 Operating System，不用过CISCO。



