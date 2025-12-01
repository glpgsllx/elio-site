vllm/v1/core/block_pool.py

Paged Attention

当把request打包的时候，比如一个request 1GB，一共有10GB，但是却只能打5个包。因为在推理的时候会变长，内存管理会很混乱

所以进行切分，也就是Paged Attention。更好地管理显存空间

