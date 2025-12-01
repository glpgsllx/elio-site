vllm/v1/attention/backends

最核心看vllm/v1/attention/backends/flash_attn.py

prefill 和 decode 会用不同的 flashattention kernel
prefill的不需要从gpu里拿任何data, flash_attn_varlen_func
decode