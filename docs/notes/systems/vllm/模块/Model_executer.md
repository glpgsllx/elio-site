model executer 里 有一个个小 model runner

最经典：vllm/model_executor/models/llama.py

最核心的是
```
class LlamaDecoderLayer(nn.Module):
	def forward():
```

执行的是 norm + Attn + norm + MLP



