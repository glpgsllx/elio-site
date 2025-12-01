常见的 Entrypoint 有两种
- LLM vllm/entrypoints/llm.py
- API Server vllm/entrypoints/openai/api_server.py

API Server 里
```
def mount_metrics(app: FastAPI):
```

发送给 Engine