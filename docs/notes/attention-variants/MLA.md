From DeepSeek-V2

> MLA, which utilizes low-rank key-value joint compression to eliminate the bottleneck of inference-time key-value cache, thus supporting efficient inference.

![MLA overview](img/Pasted image 20251125151259.png)
MHA->heavy KV cache
-> MQA and GQA, require a smaller magnitude of KV Cache, but performance does not match MHA
![MHA vs MQA vs GQA](img/Pasted image 20251125153732.png)

# 1. MHA
$h_t \in \mathbb{R}^d$, the attention input of the t-th token at an attn layer.
$q_t, k_t, v_t \in \mathbb{R}^{d_h n_h}$ , 
$W^Q, W^K, W^V \in \mathbb{R}^{d_h n_h \times d}$,
$d_h$ is the dimension per head,
$n_h$ is the number of heads.
$$
q_t = W^Q h_t,
k_t = W^K h_t,
v_t = W^V h_t
$$
then
![KV cache cost](img/Pasted image 20251125152004.png)
$W^O \in \mathbb{R}^{d \times d_h n_h}$ denotes the output proj matrix
All k and v need to be cached, so MHA needs to cache $$
2 n_h d_h l 
$$
elements for each token.
> In model deployment, this heavy KV cache is a large bottleneck that limits the maximum batch size and sequence length.

# 2. Low-Rank K-V Joint Compression
The core of MLA is the low-rank joint compression for keys and values to reduce KV cache.
$$
\begin{gather}
c_t^{KV} = W^{DKV}h_t, \; \in \mathbb{R}^{d_c} \\
k_t^{C} = W^{UK}c_t^{KV}, \; \in \mathbb{R}^{d_n h_n} \\
v_t^{C} = W^{UV}c_t^{KV}, \; \in \mathbb{R}^{d_n h_n} 
\end{gather}
$$
第一步是压缩，$d_c << d_h n_h$
第二步是 up_projection，恢复成 $d_h n_h$
存储的时候只要存 $c_t^{KV}, \quad d_c l$ elements, $l$ denotes the number of layers
> $𝑊^{𝑈𝐾}$ can be absorbed into $𝑊^𝑄$, and $𝑊^{𝑈𝑉}$ can be absorbedinto $𝑊^𝑂$

训练的时候，为了 reduce the activation memory, 对query 也使用 low-rank compression, 即使不会减少 KV cache
$$
\begin{gather}
c_t^{Q} = W^{DQ}h_t, \; \in \mathbb{R}^{d_c'} \\
q_t^{C} = W^{UQ}c_t^{Q}, \; \in \mathbb{R}^{d_n h_n} \\
\end{gather}
$$
$d_c' << d_h n_h$
# 3. Decoupled Rotary Position Embedding




