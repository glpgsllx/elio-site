# **1. 普通注意力的公式（标准版）**

标准 self-attention：

$$
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

复杂度：$O(n^2d)$，因为 $QK^\top$ 是一个 $n\times n$ 的矩阵。
# **2. linear attention 想要什么？**
想把注意力写成：

$$
\mathrm{Attn}(Q,K,V)_t = f(Q_t)^\top\, S_t
$$

其中 $S_t$ 是可累加的状态：

$$
S_t = \sum_{j=1}^{t} g(K_j, V_j)
$$

- 每一步只更新一次 $S_t$
- 然后query只做一次 $f(Q_t)^\top S_t$

所以能从 $O(n^2)$ 变成 $O(n)$。

# **3. linear attention 的核分解（kernel trick）**

softmax 非常黏，不可分解。
所以把 softmax attention 近似成一个 **可分解核**：

$$
\exp(QK^\top) \approx \phi(Q) \phi(K)^\top
$$

于是注意力变成：

$$
\mathrm{Attn}(Q,K,V)_t = \sum_{j=1}^{t} \phi(Q_t)^\top \phi(K_j)\, V_j
$$

把 $\phi(Q_t)$ 提出来：

$$
\mathrm{Attn}(Q,K,V)_t = \phi(Q_t)^\top \left( \sum_{j=1}^{t} \phi(K_j) V_j \right)
$$
现在定义：

$$
S_t = \sum_{j=1}^{t} \phi(K_j) V_j
$$

最终得到：

$$
\mathrm{Attn}(Q,K,V)_t = \phi(Q_t)^\top S_t
$$
这就是 linear attention 最核心的结构。
# **4. 为什么它是线性的？**

对每个新 token，更新状态：

$$
S_t = S_{t-1} + \phi(K_t) V_t
$$

计算输出：

$$
O_t = \phi(Q_t)^\top S_t
$$

两步都是 $O(d^2)$ 的，整个序列长度 $n$ → 总复杂度 $O(nd^2)$。  
相比原本的 $O(n^2d)$，省了一个 $n$。
# ** 5. 为什么“把位置信息加到 K/V 里会破坏 linear attention”？**

因为 linear attention 的累加项要求：

$$
S_t = \sum_{j=1}^{t} \phi(K_j) V_j
$$

必须满足：

- $\phi(K_j)$ 可累加
- $\phi(K_j)$ 不随着 t 再发生变化
- 位置编码不能让 $K_j$ 变得**不可分解**
如果你把绝对位置编码直接加到 K 上：

$$
K_j' = K_j + \text{pos}_j
$$

那么：

$$
S_t = \sum_{j=1}^t \phi(K_j + \text{pos}_j)V_j
$$

这个结构通常 **不可分解**，也不能写成“内容 × 位置”的形式，会导致：

- 状态 S 失效
- prefix 不能缓存
- 必须重新算所有 K_j

于是：**linear attention architecture 无法工作。**

对于绝对位置编码，通常 $\phi(K_j + p_j)\neq \phi(K_j)+\phi(p_j)$，所有位置信息会混在一起。  
对于相对位置编码，$\mathrm{score}(t,j)=Q_t^\top K_j + Q_t^\top a_{t-j}$，要对 $a_{t-j}$ 做实时更新，**累加状态无法构建**。
