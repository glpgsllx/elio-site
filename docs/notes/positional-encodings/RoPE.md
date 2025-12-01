> Despite the effectiveness of these approaches, they commonly add the position information to the context representation and thus render them unsuitable for the linear self-attention architecture. [Linear-Attn(org)](../attention-variants/Linear-Attn(org).md)

3个特性

- sequence length flexibility
- decaying inter-token dependency with increasing relative distances
- capability of equipping the linear self-attention with relative position encoding

核心公式：
$$
q_m = f_q(x_m, m), \quad k_n = f_k(x_n, n), \quad v_n = f_v(x_n. n)
$$

# 1. Absolute
![Absolute positional encoding](img/Pasted image 20251125162111.png)
$i = 0, 1, 2, \dots, \text{sequence length}-1$
$t = 0, 1, 2, \dots, \frac{d}{2}-1$
$k = i$


# 2. Relative
## 2.1 Self-attention with relative position representations
$$
\begin{aligned}
f_q(x_m) := W_qx_m \\
f_k(x_n, n) := W_k(x_n + p_r^k)\\
f_v(x_n, n) := W_v(x_n + p_r^v)\\
\end{aligned}
$$
$p_r^k, p_r^v \in \mathbb{R}^d$ are trainable relative position embeddings.
$r = clip (m−n,r_\min,r_\max)$ represents the relative distance between position m and n.
## 2.2 Further optimization
change $q_m^{\top} k_n$
很多相对位置编码方法，本质上都是把位置 embedding（无论是绝对位置 $p_i$ 还是相对位置 $p_{i−j}$ 直接加到 Q、K、V 里面

其中效率最高的是 
> Deberta: Decoding-enhanced bert with disentangled attention.

$$
q_m^\top k_n
= x_m^\top W_q^\top W_k x_n
+ x_m^\top W_q^\top W_k \,\tilde p_{\,m-n}
+ \tilde p_{\,m-n}^\top W_q^\top W_k x_n .
$$

- $\tilde p_{m-n}$ 是相对位置 embedding（DeBERTa 使用 disentangled 方式），
- 第一项是“内容–内容”注意力，
- 第二项是“内容–位置”，
- 第三项是“位置–内容”。

属于“加法型位置注入”，对后续的线性化、自回归缓存等机制依然不友好[[Linear-Attn(org)]]

# 3. RoPE
希望注意力分数 qₘᵀkₙ 不要依赖 token 各自的绝对位置 m、n，而只依赖它们的“相对位置 m−n“

定义一个函数，只考虑 2 个词向量和位置差：

$$
<f_q(x_m, m), f_k(x_n, n)> = g(x_m, x_n, m - n)
$$

## **3.1 2D case**
在二维情形（d = 2）下，可以利用复数平面上的几何性质来展示 RoPE 的核心思想。
### **复数形式的 Query / Key**

令位置为 m、n 的向量经过线性变换后分别为：
$W_q x_m$ 和 $W_k x_n$

RoPE 将它们表示为复数并按位置进行旋转：

$$
f_q(x_m, m) = (W_q x_m) e^{im\theta} \\
f_k(x_n, n) = (W_k x_n) e^{in\theta}
$$

其中 $\theta$ 是预设的非零常数。
### **相对位置如何出现？**
注意力的核心是内积：
$$
g(x_m, x_n, m-n) = \mathrm{Re}\big[(W_q x_m)(W_k x_n)^* e^{i(m-n)\theta}\big]
$$
这里的 $\mathrm{Re}(\cdot)^*$ 表示复共轭（$a+bi$ 的 conjugate complex number 是 $a-bi$），其作用是让 Key 的旋转方向相反（加变成减），从而得到 $m-n$。
### **旋转矩阵形式**
二维复数旋转等价于旋转矩阵
q的旋转矩阵：
$$

f_{{q,k}}(x_m, m) =

\begin{pmatrix}

\cos m\theta & -\sin m\theta \

\sin m\theta & \cos m\theta

\end{pmatrix}

\begin{pmatrix}

W^{(11)}_{{q,k}} & W^{(12)}_{{q,k}} \

W^{(21)}_{{q,k}} & W^{(22)}_{{q,k}}

\end{pmatrix}

\begin{pmatrix}

x_m^{(1)} \

x_m^{(2)}

\end{pmatrix}

$$
k的，只要把m换成n

**在二维空间里，对词向量做一次线性变换后，再按“位置 index × 固定角度”旋转它。**

**这样 qᵀk 的结果自然变成依赖 m−n，于是实现了相对位置编码。**

**这就是 RoPE 的几何直觉：用旋转取代加法，把绝对位置变成相对位置。**

## 3.2 General Form
![RoPE general form](img/Pasted image 20251128132128.png)

$$ 
\Theta = \left\{ \theta_i=10000^{-2(i-1)/d}, i\in \left[ 1,2,\dots, d/2 \right] \right\}
$$
把这个式子放在$q_m^{\top} k_n$ 上
$$
q_m^{\top} k_n = (R_{\Theta, m}^d W_qx_m)^{\top}(R_{\Theta, n}^d W_kx_n) = x^{\top}W_q R_{\Theta, (n-m)}^d W_kx_n
$$
**以前的位置编码把位置 “加” 到内容上，会污染语义，并且要在注意力公式里额外加入复杂的相对位置项。**

**RoPE 使用的是 “乘法式位置编码”（旋转），无需修改注意力结构，相对位置 (n−m) 会自然在旋转中出现。**

## 3.3 Properties
### 3.3.1 Long-term decay

### 3.3.2 Linear Attention
![RoPE with linear attention](img/Pasted image 20251128134804.png)
用线性函数，比如 $\phi(x) = \mathrm{elu}(x) + 1$
分母不变，防止为0
