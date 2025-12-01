import numpy as np
import matplotlib.pyplot as plt
from math import tanh

# 1. 定义函数与参数
g = lambda w: tanh(w - 1)  # 目标函数g(w)
w = [3.0]  # 初始值w1=3
max_iter = 10  # 只需少量迭代（图中显示w1到w4）
a_k_list = [1/k for k in range(1, max_iter+1)]  # 步长a_k=1/k

# 2. 迭代计算
for k in range(max_iter-1):
    current_w = w[-1]
    a_k = a_k_list[k]
    next_w = current_w - a_k * g(current_w)
    w.append(next_w)

# 3. 绘制函数g(w) + 迭代路径
plt.figure(figsize=(10, 6))
# 绘制g(w)曲线
w_range = np.linspace(0.5, 4, 200)
g_range = np.tanh(w_range - 1)
plt.plot(w_range, g_range, 'k-', label='$g(w)=\tanh(w-1)$')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)  # 横轴
plt.axvline(x=1, color='r', linestyle='--', label='$w^*=1$')  # 根的位置

# 绘制迭代路径（w1→w2→…→wk）
for i in range(len(w)-1):
    # 点(w_i, g(w_i))到(w_i, 0)的竖线
    plt.plot([w[i], w[i]], [0, g(w[i])], 'gray', linestyle='--', alpha=0.5)
    # 点(w_i, 0)到(w_{i+1}, 0)的横线（对应w的更新）
    plt.plot([w[i], w[i+1]], [0, 0], 'gray', linestyle='--', alpha=0.5)
    # 标记w_i的点
    plt.plot(w[i], 0, 'o', color='black')
    plt.text(w[i]+0.05, -0.1, f'$w_{i+1}$', fontsize=10)

# 标记最后一个点
plt.plot(w[-1], 0, 'o', color='black')
plt.text(w[-1]+0.05, -0.1, f'$w_{len(w)}$', fontsize=10)

# 图形设置
plt.xlabel('$w$')
plt.ylabel('$g(w)$')
plt.title('RM算法迭代路径（逼近$w^*=1$）')
plt.legend()
plt.xlim(0.5, 4)
plt.ylim(-1, 1.5)
plt.grid(True, alpha=0.3)
plt.show()