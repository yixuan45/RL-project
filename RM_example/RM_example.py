import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体（黑体）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def g(w, noise_std=0.1):
    """带观测误差的目标函数"""
    true_value = w**3 - 5
    noise = np.random.normal(0, noise_std)  # 高斯观测噪声
    return true_value + noise

def robbins_monro(initial_w, a=1, b=10, max_iter=10000, tol=1e-4, noise_std=0.1):
    """
    含观测噪声的Robbins-Monro算法
    
    参数:
        initial_w: 初始值
        a, b: 步长参数 η_n = a/(n + b)
        max_iter: 最大迭代次数
        tol: 收敛阈值（基于滑动平均）
        noise_std: 观测噪声标准差
    """
    w = initial_w
    w_history = [w]  # 记录迭代历史
    
    for n in range(1, max_iter + 1):
        eta = a / (n + b)  # 递减步长
        noisy_g = g(w, noise_std)  # 带噪声的函数值
        w = w - eta * noisy_g
        
        w_history.append(w)
        
        # 滑动平均判断收敛（减少噪声影响）
        if len(w_history) > 100:
            recent_mean = np.mean(w_history[-100:])
            if np.abs(w - recent_mean) < tol:
                print(f"迭代{n}次后收敛（滑动平均）")
                return w, w_history
    
    print("达到最大迭代次数")
    return w, w_history

# 运行算法
root, history = robbins_monro(initial_w=2.0, noise_std=0.2)
print(f"方程w³-5=0的根近似值: {root:.6f}")
print(f"真实根: {5**(1/3):.6f}")
print(f"带噪声验证: g({root:.6f}) = {g(root, 0):.6f}（无噪声）")

# 可视化迭代过程
import matplotlib.pyplot as plt
plt.plot(history, label='迭代值')
plt.axhline(y=5**(1/3), color='r', linestyle='--', label='真实根')
plt.xlabel('迭代次数')
plt.ylabel('w值')
plt.legend()
plt.title('Robbins-Monro算法迭代过程（含观测噪声）')
plt.show()