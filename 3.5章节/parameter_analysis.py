import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 1. 定义网格世界环境（修正目标状态位置）
# --------------------------
class GridWorldEnv:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size  # 5x5网格（书中标准环境）
        self.n_states = grid_size * grid_size
        self.n_actions = 5  # 上、右、下、左、停留（a1-a5）
        # 修正：右下角目标状态的1维索引 = (grid_size-1)*grid_size + (grid_size-1)
        self.target_state = (grid_size - 1) * grid_size + (grid_size - 1)  # 5x5时为 4*5+4=24（正确）
        self.forbidden_states = [6, 11, 13]  # 禁止区域（可自定义，索引0~24）
        self.boundary_penalty = -1  # 边界惩罚
        self.forbidden_penalty = -1  # 禁止区域惩罚（可修改）
        self.target_reward = 1  # 目标奖励
        self.other_reward = 0  # 其他动作奖励

    def state_to_pos(self, s):
        """状态索引转坐标 (row, col)"""
        return s // self.grid_size, s % self.grid_size

    def pos_to_state(self, row, col):
        """坐标转状态索引"""
        return row * self.grid_size + col

    def get_reward(self, s, a):
        """计算动作a在状态s的即时奖励"""
        row, col = self.state_to_pos(s)
        # 计算下一步坐标
        if a == 0:  # 上
            new_row, new_col = row - 1, col
        elif a == 1:  # 右
            new_row, new_col = row, col + 1
        elif a == 2:  # 下
            new_row, new_col = row + 1, col
        elif a == 3:  # 左
            new_row, new_col = row, col - 1
        else:  # 停留（a5）
            new_row, new_col = row, col

        # 边界判断（超出网格则惩罚）
        if new_row < 0 or new_row >= self.grid_size or new_col < 0 or new_col >= self.grid_size:
            return self.boundary_penalty
        # 禁止区域判断（进入则惩罚）
        new_s = self.pos_to_state(new_row, new_col)
        if new_s in self.forbidden_states:
            return self.forbidden_penalty
        # 目标判断（到达则奖励）
        if new_s == self.target_state:
            return self.target_reward
        # 其他情况（无奖励无惩罚）
        return self.other_reward

    def get_transition(self, s, a):
        """状态转移（确定性：动作生效，边界/禁止区域不反弹，直接停在原位置）"""
        row, col = self.state_to_pos(s)
        if a == 0: new_row, new_col = row - 1, col
        elif a == 1: new_row, new_col = row, col + 1
        elif a == 2: new_row, new_col = row + 1, col
        elif a == 3: new_row, new_col = row, col - 1
        else: new_row, new_col = row, col  # 停留

        # 边界检查：超出则留在原状态
        if new_row < 0 or new_row >= self.grid_size or new_col < 0 or new_col >= self.grid_size:
            return s
        # 禁止区域检查：进入则留在原状态（避免直接踏入惩罚区域）
        new_s = self.pos_to_state(new_row, new_col)
        if new_s in self.forbidden_states:
            return s
        # 正常转移到新状态
        return new_s

# --------------------------
# 2. 价值迭代算法（求解最优策略）
# --------------------------
def value_iteration(env, gamma=0.9, theta=1e-6):
    """价值迭代求解最优状态值v*和最优策略pi*"""
    v = np.zeros(env.n_states)  # 初始状态值（所有状态值为0）
    while True:
        delta = 0  # 记录本次迭代中状态值的最大变化
        for s in range(env.n_states):
            # 计算当前状态s下所有动作的Q值
            q_values = []
            for a in range(env.n_actions):
                r = env.get_reward(s, a)  # 即时奖励
                s_prime = env.get_transition(s, a)  # 下一状态
                q = r + gamma * v[s_prime]  # Q值公式（即时奖励+折扣后未来价值）
                q_values.append(q)
            v_new = max(q_values)  # 最优状态值=最大Q值
            delta = max(delta, abs(v_new - v[s]))  # 更新最大变化
            v[s] = v_new  # 更新状态值
        if delta < theta:  # 变化足够小，收敛
            break
    # 提取最优策略（贪心策略：选择Q值最大的动作）
    pi = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        q_values = []
        for a in range(env.n_actions):
            r = env.get_reward(s, a)
            s_prime = env.get_transition(s, a)
            q = r + gamma * v[s_prime]
            q_values.append(q)
        best_a = np.argmax(q_values)  # 最优动作
        pi[s, best_a] = 1.0  # 策略矩阵：最优动作设为1，其余为0
    return v, pi

# --------------------------
# 3. 参数敏感分析实验（匹配书中3.5章节）
# --------------------------
def parameter_sensitivity_analysis():
    # 实验参数设置（覆盖书中关键场景：折扣率、惩罚强度）
    gamma_list = [0.0, 0.5, 0.9, 0.99]  # 折扣率（短视→远视）
    forbidden_penalty_list = [-1, -10, -100]  # 禁止区域惩罚（弱→强）
    env = GridWorldEnv(grid_size=5)  # 初始化5x5网格环境

    # 创建子图（2行4列：上排折扣率分析，下排惩罚分析，最后一个放颜色条）
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("参数敏感分析（修正目标状态位置）", fontsize=16)

    # 自定义颜色映射（负价值→正价值：红→黄→绿）
    cmap = LinearSegmentedColormap.from_list("custom", ["#e74c3c", "#f39c12", "#2ecc71"])
    v_min, v_max = -20, 5  # 颜色条范围（适配惩罚强度变化）

    # 3.1 折扣率γ敏感性分析（上排4个子图）
    for idx, gamma in enumerate(gamma_list):
        v_opt, pi_opt = value_iteration(env, gamma=gamma)  # 求解最优价值和策略
        v_grid = v_opt.reshape(env.grid_size, env.grid_size)  # 转为2维网格用于可视化
        ax = axes[0, idx]
        im = ax.imshow(v_grid, cmap=cmap, vmin=v_min, vmax=v_max)
        ax.set_title(f"折扣率γ={gamma}", fontsize=12)
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        # 标注每个格子的状态值和禁止区域（×）
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                s = env.pos_to_state(i, j)
                if s in env.forbidden_states:
                    ax.text(j, i, "×", ha="center", va="center", fontsize=14, color="black", fontweight="bold")
                elif s == env.target_state:
                    ax.text(j, i, f"★\n{v_grid[i,j]:.1f}", ha="center", va="center", fontsize=10, color="red")
                else:
                    ax.text(j, i, f"{v_grid[i,j]:.1f}", ha="center", va="center", fontsize=10)

    # 3.2 禁止区域惩罚敏感性分析（下排前3个子图）
    for idx, penalty in enumerate(forbidden_penalty_list):
        env.forbidden_penalty = penalty  # 更新惩罚强度
        v_opt, pi_opt = value_iteration(env, gamma=0.9)  # 固定折扣率=0.9
        v_grid = v_opt.reshape(env.grid_size, env.grid_size)
        ax = axes[1, idx]
        im = ax.imshow(v_grid, cmap=cmap, vmin=v_min, vmax=v_max)
        ax.set_title(f"禁止区域惩罚={penalty}", fontsize=12)
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        # 标注每个格子的状态值和禁止区域（×）
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                s = env.pos_to_state(i, j)
                if s in env.forbidden_states:
                    ax.text(j, i, "×", ha="center", va="center", fontsize=14, color="black", fontweight="bold")
                elif s == env.target_state:
                    ax.text(j, i, f"★\n{v_grid[i,j]:.1f}", ha="center", va="center", fontsize=10, color="red")
                else:
                    ax.text(j, i, f"{v_grid[i,j]:.1f}", ha="center", va="center", fontsize=10)

    # 隐藏最后一个多余子图，添加全局颜色条
    axes[1, 3].axis("off")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label("最优状态值v*", fontsize=12)

    plt.tight_layout()
    plt.show()

# --------------------------
# 4. 运行实验
# --------------------------
if __name__ == "__main__":
    parameter_sensitivity_analysis()