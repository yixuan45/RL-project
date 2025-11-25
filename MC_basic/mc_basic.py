import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
from tqdm import tqdm

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 环境参数
GRID_SIZE = 5
FORBIDDEN_STATES = [(1,1), (1,2), (2,2), (3,1), (3,3)]  # 禁止区域（黄色）
TARGET_STATE = (3,2)  # 目标位置（青色）
REWARD_BOUNDARY = -1  # 边界奖励
REWARD_FORBIDDEN = -1  # 禁止区域奖励（进入后会受到惩罚，但可以通过）
REWARD_TARGET = 1  # 目标奖励
GAMMA = 0.9  # 折扣因子
ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)]  # 上、下、左、右（动作空间）
ACTION_SYMBOLS = ['↑', '↓', '←', '→']  # 动作符号用于可视化


def init_policy():
    """初始化随机策略：每个状态的所有动作概率均等"""
    policy = np.ones([GRID_SIZE, GRID_SIZE, len(ACTIONS)]) / len(ACTIONS)
    return policy


def generate_episode_from(s, a, policy, max_steps):
    """从指定状态s和动作a出发，遵循当前策略生成一条轨迹（允许经过禁止区域）"""
    episode = [] # 存储了一条trajectory的信息
    current_state = s
    
    # 第一步：强制执行初始动作a
    dx, dy = ACTIONS[a]
    next_state = (current_state[0] + dx, current_state[1] + dy)
    
    # 计算第一步的奖励（核心修改：允许进入禁止区域，不强制停留在原地）
    if next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
        reward = REWARD_BOUNDARY
        next_state = current_state  # 撞墙后仍停在原地（边界无法通过）
    elif next_state == TARGET_STATE:
        reward = REWARD_TARGET
    elif next_state in FORBIDDEN_STATES:
        reward = REWARD_FORBIDDEN  # 进入禁止区域受惩罚，但可以继续移动
        # 不再强制停留在原地，允许进入禁止区域
    else:
        reward = 0  # 普通格子无奖励
    
    episode.append((current_state, a, reward))
    current_state = next_state  # 即使进入禁止区域，也更新当前状态
    
    # 后续步骤：按当前策略选择动作（允许在禁止区域内继续行动）
    for _ in range(max_steps - 1):
        if current_state == TARGET_STATE:
            break  # 到达目标终止
        
        # 根据当前策略选择动作（即使在禁止区域也能选择动作）
        action_probs = policy[current_state[0], current_state[1]]
        action = np.random.choice(len(ACTIONS), p=action_probs)
        dx, dy = ACTIONS[action]
        
        # 计算下一个状态和奖励（允许经过禁止区域）
        next_state = (current_state[0] + dx, current_state[1] + dy)
        if next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
            reward = REWARD_BOUNDARY
            next_state = current_state  # 边界无法通过，停在原地
        elif next_state == TARGET_STATE:
            reward = REWARD_TARGET
        elif next_state in FORBIDDEN_STATES:
            reward = REWARD_FORBIDDEN  # 进入禁止区域受惩罚，但可以继续移动
        else:
            reward = 0
        
        episode.append((current_state, action, reward))
        current_state = next_state  # 无论是否在禁止区域，都更新当前状态
    
    return episode


def mc_basic(iterations_num, episodes_per_sa, max_steps):
    """实现伪代码中的MC Basic算法"""
    policy = init_policy()
    q_table = np.zeros([GRID_SIZE, GRID_SIZE, len(ACTIONS)])
    returns = {(i, j, a): [] for i in range(GRID_SIZE) 
               for j in range(GRID_SIZE) 
               for a in range(len(ACTIONS))}
    
    for k in range(iterations_num):
        print(f"\n===== 策略迭代第 {k+1}/{iterations_num} 轮 =====")
        
        # 1. 策略评估：估计每个(s,a)的动作价值q(s,a)
        print("开始策略评估（估计动作价值q(s,a)...）")
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                s = (i, j)
                # 跳过目标区域（禁止区域仍需评估，因为可以经过）
                if s == TARGET_STATE:
                    continue
                # 对每个状态s的所有可能动作a进行评估（包括禁止区域的状态）
                for a in range(len(ACTIONS)):
                    # 从(s,a)出发采样多个轨迹
                    for _ in tqdm(range(episodes_per_sa), 
                                  desc=f"状态({i},{j}) 动作{a}[{ACTION_SYMBOLS[a]}] 采样"):
                        episode = generate_episode_from(s, a, policy, max_steps)
                        G = 0  # 回报初始化
                        # 反向遍历轨迹计算回报
                        for t in reversed(range(len(episode))):
                            state_t, action_t, reward_t = episode[t]
                            G = reward_t + GAMMA * G
                            # 首次访问(s,a)时记录回报
                            if (state_t, action_t) == (s, a) and (s, a) not in [(x[0], x[1]) for x in episode[:t]]:
                                returns[(i, j, a)].append(G)
                                q_table[i, j, a] = np.mean(returns[(i, j, a)])
        
        # 2. 策略改进：基于q_table进行贪婪更新（包括禁止区域的策略）
        print("开始策略改进（贪婪更新策略...）")
        policy_stable = True
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                s = (i, j)
                if s == TARGET_STATE:
                    continue  # 目标区域无策略
                # 即使是禁止区域，也更新策略（因为可以经过）
                old_best_a = np.argmax(policy[i, j, :])
                new_best_a = np.argmax(q_table[i, j, :])
                policy[i, j, :] = 0
                policy[i, j, new_best_a] = 1
                if old_best_a != new_best_a:
                    policy_stable = False
        
        if policy_stable:
            print(f"策略在第 {k+1} 轮迭代后收敛！")
            break
    
    return q_table, policy


def plot_results(q_table, policy):
    """可视化动作价值q(s,a)和最终策略"""
    v_table = np.zeros([GRID_SIZE, GRID_SIZE])
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            s = (i, j)
            if s == TARGET_STATE:
                v_table[i, j] = 0
            else:
                v_table[i, j] = np.max(q_table[i, j, :])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            s = (i, j)
            if s == TARGET_STATE:
                text = "目标(T)\n价值: --"
                color = "cyan"
            elif s in FORBIDDEN_STATES:
                # 禁止区域显示价值和最优动作（因为可以经过）
                best_a = np.argmax(policy[i, j, :])
                text = f"禁止(F)\n价值: {v_table[i,j]:.1f}\n最优动作: {ACTION_SYMBOLS[best_a]}"
                color = "orange"
            else:
                best_a = np.argmax(policy[i, j, :])
                text = f"价值: {v_table[i,j]:.1f}\n最优动作: {ACTION_SYMBOLS[best_a]}"
                color = "white"
            
            cell = tb.add_cell(
                i, j, 1/GRID_SIZE, 1/GRID_SIZE, 
                text=text, loc='center', 
                facecolor=color, edgecolor='black'
            )
            cell.get_text().set_fontsize(10)
    
    ax.add_table(tb)
    plt.title("MC Basic算法最终结果（允许经过禁止区域）")
    plt.show()


def run_experiment():
    """运行实验：参数可根据需要调整"""
    iterations_num = 5  # 策略迭代次数
    episodes_per_sa = 1000  # 每个(s,a)对采样的轨迹数量
    max_steps = 30  # 每条轨迹的最大长度
    
    q_table, policy = mc_basic(iterations_num, episodes_per_sa, max_steps)
    plot_results(q_table, policy)


if __name__ == "__main__":
    run_experiment()