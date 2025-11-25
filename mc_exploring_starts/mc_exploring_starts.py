import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
from tqdm import tqdm

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 环境参数（保持与之前一致）
GRID_SIZE = 5
FORBIDDEN_STATES = [(1,1), (1,2), (2,2), (3,1), (3,3)]  # 禁止区域
TARGET_STATE = (3,2)  # 目标位置
REWARD_BOUNDARY = -1  # 边界奖励
REWARD_FORBIDDEN = -1  # 禁止区域奖励
REWARD_TARGET = 10  # 目标奖励（已调整为正向激励）
GAMMA = 0.9  # 折扣因子
ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)]  # 上、下、左、右
ACTION_SYMBOLS = ['↑', '↓', '←', '→']  # 动作符号

def init_policy_and_q():
    """初始化策略和动作价值q(s,a)"""
    # 初始策略：随机策略（每个动作等概率）
    policy = np.ones([GRID_SIZE, GRID_SIZE, len(ACTIONS)]) / len(ACTIONS)
    # 初始动作价值：全0
    q_table = np.zeros([GRID_SIZE, GRID_SIZE, len(ACTIONS)])
    return policy, q_table

def select_exploring_start():
    """探索式选择起始状态-动作对(s0, a0)：确保所有(s,a)都可能被选中"""
    while True:
        # 随机选择状态s0（排除目标区域，允许禁止区域）
        s0 = (np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE))
        if s0 == TARGET_STATE:
            continue  # 目标区域无动作
        # 随机选择动作a0（所有动作都可能被选中）
        a0 = np.random.randint(len(ACTIONS))
        return s0, a0
    
def generate_episode(s0, a0, policy, max_steps):
    """从起始状态-动作对(s0,a0)出发，遵循当前策略生成轨迹"""
    episode = []
    current_state = s0
    
    # 第一步：使用探索式选择的a0
    dx, dy = ACTIONS[a0]
    next_state = (current_state[0] + dx, current_state[1] + dy)
    
    # 计算奖励
    if next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
        reward = REWARD_BOUNDARY
        next_state = current_state  # 撞墙停在原地
    elif next_state == TARGET_STATE:
        reward = REWARD_TARGET
    elif next_state in FORBIDDEN_STATES:
        reward = REWARD_FORBIDDEN
    else:
        # 中间奖励：距离目标越近奖励越高
        distance = abs(next_state[0] - TARGET_STATE[0]) + abs(next_state[1] - TARGET_STATE[1])
        reward = max(0, 5 - distance)  # 0~5的正向奖励
    
    episode.append((current_state, a0, reward))
    current_state = next_state
    
    # 后续步骤：按当前策略选择动作
    for _ in range(max_steps - 1):
        if current_state == TARGET_STATE:
            break  # 到达目标终止
        
        # 根据当前策略选择动作
        action_probs = policy[current_state[0], current_state[1]]
        action = np.random.choice(len(ACTIONS), p=action_probs)
        dx, dy = ACTIONS[action]
        
        # 计算下一个状态和奖励
        next_state = (current_state[0] + dx, current_state[1] + dy)
        if next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
            reward = REWARD_BOUNDARY
            next_state = current_state
        elif next_state == TARGET_STATE:
            reward = REWARD_TARGET
        elif next_state in FORBIDDEN_STATES:
            reward = REWARD_FORBIDDEN
        else:
            distance = abs(next_state[0] - TARGET_STATE[0]) + abs(next_state[1] - TARGET_STATE[1])
            reward = max(0, 5 - distance)
        
        episode.append((current_state, action, reward))
        current_state = next_state
    
    return episode

def mc_exploring_starts(total_episodes, max_steps):
    """实现MC Exploring Starts算法（Algorithm 5.2）"""
    policy, q_table = init_policy_and_q()
    # 记录每个(s,a)的总回报和访问次数
    returns_sum = np.zeros([GRID_SIZE, GRID_SIZE, len(ACTIONS)])  # Returns(s,a)
    returns_count = np.zeros([GRID_SIZE, GRID_SIZE, len(ACTIONS)])  # Num(s,a)
    
    for episode_idx in tqdm(range(total_episodes), desc="总训练进度"):
        # 1. 探索式选择起始状态-动作对
        s0, a0 = select_exploring_start()
        
        # 2. 生成轨迹
        episode = generate_episode(s0, a0, policy, max_steps)
        
        # 3. 反向遍历轨迹，更新回报和策略
        G = 0  # 初始化回报g
        # 从轨迹末尾向前遍历（t = T-1, T-2, ..., 0）
        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t = episode[t]
            # 计算累积回报：g = γ*g + r_{t+1}（注意reward_t是执行action_t后的奖励，即r_{t+1}）
            G = GAMMA * G + reward_t
            
            # 4. 更新动作价值（策略评估）
            returns_sum[state_t[0], state_t[1], action_t] += G
            returns_count[state_t[0], state_t[1], action_t] += 1
            # 用平均值更新q(s,a)：q = Returns / Num
            q_table[state_t[0], state_t[1], action_t] = (
                returns_sum[state_t[0], state_t[1], action_t] / 
                returns_count[state_t[0], state_t[1], action_t]
            )
            
            # 5. 策略改进：贪婪更新当前状态的策略
            best_action = np.argmax(q_table[state_t[0], state_t[1], :])
            policy[state_t[0], state_t[1], :] = 0  # 其他动作概率为0
            policy[state_t[0], state_t[1], best_action] = 1  # 最优动作概率为1
    
    return q_table, policy

def plot_results(q_table, policy):
    """可视化结果（保持与之前一致）"""
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
    plt.title("MC Exploring Starts算法最终结果")
    plt.show()


def run_experiment():
    """运行实验"""
    total_episodes = 100000  # 总轨迹数量（可调整）
    max_steps = 30  # 每条轨迹最大长度
    
    q_table, policy = mc_exploring_starts(total_episodes, max_steps)
    plot_results(q_table, policy)


if __name__ == "__main__":
    run_experiment()