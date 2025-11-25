import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
from tqdm import tqdm

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 环境参数（保持不变）
GRID_SIZE = 5
FORBIDDEN_STATES = [(1,1), (1,2), (2,2), (3,1), (3,3)]
TARGET_STATE = (3,2)
REWARD_BOUNDARY = -1
REWARD_FORBIDDEN = -1
REWARD_TARGET = 1
GAMMA = 0.9
ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)]  # 上、下、左、右
ACTION_SYMBOLS = ['↑', '↓', '←', '→']
NUM_ACTIONS = len(ACTIONS)  # 动作空间大小 |A(s)|


def init_policy_and_q(epsilon=0.1):
    """初始化ε-贪婪策略和动作价值q(s,a)"""
    # 初始策略：所有动作等概率（符合ε-贪婪的初始状态）
    policy = np.ones([GRID_SIZE, GRID_SIZE, NUM_ACTIONS]) / NUM_ACTIONS
    # 初始动作价值：全0
    q_table = np.zeros([GRID_SIZE, GRID_SIZE, NUM_ACTIONS])
    # 记录总回报和访问次数（替代原returns字典）
    returns_sum = np.zeros([GRID_SIZE, GRID_SIZE, NUM_ACTIONS])
    returns_count = np.zeros([GRID_SIZE, GRID_SIZE, NUM_ACTIONS])
    return policy, q_table, returns_sum, returns_count


def select_start_state():
    """选择起始状态（无需强制指定起始动作，按策略选择）"""
    while True:
        s0 = (np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE))
        if s0 != TARGET_STATE:  # 排除目标状态
            return s0


def generate_episode(start_state, policy, max_steps):
    """从起始状态出发，按当前策略生成轨迹（无需固定起始动作）"""
    episode = []
    current_state = start_state
    
    # 第一步动作：按当前策略选择（非固定）
    action_probs = policy[current_state[0], current_state[1]]
    action = np.random.choice(NUM_ACTIONS, p=action_probs)
    dx, dy = ACTIONS[action]
    next_state = (current_state[0] + dx, current_state[1] + dy)
    
    # 计算奖励
    if next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
        reward = REWARD_BOUNDARY
        next_state = current_state
    elif next_state == TARGET_STATE:
        reward = REWARD_TARGET
    elif next_state in FORBIDDEN_STATES:
        reward = REWARD_FORBIDDEN
    else:
        reward = 0
    
    episode.append((current_state, action, reward))
    current_state = next_state
    
    # 后续步骤：按策略选择动作
    for _ in range(max_steps - 1):
        if current_state == TARGET_STATE:
            break
        
        action_probs = policy[current_state[0], current_state[1]]
        action = np.random.choice(NUM_ACTIONS, p=action_probs)
        dx, dy = ACTIONS[action]
        next_state = (current_state[0] + dx, current_state[1] + dy)
        
        # 奖励计算逻辑同上
        if next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
            reward = REWARD_BOUNDARY
            next_state = current_state
        elif next_state == TARGET_STATE:
            reward = REWARD_TARGET
        elif next_state in FORBIDDEN_STATES:
            reward = REWARD_FORBIDDEN
        else:
            reward = 0
        
        episode.append((current_state, action, reward))
        current_state = next_state
    
    return episode


def mc_epsilon_greedy(total_episodes, max_steps, epsilon=0.1):
    """实现MC ε-Greedy算法（Algorithm 5.3）"""
    policy, q_table, returns_sum, returns_count = init_policy_and_q(epsilon)
    
    for episode_idx in tqdm(range(total_episodes), desc="训练进度"):
        # 1. 选择起始状态（无需固定起始动作）
        start_state = select_start_state()
        
        # 2. 生成轨迹（按当前ε-贪婪策略）
        episode = generate_episode(start_state, policy, max_steps)
        
        # 3. 反向遍历轨迹更新价值和策略
        G = 0  # 累积回报
        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t = episode[t]
            G = GAMMA * G + reward_t  # 计算回报（g = γ*g + r_{t+1}）
            
            # 4. 策略评估：更新动作价值q(s,a)
            returns_sum[state_t[0], state_t[1], action_t] += G
            returns_count[state_t[0], state_t[1], action_t] += 1
            q_table[state_t[0], state_t[1], action_t] = (
                returns_sum[state_t[0], state_t[1], action_t] / 
                returns_count[state_t[0], state_t[1], action_t]
            )
            
            # 5. 策略改进：更新为ε-贪婪策略
            best_action = np.argmax(q_table[state_t[0], state_t[1], :])  # a* = argmax q(s,a)
            # 计算每个动作的概率
            for a in range(NUM_ACTIONS):
                if a == best_action:
                    # 最优动作概率：1-ε + ε/|A(s)|
                    policy[state_t[0], state_t[1], a] = 1 - epsilon + epsilon / NUM_ACTIONS
                else:
                    # 其他动作概率：ε/|A(s)|
                    policy[state_t[0], state_t[1], a] = epsilon / NUM_ACTIONS
    
    return q_table, policy


def plot_results(q_table, policy):
    """可视化结果（保持不变）"""
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
    plt.title("MC ε-Greedy算法最终结果")
    plt.show()


def run_experiment():
    """运行实验：参数可调整"""
    total_episodes = 100000  # 总轨迹数量
    max_steps = 30  # 每条轨迹最大长度
    epsilon = 0.1  # ε值（探索概率）
    
    q_table, policy = mc_epsilon_greedy(total_episodes, max_steps, epsilon)
    plot_results(q_table, policy)


if __name__ == "__main__":
    run_experiment()