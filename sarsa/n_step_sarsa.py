import numpy as np
import matplotlib.pyplot as plt

# 1.环境初始化（与之前保持一致）
GRID_SIZE = 5
START_STATE = (0, 0)  # 起点
TARGET_STATE = (3, 2)  # 终点
FORBIDDEN_STATES = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)]  # 禁止通行的状态
BOUNDARY_REWARD = -10
FORBIDDEN_REWARD = -10
OTHER_REWARD = -1

# 动作定义：上、下、左、右（对应索引0-3）
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ['上', '下', '左', '右']
NUM_ACTIONS = len(ACTIONS)

# 2.n步Sarsa算法实现（修复核心错误）
class NStepSarsaAgent:
    def __init__(self, n=5, alpha=0.1, gamma=1.0, epsilon=0.1):
        self.n = n  # n步的步长
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # ε-贪心策略参数
        # 初始化动作价值函数q(s,a)
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS))

    def choose_action(self, state):
        """ε-贪心策略选择动作"""
        if np.random.uniform(0, 1) < self.epsilon:
            # 探索：随机选动作
            return np.random.choice(NUM_ACTIONS)
        else:
            # 利用：选q值最大的动作（多个最大值随机选）
            q_values = self.q_table[state[0], state[1], :]
            max_q = np.max(q_values)
            return np.random.choice(np.where(q_values == max_q)[0])

    def update_q_table(self, transitions):
        """根据n步转移序列更新q_table（修复空动作问题）"""
        # transitions是列表，每个元素是(s, a, r)，无终点的空动作元素
        T = len(transitions)  # 终止时间步（最后一个有效转移的索引是T-1）
        t = 0
        # 遍历每个时间步，计算n步返回值并更新
        while t < T:
            # 计算n步的范围：t到t+n（不超过T）
            n_step = min(t + self.n, T)
            # 计算n步返回值G_t(n)
            G = 0.0
            for i in range(t, n_step):
                # 累加奖励：γ^(i-t) * R_{i+1}（transitions[i][2]是执行动作后的奖励）
                G += (self.gamma ** (i - t)) * transitions[i][2]
            # 加上最后一步的γ^k * Q(S_{n_step}, A_{n_step})，其中k是实际的步数（n_step - t）
            k = n_step - t  # 实际走的步数（可能小于n）
            if n_step < T:
                s_n, a_n = transitions[n_step][0], transitions[n_step][1]
                G += (self.gamma ** k) * self.q_table[s_n[0], s_n[1], a_n]
            # 当前状态和动作
            s_t, a_t = transitions[t][0], transitions[t][1]
            # 更新q_table
            self.q_table[s_t[0], s_t[1], a_t] += self.alpha * (G - self.q_table[s_t[0], s_t[1], a_t])
            t += 1

# 3.环境交互函数（与之前保持一致）
def step(state, action):
    """根据当前状态和动作，返回下一个状态和奖励"""
    x, y = state
    dx, dy = ACTIONS[action]
    next_x = x + dx
    next_y = y + dy

    # 判断下一个状态是否合法
    if (next_x, next_y) == TARGET_STATE:
        return (next_x, next_y), 0  # 到达目标状态，奖励为0
    if (next_x, next_y) in FORBIDDEN_STATES:
        return state, FORBIDDEN_REWARD  # 禁止通行，保持原状态，奖励为-10
    if 0 <= next_x < GRID_SIZE and 0 <= next_y < GRID_SIZE:
        return (next_x, next_y), OTHER_REWARD  # 合法移动，奖励为-1
    else:
        return (x, y), BOUNDARY_REWARD  # 越界，保持原状态，奖励为-10

# 4.训练过程（n步Sarsa，修复转移序列存储逻辑）
def train_n_step_sarsa(episodes=500, n=5):
    agent = NStepSarsaAgent(n=n, alpha=0.1, epsilon=0.1)
    total_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        state = START_STATE
        action = agent.choose_action(state)
        total_reward = 0
        length = 0
        # 存储n步的转移序列：[(s0, a0, r1), (s1, a1, r2), ..., (sT-1, aT-1, rT)]（无空动作）
        transitions = []

        while True:
            # 执行动作，得到下一个状态和奖励
            next_state, reward = step(state, action)
            total_reward += reward
            length += 1
            # 存储当前转移（s, a, r），r是执行a后得到的奖励
            transitions.append((state, action, reward))
            # 判断是否到达终点
            if next_state == TARGET_STATE:
                break  # 到达终点，停止存储转移
            # 选择下一个动作
            next_action = agent.choose_action(next_state)
            # 转移状态和动作
            state = next_state
            action = next_action

        # 利用n步转移序列更新q_table
        agent.update_q_table(transitions)
        # 记录每轮的奖励和长度
        total_rewards.append(total_reward)
        episode_lengths.append(length)

        # 每100轮打印进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            print(f"Episode {episode+1}, n-step: {n}, Avg Reward: {avg_reward:.2f}")

    return agent, total_rewards, episode_lengths

# 5.运行训练并可视化结果
if __name__ == "__main__":
    # 可调整n的值，比如n=1（等价于单步Sarsa）、n=3、n=5、n=10
    N_STEP = 5
    agent, total_rewards, episode_lengths = train_n_step_sarsa(episodes=500, n=N_STEP)

    # 绘制奖励和episode长度曲线
    plt.figure(figsize=(12, 5))
    # 总奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(total_rewards)
    plt.title(f"Total Reward per Episode (n={N_STEP})")
    plt.xlabel("Episode Index")
    plt.ylabel("Total Reward")
    # Episode长度曲线
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths)
    plt.title(f"Episode Length per Episode (n={N_STEP})")
    plt.xlabel("Episode Index")
    plt.ylabel("Episode Length")
    plt.tight_layout()
    plt.show()

    # 打印最终策略（每个状态的最优动作）
    print(f"\nFinal Optimal Policy (n={N_STEP}, 0=上,1=下,2=左,3=右):")
    for x in range(GRID_SIZE):
        row = []
        for y in range(GRID_SIZE):
            if (x, y) == TARGET_STATE:
                row.append("T")  # 终点
            elif (x, y) in FORBIDDEN_STATES:
                row.append("F")  # 禁止区域
            else:
                # 选q值最大的动作
                best_action = np.argmax(agent.q_table[x, y, :])
                row.append(str(best_action))
        print(" ".join(row))