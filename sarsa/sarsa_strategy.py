import numpy as np
import matplotlib.pyplot as plt

# 1.环境初始化
GRID_SIZE = 5
START_STATE = (0, 0)  # 起点：左上角（对应图中（1，1））
TARGET_STATE = (3, 2)  # 终点：图中（4，3）
FORBIDDEN_STATES = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)]  # 禁止通行的状态
BOUNDARY_REWARD = -10
FORBIDDEN_REWARD = -10
OTHER_REWARD = -1

# 动作定义：上、下、左、右（对应索引0-3）
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ['上', '下', '左', '右']

# 2.Sarsa算法实现（修复概率求和问题）
class SarsaAgent:
    def __init__(self, alpha=0.1, gamma=1.0, epsilon=0.1):
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # ε-贪婪策略参数
        # 初始化动作价值函数q(s,a)：形状（GRID_SIZE,GRID_SIZE,动作数）
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
        # 新增：初始化策略矩阵pi(s,a)（每个状态下每个动作的选择概率）
        self.pi = np.ones((GRID_SIZE, GRID_SIZE, len(ACTIONS))) / len(ACTIONS)  # 初始均匀分布

    def update_policy(self, state):
        """显式更新策略矩阵pi，修复概率求和问题"""
        s_x, s_y = state
        num_actions = len(ACTIONS)
        # 1. 找到当前状态下q值最大的动作（最优动作）
        q_values = self.q_table[s_x, s_y, :]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        num_best = len(best_actions)

        # 2. 计算策略概率（对应截图公式，优化计算逻辑减少误差）
        # 非最优动作的概率：ε / |A(s)|
        prob_other = self.epsilon / num_actions
        # 最优动作的概率：(1 - ε) / num_best + prob_other （等价于原逻辑，更直观）
        prob_best = (1 - self.epsilon) / num_best + prob_other

        # 3. 初始化当前状态的策略概率
        new_pi = np.full(num_actions, prob_other)
        # 4. 为最优动作分配概率
        new_pi[best_actions] = prob_best
        # 5. 关键修复：归一化处理，确保概率总和严格为1
        new_pi = new_pi / new_pi.sum()

        # 6. 更新策略矩阵
        self.pi[s_x, s_y] = new_pi

    def choose_action(self, state):
        """从策略矩阵pi中采样选择动作"""
        s_x, s_y = state
        # 根据当前状态的动作概率分布采样动作
        return np.random.choice(len(ACTIONS), p=self.pi[s_x, s_y])

    def update_q_table(self, state, action, reward, next_state, next_action):
        """更新动作价值函数q_table（原有逻辑不变）"""
        current_q = self.q_table[state[0], state[1], action]
        next_q = self.q_table[next_state[0], next_state[1], next_action]
        # Sarsa更新公式：q(s,a)=q(s,a)+α[r+γ*q(s',a')-q(s,a)]
        self.q_table[state[0], state[1], action] += self.alpha * (
            reward + self.gamma * next_q - current_q
        )

# 3.环境交互函数（原有逻辑不变）
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

# 4.训练过程（原有逻辑不变）
def train_sarsa(episodes=500):
    agent = SarsaAgent(alpha=0.1, epsilon=0.1)
    total_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        state = START_STATE
        # 初始化时先更新一次当前状态的策略（确保初始动作选择符合策略）
        agent.update_policy(state)
        action = agent.choose_action(state)
        total_reward = 0
        length = 0

        while state != TARGET_STATE:
            # 执行动作，得到下一个状态、奖励
            next_state, reward = step(state, action)
            # 更新下一个状态的策略（为选择下一个动作做准备）
            agent.update_policy(next_state)
            # 选择下一个动作（从更新后的策略中采样）
            next_action = agent.choose_action(next_state)
            # 更新q_table（原有逻辑）
            agent.update_q_table(state, action, reward, next_state, next_action)
            # 更新当前状态的策略（对应截图中的策略更新步骤）
            agent.update_policy(state)

            # 累计信息
            total_reward += reward
            length += 1

            # 转移到下一个状态、动作
            state = next_state
            action = next_action

        total_rewards.append(total_reward)
        episode_lengths.append(length)
        # 每100轮打印进度
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}, Avg Reward: {np.mean(total_rewards[-100:]):.2f}")

    return agent, total_rewards, episode_lengths

# 5.运行训练并可视化结果（原有逻辑不变）
if __name__ == "__main__":
    agent, total_rewards, episode_lengths = train_sarsa(episodes=500)

    # 绘制奖励和 episode 长度曲线
    plt.figure(figsize=(12, 5))
    # 总奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(total_rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode Index")
    plt.ylabel("Total Reward")
    # Episode长度曲线
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths)
    plt.title("Episode Length per Episode")
    plt.xlabel("Episode Index")
    plt.ylabel("Episode Length")
    plt.tight_layout()
    plt.show()

    # 打印最终策略（每个状态的最优动作）
    print("\nFinal Optimal Policy (0=上,1=下,2=左,3=右):")
    for x in range(GRID_SIZE):
        row = []
        for y in range(GRID_SIZE):
            if (x, y) == TARGET_STATE:
                row.append("T")  # 终点
            elif (x, y) in FORBIDDEN_STATES:
                row.append("F")  # 禁止区域
            else:
                # 选q值最大的动作（最优策略）
                best_action = np.argmax(agent.q_table[x, y, :])
                row.append(str(best_action))
        print(" ".join(row))

    # 可选：打印某个状态的策略概率分布（示例）
    print("\n示例：状态(0,0)的策略概率分布（上、下、左、右）：")
    print(agent.pi[0, 0])
    print(f"概率总和：{agent.pi[0,0].sum():.10f}")  # 验证总和是否为1