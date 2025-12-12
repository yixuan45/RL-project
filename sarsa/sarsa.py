import numpy as np
import matplotlib.pyplot as plt

# 1.环境初始化
GRID_SIZE=5
START_STATE=(0,0) # 起点：左上角（对应图中（1，1））
TARGET_STATE=(3,2) # 终点：图中（4，3）
FORBIDDEN_STATES=[(1,1),(1,2),(2,2),(3,1),(3,3),(4,1)] # 禁止通行的状态
BOUNDARY_REWARD = -10
FORBIDDEN_REWARD = -10
OTHER_REWARD = -1

# 动作定义：上、下、左、右（对应缩影0-3）
ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]
ACTION_NAMES=['上', '下', '左', '右']

#2.Sarsa算法实现
class SarsaAgent:
    def __init__(self,alpha=0.1,gamma=1.0,epsilon=0.1):
        self.alpha=alpha # 学习率
        self.gamma=gamma # 折扣因子
        self.epsilon=epsilon # ε-贪婪策略参数
        # 初始化动作价值函数q(s,a)：形状（GRID_SIZE,GRID_SIZE,动作数）
        self.q_table=np.zeros((GRID_SIZE,GRID_SIZE,len(ACTIONS)))
        
    def choose_action(self,state):
        """ε-贪心策略选择动作"""
        if np.random.uniform(0,1)<self.epsilon:
            return np.random.choice(len(ACTIONS)) # 探索
        else:
            # 利用：选当前q值最大的动作（若有多个最大值，随机选）
            q_values=self.q_table[state[0],state[1],:]
            max_q=np.max(q_values)
            return np.random.choice(np.where(q_values==max_q)[0])
        
    def update_q_table(self,state,action,reward,next_state,next_action):
        """更新动作价值函数"""
        current_q=self.q_table[state[0],state[1],action]
        next_q=self.q_table[next_state[0],next_state[1],next_action]
        # Sarsa更新公式：q(s,a)=q(s,a)+α[r+γ*q(s',a')-q(s,a)]
        self.q_table[state[0],state[1],action]+=self.alpha*(reward+self.gamma*next_q-current_q)
     

# 3.环境交互与训练过程
def step(state,action):
    """根据当前状态和动作，返回下一个状态和奖励"""
    x,y=state
    dx,dy=ACTIONS[action]
    next_x=x+dx
    next_y=y+dy
    
    # 判断下一个状态是否合法
    if (next_x,next_y)==TARGET_STATE:
        return (next_x,next_y),0 # 到达目标状态，奖励为0
    if (next_x,next_y) in FORBIDDEN_STATES:
        return state,FORBIDDEN_REWARD # 禁止通行，保持原状态，奖励为-10
    if 0<=next_x<GRID_SIZE and 0<=next_y<GRID_SIZE:
        return (next_x,next_y),OTHER_REWARD # 合法移动，奖励为-1
    else:
        return(x,y),BOUNDARY_REWARD # 越界，保持原状态，奖励为-10
    
# 3.训练过程
def train_sarsa(episodes=500):
    agent=SarsaAgent(alpha=0.1,epsilon=0.1)
    total_rewards=[]
    episode_lengths=[]
    
    for episode in range(episodes):
        state=START_STATE
        action=agent.choose_action(state)
        total_reward=0
        length=0
        
        while state!=TARGET_STATE:
            #执行动作，得到下一个状态、奖励
            next_state,reward=step(state,action)
            # 选择下一个动作
            next_action=agent.choose_action(next_state)
            # 更新q表
            agent.update_q_table(state,action,reward,next_state,next_action)
            
            # 累计信息
            total_reward+=reward
            length+=1
            
            # 转移到下一个状态、动作
            state=next_state
            action=next_action
        
        total_rewards.append(total_reward)
        episode_lengths.append(length)
        # 每100轮打印进度
        if (episode+1)%100==0:
            print(f"Episode {episode+1}, Avg Reward: {np.mean(total_rewards[-100:]):.2f}")
        
    return agent,total_rewards,episode_lengths

#5.运行训练并可视化结果
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
                # 选q值最大的动作
                best_action = np.argmax(agent.q_table[x, y, :])
                row.append(str(best_action))
        print(" ".join(row))
        