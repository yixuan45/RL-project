import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class GridWorldEnv:
    """
    网格世界环境类：模拟5x5网格环境，包含禁止区域、目标区域和动作交互逻辑
    核心功能：状态与坐标转换、动作执行与奖励计算、环境可视化渲染
    """

    def __init__(self):
        # 1. 环境基础参数配置
        self.grid_size = 5  # 网格尺寸：5行5列
        self.state_space = self.grid_size * self.grid_size  # 状态空间大小：25个状态（索引0-24）
        self.action_space = 5  # 动作空间：5个离散动作（上/右/下/左/停留）

        # 2. 特殊区域定义（与目标网格完全对齐）
        # 禁止区域（橙色）：对应网格坐标(行,列)→状态索引映射
        self.forbidden_states = [7, 8, 12, 17, 22, 13, 18]  # 分别对应：
        # (1,2),(1,3),(2,2),(3,2),(4,2),(3,3),(3,4)（注意：行/列从0开始计数）
        self.target_state = 14  # 目标区域（蓝色）：对应网格(3,4)（状态索引14）
        # 奖励配置：引导智能体避开危险、趋向目标
        self.boundary_penalty = -1  # 碰撞边界惩罚
        self.forbidden_penalty = -1  # 进入禁止区域惩罚
        self.target_reward = 10  # 到达目标奖励（设置较高值增强吸引力）
        self.other_reward = 0  # 普通移动无奖励

        # 3. 动作对网格坐标的影响（行变化, 列变化）
        self.action_effects = [
            (-1, 0),  # 0: 上（行减1，列不变）
            (0, 1),   # 1: 右（行不变，列加1）
            (1, 0),   # 2: 下（行加1，列不变）
            (0, -1),  # 3: 左（行不变，列减1）
            (0, 0)    # 4: 停留（行、列均不变）
        ]

    def state_to_pos(self, state):
        """
        状态索引转换为网格坐标（行, 列）
        转换规则：行 = 状态索引 // 网格尺寸，列 = 状态索引 % 网格尺寸
        :param state: 状态索引（0-24）
        :return: (row, col) 网格坐标（行、列均从0开始）
        """
        row = state // self.grid_size
        col = state % self.grid_size
        return (row, col)

    def pos_to_state(self, pos):
        """
        网格坐标转换为状态索引
        转换规则：状态索引 = 行 * 网格尺寸 + 列
        :param pos: (row, col) 网格坐标（行、列均从0开始）
        :return: state 状态索引（0-24）
        """
        row, col = pos
        return row * self.grid_size + col

    def step(self, state, action):
        """
        执行动作并返回环境反馈（状态转移+即时奖励）
        核心逻辑：根据动作计算新坐标→判断是否越界/进入特殊区域→返回下一个状态和奖励
        :param state: 当前状态索引（0-24）
        :param action: 选择的动作（0-4，对应上/右/下/左/停留）
        :return: next_state（下一个状态索引）、reward（即时奖励）
        """
        # 当前状态转换为网格坐标
        row, col = self.state_to_pos(state)
        # 计算动作执行后的新坐标
        delta_row, delta_col = self.action_effects[action]
        new_row = row + delta_row
        new_col = col + delta_col

        # 4. 状态转移与奖励判定（按优先级顺序）
        # 情况1：边界碰撞（新坐标超出网格范围）
        if new_row < 0 or new_row >= self.grid_size or new_col < 0 or new_col >= self.grid_size:
            next_state = state  # 反弹回原状态
            reward = self.boundary_penalty
        # 情况2：在网格内，进一步判断是否进入特殊区域
        else:
            next_state_candidate = self.pos_to_state((new_row, new_col))
            # 子情况2.1：进入禁止区域
            if next_state_candidate in self.forbidden_states:
                next_state = next_state_candidate
                reward = self.forbidden_penalty
            # 子情况2.2：到达目标区域
            elif next_state_candidate == self.target_state:
                next_state = next_state_candidate
                reward = self.target_reward
            # 子情况2.3：普通可行区域（白色）
            else:
                next_state = next_state_candidate
                reward = self.other_reward

        return next_state, reward

    def render(self, policy=None, state_values=None, iteration=None, is_final=False):
        """
        环境可视化渲染：绘制网格、特殊区域、状态值、策略箭头
        渲染触发条件：每100次迭代 或 策略最终收敛
        :param policy: 策略矩阵（25x5，每行是对应状态的动作概率分布）
        :param state_values: 状态值数组（25个元素，对应每个状态的价值）
        :param iteration: 当前迭代次数（用于标题标注）
        :param is_final: 是否为最终收敛结果（用于区分标题）
        """
        # 非100次迭代且非最终收敛，不渲染
        if not (iteration % 100 == 0 or is_final):
            return

        # 初始化网格矩阵（0=可行区，-1=禁止区，1=目标区）
        grid = np.zeros((self.grid_size, self.grid_size))
        # 标记禁止区域（橙色）
        for s in self.forbidden_states:
            row, col = self.state_to_pos(s)
            grid[row, col] = -1
        # 标记目标区域（蓝色）
        t_row, t_col = self.state_to_pos(self.target_state)
        grid[t_row, t_col] = 1

        # 颜色映射配置：白色（可行区）、浅蓝色（目标区）、浅红色（禁止区）
        cmap = ListedColormap(['white', 'lightblue', 'lightcoral'])
        # 创建画布并绘制网格
        plt.figure(figsize=(10, 8))
        plt.imshow(grid, cmap=cmap, vmin=-1, vmax=1)  # vmin/vmax固定颜色映射范围

        # 标注状态值（保留1位小数，与网格示例格式一致）
        if state_values is not None:
            for state in range(self.state_space):
                row, col = self.state_to_pos(state)
                value_text = f"{state_values[state]:.1f}"
                # 目标区域的状态值用红色突出显示
                color = 'red' if state == self.target_state else 'black'
                # 在网格中心添加状态值文本
                plt.text(col, row, value_text, ha='center', va='center', fontsize=10, color=color)

        # 绘制策略箭头（仅可行区域显示，禁止区和目标区不画）
        if policy is not None:
            # 动作→箭头映射（与action_effects顺序一致）
            arrow_dirs = ['↑', '→', '↓', '←', '○']
            for state in range(self.state_space):
                row, col = self.state_to_pos(state)
                # 选择当前状态下概率最大的动作（贪婪策略可视化）
                best_action = np.argmax(policy[state])
                # 仅在可行区域（非禁止/非目标）绘制箭头
                if state not in self.forbidden_states and state != self.target_state:
                    plt.text(col, row, arrow_dirs[best_action], ha='center', va='center', fontsize=12, color='blue')

        # 标题区分：最终收敛结果 vs 100次迭代中间结果
        if is_final:
            plt.title(f'Grid World - Final Converged Result (Iteration {iteration})', fontsize=14)
        else:
            plt.title(f'Grid World - Iteration {iteration} - Policy & State Values', fontsize=14)
        
        # 坐标轴标注（列/行从1开始，符合日常网格认知）
        plt.xticks(range(self.grid_size), [f'Col {i+1}' for i in range(self.grid_size)])
        plt.yticks(range(self.grid_size), [f'Row {i+1}' for i in range(self.grid_size)])
        # 添加网格线（虚线，透明度0.5，便于区分单元格）
        plt.grid(True, linestyle='--', alpha=0.5)
        # 显示图像
        plt.show()


def policy_evaluation(env, policy, gamma=0.9, theta=1e-6):
    """
    策略评估：计算给定策略下的状态值函数V(s)
    核心算法：迭代法求解贝尔曼方程，直到状态值变化小于收敛阈值theta
    :param env: 网格环境对象（GridWorldEnv实例）
    :param policy: 待评估的策略（25x5矩阵，policy[s][a]表示状态s选择动作a的概率）
    :param gamma: 折扣因子（0-1，控制未来奖励的权重，默认0.9）
    :param theta: 收敛阈值（状态值最大变化小于theta时停止迭代，默认1e-6）
    :return: v 状态值数组（25个元素，V(s)表示状态s的长期累积奖励）
    """
    # 初始化状态值：所有状态初始价值为0
    v = np.zeros(env.state_space)
    while True:
        delta = 0  # 记录本轮迭代中状态值的最大变化（用于收敛判断）
        # 遍历所有状态，更新状态值
        for state in range(env.state_space):
            v_old = v[state]  # 保存更新前的旧状态值
            v_new = 0  # 初始化新状态值（按贝尔曼方程计算）

            # 贝尔曼方程：V(s) = Σπ(a|s) * [r(s,a) + γ*V(s')]
            # 遍历所有可能动作，累加每个动作的贡献
            for action in range(env.action_space):
                action_prob = policy[state][action]  # 策略π(a|s)：状态s选择动作a的概率
                next_state, reward = env.step(state, action)  # 执行动作a，得到s'和r
                # 累加动作a对状态s价值的贡献
                v_new += action_prob * (reward + gamma * v[next_state])

            # 更新当前状态的价值
            v[state] = v_new
            # 更新本轮最大状态值变化
            delta = max(delta, abs(v_old - v_new))

        # 收敛判断：若最大变化小于theta，说明状态值已稳定，停止迭代
        if delta < theta:
            break
    return v


def policy_improvement(env, v, gamma=0.9, old_policy=None):
    """
    策略改进：基于当前状态值函数V(s)，生成更优的贪婪策略
    核心逻辑：对每个状态，选择使Q(s,a)最大的动作（贪婪选择），更新策略
    :param env: 网格环境对象（GridWorldEnv实例）
    :param v: 状态值数组（由策略评估得到）
    :param gamma: 折扣因子（与策略评估一致，默认0.9）
    :param old_policy: 上一轮的策略（用于判断策略是否稳定）
    :return: new_policy（改进后的新策略）、policy_stable（策略是否稳定的标记）
    """
    # 初始化新策略：25x5全0矩阵（确定性策略，仅最优动作概率为1）
    new_policy = np.zeros((env.state_space, env.action_space))
    policy_stable = True  # 策略稳定性标记（初始假设稳定，若有状态动作变化则置为False）

    # 遍历所有状态，优化每个状态的动作选择
    for state in range(env.state_space):
        # 1. 计算当前状态下每个动作的动作值Q(s,a)
        q_values = np.zeros(env.action_space)  # Q(s,a)数组（5个元素，对应5个动作）
        for action in range(env.action_space):
            next_state, reward = env.step(state, action)  # 执行动作a得到s'和r
            # Q值公式：Q(s,a) = r(s,a) + γ*V(s')（基于当前状态值V(s)计算）
            q_values[action] = reward + gamma * v[next_state]

        # 2. 贪婪选择：选择Q值最大的动作（最优动作）
        best_action = np.argmax(q_values)
        # 3. 对比上一轮策略的最优动作（判断策略是否变化）
        # 若无上一轮策略（首次迭代），则旧最优动作设为-1（无效值）
        old_best_action = np.argmax(old_policy[state]) if old_policy is not None else -1

        # 4. 更新新策略：最优动作概率设为1，其他动作设为0（确定性策略）
        new_policy[state] = 0
        new_policy[state][best_action] = 1

        # 5. 判定策略稳定性：若当前状态的最优动作与上一轮不同，说明策略未稳定
        if best_action != old_best_action:
            policy_stable = False

    return new_policy, policy_stable


def policy_iteration(env, gamma=0.9, theta=1e-6, max_iter=1000):
    """
    策略迭代算法：交替执行「策略评估」和「策略改进」，直到策略稳定（找到最优策略）
    核心流程：初始化策略→评估→改进→判断稳定？→是（输出最优）/否（重复评估改进）
    :param env: 网格环境对象（GridWorldEnv实例）
    :param gamma: 折扣因子（默认0.9）
    :param theta: 策略评估的收敛阈值（默认1e-6）
    :param max_iter: 最大迭代次数（防止无限循环，默认1000）
    :return: optimal_policy（最优策略）、optimal_v（最优状态值数组）
    """
    # 初始化策略：均匀随机策略（每个状态的5个动作概率均为0.2，保证初始探索性）
    policy = np.ones((env.state_space, env.action_space)) / env.action_space

    print("开始策略迭代...")
    iteration = 0  # 迭代计数器
    # 迭代循环：直到策略稳定或达到最大迭代次数
    while iteration < max_iter:
        iteration += 1
        # 步骤1：策略评估：计算当前策略下的状态值V(s)
        v = policy_evaluation(env, policy, gamma, theta)
        # 步骤2：策略改进：基于状态值V(s)生成新策略，并判断稳定性
        new_policy, policy_stable = policy_improvement(env, v, gamma, old_policy=policy)

        # 输出当前迭代信息（迭代次数+策略稳定性）
        print(f"迭代次数: {iteration} | 策略是否稳定: {policy_stable}")

        # 步骤3：可视化渲染（每100次迭代或最终收敛时触发）
        env.render(policy=new_policy, state_values=v, iteration=iteration, is_final=False)
        # 步骤4：终止条件判断：若策略稳定，说明找到最优策略，结束迭代
        if policy_stable:
            print(f"策略迭代完成，在第{iteration}次迭代收敛！")
            # 渲染最终收敛结果
            env.render(policy=new_policy, state_values=v, iteration=iteration, is_final=True)
            break

        # 更新策略：将新策略作为下一轮评估的输入
        policy = new_policy

    # 异常处理：达到最大迭代次数仍未收敛
    if iteration >= max_iter:
        print(f"已达到最大迭代次数{max_iter}，未完全收敛")
        env.render(policy=policy, state_values=v, iteration=iteration, is_final=True)

    return policy, v


# ------------------- 主程序：执行策略迭代，运行网格世界环境 -------------------
if __name__ == "__main__":
    # 1. 创建网格环境实例（初始化网格、特殊区域、奖励规则等）
    env = GridWorldEnv()

    # 2. 运行策略迭代算法，求解最优策略和最优状态值
    optimal_policy, optimal_v = policy_iteration(
        env,
        gamma=0.9,    # 折扣因子：重视未来奖励
        theta=1e-6,   # 收敛阈值：保证状态值精度
        max_iter=1000 # 最大迭代次数：防止死循环
    )

    # 3. 输出最终结果（数值形式，便于验证）
    print("\n=== 最优状态值（5x5网格排列，行优先）===")
    # 将一维状态值数组reshape为5x5网格，保留1位小数
    print(optimal_v.reshape(env.grid_size, env.grid_size).round(1))
    print("\n=== 最优策略（5x5网格排列，行优先）===")
    # 提取每个状态的最优动作（0=上,1=右,2=下,3=左,4=停留）
    optimal_policy_actions = np.argmax(optimal_policy, axis=1)
    print(optimal_policy_actions.reshape(env.grid_size, env.grid_size))