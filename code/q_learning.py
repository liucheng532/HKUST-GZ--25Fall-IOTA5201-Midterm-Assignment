"""
Q-Learning算法实现
"""
import numpy as np
import random


class QLearning:
    """
    Q-Learning算法类
    """
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01):
        """
        初始化Q-Learning算法
        
        Args:
            env: 环境对象
            learning_rate: 学习率（alpha）
            discount_factor: 折扣因子（gamma）
            epsilon: 初始探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 初始化Q表：n_states x n_actions
        self.n_states = env.get_state_count()
        self.n_actions = env.get_action_count()
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        # 记录训练历史
        self.episode_rewards = []
        self.episode_q_values = []  # 记录每集结束时的Q表最大值
        self.episode_lengths = []
        
    def choose_action(self, state):
        """
        使用epsilon-greedy策略选择动作（参考frozenlake_q_learning.py）
        
        Args:
            state: 当前状态
        
        Returns:
            action: 选择的动作
        """
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.randint(0, self.n_actions - 1)
        else:
            # 利用：选择Q值最大的动作
            # 如果有多个动作具有相同的最大Q值，随机选择其中一个（与参考代码保持一致）
            max_q = np.max(self.q_table[state])
            max_indices = np.where(self.q_table[state] == max_q)[0]
            return random.choice(max_indices)
    
    def update_q_table(self, state, action, reward, next_state, done):
        """
        更新Q表（参考frozenlake_q_learning.py的实现）
        
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        
        注意：即使done=True，我们也计算max(Q(s', a'))，因为next_state仍然是有效的状态
        参考代码总是计算gamma * max(Q(s', a'))，这在到达终点时也能学习到正确的Q值
        """
        current_q = self.q_table[state, action]
        
        # 参考frozenlake代码：总是计算max_next_q，即使done=True
        # 这样可以确保在到达终点时也能正确更新Q值
        max_next_q = np.max(self.q_table[next_state])
        
        # 计算目标Q值
        target_q = reward + self.discount_factor * max_next_q
        
        # 计算delta并更新（与参考代码保持一致）
        delta = target_q - current_q
        self.q_table[state, action] = current_q + self.learning_rate * delta
    
    def train_episode(self, max_steps=1000):
        """
        训练一个episode
        
        Args:
            max_steps: 最大步数限制，防止无限循环
        
        Returns:
            total_reward: 该episode的总奖励
            steps: 该episode的步数
        """
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        while steps < max_steps:
            # 选择动作
            action = self.choose_action(state)
            
            # 执行动作（gymnasium返回terminated和truncated）
            result = self.env.step(action)
            if len(result) == 4:
                # 兼容旧接口
                next_state, reward, done, info = result
                terminated = done
                truncated = False
            else:
                # gymnasium新接口：返回(next_state, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            
            # 更新Q表（参考代码：总是传入next_state，即使done=True）
            self.update_q_table(state, action, reward, next_state, done)
            
            # 更新状态和统计
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        return total_reward, steps
    
    def train(self, num_episodes=1000, verbose=True, print_interval=100):
        """
        训练Q-Learning算法
        
        Args:
            num_episodes: 训练episode数量
            verbose: 是否打印训练信息
            print_interval: 打印间隔
        """
        self.episode_rewards = []
        self.episode_q_values = []
        self.episode_lengths = []
        
        # 记录成功到达终点的次数
        success_count = 0
        
        for episode in range(num_episodes):
            # 训练一个episode
            total_reward, steps = self.train_episode()
            
            # 记录统计信息
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # 统计成功次数
            if total_reward > 0:
                success_count += 1
            
            # 记录当前Q表的最大值（用于观察收敛）
            max_q_value = np.max(self.q_table)
            self.episode_q_values.append(max_q_value)
            
            # 衰减探索率
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # 打印进度
            if verbose and (episode + 1) % print_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-print_interval:])
                avg_steps = np.mean(self.episode_lengths[-print_interval:])
                success_rate = success_count / (episode + 1) * 100
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Avg Reward: {avg_reward:.3f}, "
                      f"Avg Steps: {avg_steps:.2f}, "
                      f"Success Rate: {success_rate:.1f}%, "
                      f"Epsilon: {self.epsilon:.3f}, "
                      f"Max Q: {max_q_value:.3f}")
        
        # 打印最终统计
        if verbose:
            final_success_rate = success_count / num_episodes * 100
            print(f"\n训练完成！总成功率: {final_success_rate:.1f}% ({success_count}/{num_episodes})")
    
    def get_optimal_policy(self):
        """
        获取最优策略
        
        Returns:
            policy: 策略矩阵，shape为(n_states,)，值为每个状态的最优动作
        """
        return np.argmax(self.q_table, axis=1)
    
    def get_optimal_path(self, start_state=None):
        """
        根据Q表获取从起点到终点的最优路径
        
        Args:
            start_state: 起始状态，如果为None则使用环境的起点
        
        Returns:
            path: 路径列表，包含状态序列
        """
        if start_state is None:
            start_state = self.env.reset()
        else:
            self.env.reset()
        
        path = [start_state]
        policy = self.get_optimal_policy()
        current_state = start_state
        visited = set([start_state])  # 跟踪已访问的状态，避免循环
        
        max_steps = self.n_states * 2  # 防止无限循环
        steps = 0
        
        while steps < max_steps:
            action = policy[current_state]
            result = self.env.step(action)
            
            if len(result) == 4:
                next_state, reward, done, info = result
                terminated = done
                truncated = False
            else:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            
            if done:
                path.append(next_state)
                break
            
            # 检测循环：如果下一个状态已经在路径中，停止
            if next_state in visited:
                break
            
            visited.add(next_state)
            path.append(next_state)
            current_state = next_state
            steps += 1
        
        return path

