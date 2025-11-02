"""
使用Gymnasium的FrozenLake环境
9x7的grid环境，包含frozen grids（冰块区域）
"""
import numpy as np
import gymnasium as gym


class GridEnv:
    """
    基于Gymnasium FrozenLake环境的网格环境
    - 大小为9列×7行（width=9, height=7）
    - 包含frozen grids（Holes，agent进入会结束episode）
    - 包含起点和终点
    """
    
    # 定义动作：FrozenLake使用的动作编码
    # 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    
    ACTION_NAMES = ['LEFT', 'DOWN', 'RIGHT', 'UP']
    
    @staticmethod
    def user_to_internal(user_x, user_y, height):
        """
        将用户坐标（1-based，左下角为原点）转换为内部坐标（0-based，左上角为原点）
        
        用户坐标系：
        - 左下角为(1, 1)
        - 右上角为(9, 7) (假设9列7行)
        - x: 列索引，从1到width
        - y: 行索引，从1到height，y=1是最下面，y=height是最上面
        
        内部坐标系（FrozenLake）：
        - 左上角为(0, 0)
        - x: 列索引，从0到width-1
        - y: 行索引，从0到height-1，y=0是最上面，y=height-1是最下面
        
        Args:
            user_x: 用户x坐标（列，1-based）
            user_y: 用户y坐标（行，1-based，1是最下面）
            height: 环境高度
            
        Returns:
            (internal_x, internal_y): 内部坐标（0-based）
        """
        internal_x = user_x - 1  # 列：从1-based转换为0-based
        internal_y = height - user_y  # 行：翻转，因为用户y=1是下面，内部y=0是上面
        return (internal_x, internal_y)
    
    @staticmethod
    def internal_to_user(internal_x, internal_y, height):
        """
        将内部坐标（0-based）转换为用户坐标（1-based）
        
        Args:
            internal_x: 内部x坐标（列，0-based）
            internal_y: 内部y坐标（行，0-based）
            height: 环境高度
            
        Returns:
            (user_x, user_y): 用户坐标（1-based）
        """
        user_x = internal_x + 1
        user_y = height - internal_y
        return (user_x, user_y)
    
    def __init__(self, width=9, height=7, frozen_cells=None, 
                 start_pos=(6, 1), goal_pos=(6, 7), 
                 is_slippery=False, seed=None, render_mode="human"):
        """
        初始化环境
        
        Args:
            width: 网格宽度（列数）
            height: 网格高度（行数）
            frozen_cells: Frozen grids坐标列表，格式为[(x, y), ...]
                        用户坐标系统：左下角为(1,1)，右上角为(width, height)
                        x是列索引（1到width），y是行索引（1到height，1是最下面）
            start_pos: 起点坐标 (x, y)，用户坐标系统（1-based）
            goal_pos: 终点坐标 (x, y)，用户坐标系统（1-based）
            is_slippery: 是否滑溜（FrozenLake参数）
            seed: 随机种子
            render_mode: 渲染模式
        """
        self.width = width
        self.height = height
        self.seed = seed
        
        # 创建地图描述（desc）
        # FrozenLake使用字符串列表，每个字符代表一个格子
        # 'S' = Start, 'G' = Goal, 'F' = Frozen (safe), 'H' = Hole (dangerous)
        desc = []
        for row in range(height):
            desc_row = []
            for col in range(width):
                desc_row.append('F')  # 默认是可通行的冰面
            desc.append(''.join(desc_row))
        
        # 设置frozen cells（危险区域）
        if frozen_cells is None:
            # 默认frozen cells坐标（用户坐标系统：1-based，左下角为(1,1)）
            frozen_cells = [(2, 4), (3, 2), (3, 4), (4, 4), (5, 2), (5, 4),
                           (6, 2), (6, 4), (7, 1), (7, 2), (7, 4), (7, 6),
                           (7, 7), (8, 6)]
        
        # 设置frozen cells为Holes
        for cell in frozen_cells:
            user_x, user_y = cell  # 用户坐标（1-based）
            # 转换为内部坐标
            internal_x, internal_y = self.user_to_internal(user_x, user_y, height)
            
            # 验证坐标范围
            if 1 <= user_x <= width and 1 <= user_y <= height:
                if 0 <= internal_y < height and 0 <= internal_x < width:
                    # 将对应位置设置为'H'（Hole）
                    row_str = list(desc[internal_y])
                    row_str[internal_x] = 'H'
                    desc[internal_y] = ''.join(row_str)
            else:
                print(f"警告: Frozen cell坐标({user_x}, {user_y})超出用户坐标范围 [x:1-{width}, y:1-{height}]")
        
        # 保存用户坐标（用于显示）
        self.start_pos_user = start_pos
        self.goal_pos_user = goal_pos
        
        # 设置起点（转换为内部坐标）
        start_x_user, start_y_user = start_pos  # 用户坐标（1-based）
        internal_start_x, internal_start_y = self.user_to_internal(start_x_user, start_y_user, height)
        
        # 验证用户坐标范围
        if not (1 <= start_x_user <= width and 1 <= start_y_user <= height):
            raise ValueError(f"起点坐标({start_x_user}, {start_y_user})超出用户坐标范围！"
                           f"有效范围: x=[1-{width}], y=[1-{height}]")
        
        if 0 <= internal_start_y < height and 0 <= internal_start_x < width:
            row_str = list(desc[internal_start_y])
            row_str[internal_start_x] = 'S'
            desc[internal_start_y] = ''.join(row_str)
            self.start_pos = (internal_start_x, internal_start_y)  # 保存内部坐标
        else:
            raise ValueError(f"起点坐标转换错误！用户坐标({start_x_user}, {start_y_user}) -> "
                           f"内部坐标({internal_start_x}, {internal_start_y})超出范围")
        
        # 设置终点（转换为内部坐标）
        goal_x_user, goal_y_user = goal_pos  # 用户坐标（1-based）
        internal_goal_x, internal_goal_y = self.user_to_internal(goal_x_user, goal_y_user, height)
        
        # 验证用户坐标范围
        if not (1 <= goal_x_user <= width and 1 <= goal_y_user <= height):
            raise ValueError(f"终点坐标({goal_x_user}, {goal_y_user})超出用户坐标范围！"
                           f"有效范围: x=[1-{width}], y=[1-{height}]")
        
        if 0 <= internal_goal_y < height and 0 <= internal_goal_x < width:
            row_str = list(desc[internal_goal_y])
            row_str[internal_goal_x] = 'G'
            desc[internal_goal_y] = ''.join(row_str)
            self.goal_pos = (internal_goal_x, internal_goal_y)  # 保存内部坐标
        else:
            raise ValueError(f"终点坐标转换错误！用户坐标({goal_x_user}, {goal_y_user}) -> "
                           f"内部坐标({internal_goal_x}, {internal_goal_y})超出范围")
        
        # 确保起点和终点不是Hole
        # （已经在上面设置过了，这里不需要额外处理）
        
        # 创建FrozenLake环境
        # 设置较大的max_episode_steps，让agent有足够时间找到路径
        self.env = gym.make(
            "FrozenLake-v1",
            desc=desc,
            is_slippery=is_slippery,
            render_mode=render_mode,  # "human" for GUI window, "rgb_array" for image array
        )
        
        # 设置更大的episode步数限制（默认可能是100，我们需要更多）
        # 通过TimeLimit wrapper来实现
        from gymnasium.wrappers import TimeLimit
        # 如果环境已经被TimeLimit包装，更新它；否则包装它
        if isinstance(self.env, TimeLimit):
            self.env._max_episode_steps = 1000  # 增加最大步数
        else:
            self.env = TimeLimit(self.env, max_episode_steps=1000)
        
        self.render_mode = render_mode
        
        if seed is not None:
            self.env.reset(seed=seed)
        
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        
        # 保存地图描述用于可视化
        self.desc = desc
        self.map_array = np.array([list(row) for row in desc])
        
    def reset(self, seed=None):
        """重置环境到初始状态"""
        if seed is None:
            seed = self.seed
        result = self.env.reset(seed=seed)
        if isinstance(result, tuple):
            state, info = result
        else:
            state = result
            info = {}
        self._current_state = state  # 跟踪当前状态
        return state
    
    def step(self, action):
        """
        执行动作
        
        Args:
            action: 动作（0=LEFT, 1=DOWN, 2=RIGHT, 3=UP）
        
        Returns:
            next_state: 下一个状态
            reward: 奖励（到达终点为1.0，否则为0.0）
            done: 是否结束
            info: 额外信息
        
        注意：如果agent尝试进入frozen cell（Hole），会被阻止，停留在原位置
        """
        # 获取当前状态
        # 如果还没有跟踪状态，尝试从环境获取
        if not hasattr(self, '_current_state'):
            # 尝试从环境获取当前状态
            try:
                if hasattr(self.env, 's'):
                    self._current_state = self.env.s
                elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 's'):
                    self._current_state = self.env.unwrapped.s
                else:
                    # 如果无法获取，使用初始状态
                    self._current_state = 0
            except:
                self._current_state = 0
        
        current_state = self._current_state
        current_pos = self._state_to_pos(current_state, user_coords=False)
        current_row, current_col = current_pos[1], current_pos[0]
        
        # 计算下一个位置（基于当前位置和动作）
        next_row, next_col = current_row, current_col
        
        if action == self.LEFT:  # 0
            next_col = max(0, current_col - 1)
        elif action == self.DOWN:  # 1
            next_row = min(self.height - 1, current_row + 1)
        elif action == self.RIGHT:  # 2
            next_col = min(self.width - 1, current_col + 1)
        elif action == self.UP:  # 3
            next_row = max(0, current_row - 1)
        
        # 检查下一个位置是否是Hole（frozen cell）
        if self.map_array[next_row, next_col] == 'H':
            # 如果是Hole，阻止移动，停留在当前位置
            # 给予小的负奖励来学习避免Holes
            next_state = current_state
            reward = -0.01  # 小的负奖励，鼓励agent避免Holes
            done = False
            terminated = False
            truncated = False
            info = {'blocked': True, 'reason': 'frozen_cell'}
            # 不更新_current_state，因为状态没有改变
        else:
            # 执行动作
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            info['blocked'] = False
            # 更新当前状态
            self._current_state = next_state
        
        return next_state, reward, done, info
    
    def render(self, mode=None):
        """
        可视化当前环境状态
        
        Args:
            mode: 渲染模式，None表示使用初始化时的模式
                 - "human": 显示GUI窗口（需要pygame）
                 - "rgb_array": 返回RGB图像数组
                 - "ansi": 返回文本形式的地图
        """
        if mode is None:
            mode = self.render_mode
        
        # 如果使用human模式，直接调用gymnasium的render
        if mode == "human":
            return self.env.render()
        elif mode == "rgb_array":
            return self.env.render()
        elif mode == "ansi" or mode == "console":
            # 文本形式输出（显示用户坐标）
            print("\nEnvironment Grid (9 columns × 7 rows):")
            print("坐标系：左下角为(1,1)，右上角为(9,7)")
            print("S=Start, G=Goal, F=Frozen (safe), H=Hole (dangerous)")
            print(f"起点（用户坐标）: {self.start_pos_user}")
            print(f"终点（用户坐标）: {self.goal_pos_user}\n")
            
            # 从上到下显示（desc[0]是上面，对应用户坐标y=7）
            for i in range(self.height):
                row_idx = self.height - 1 - i  # 翻转，从上面开始显示
                row_display = ' '.join(self.desc[row_idx])
                user_y = i + 1  # 用户y坐标（从下往上，所以第一行是y=1）
                print(f"Row y={user_y} (从下往上): {row_display}")
            print("      ", end="")
            for x in range(1, self.width + 1):
                print(f"x={x:2d}  ", end="")
            print()
            return None
        else:
            # 默认文本输出
            self.render(mode="console")
            return None
    
    def get_state_count(self):
        """返回状态总数"""
        return self.n_states
    
    def get_action_count(self):
        """返回动作总数"""
        return self.n_actions
    
    def _state_to_pos(self, state, user_coords=True):
        """
        将状态编号转换为位置坐标
        
        Args:
            state: 状态编号
            user_coords: 如果True，返回用户坐标（1-based），否则返回内部坐标（0-based）
        
        Returns:
            (x, y): 位置坐标
        """
        row = state // self.width
        col = state % self.width
        
        if user_coords:
            # 转换为用户坐标系统
            user_x, user_y = self.internal_to_user(col, row, self.height)
            return (user_x, user_y)
        else:
            # 返回内部坐标
            return (col, row)
    
    def _pos_to_state(self, pos, user_coords=True):
        """
        将位置坐标转换为状态编号
        
        Args:
            pos: 位置坐标 (x, y)
            user_coords: 如果True，pos是用户坐标（1-based），否则是内部坐标（0-based）
        
        Returns:
            state: 状态编号
        """
        x, y = pos
        if user_coords:
            # 转换为内部坐标
            internal_x, internal_y = self.user_to_internal(x, y, self.height)
            return internal_y * self.width + internal_x
        else:
            # 直接使用内部坐标
            return y * self.width + x
    
    def get_map_description(self):
        """获取地图描述"""
        return self.desc
