"""
主程序：运行Q-Learning训练并可视化结果
"""
import os
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
from environment import GridEnv
from q_learning import QLearning


def plot_convergence(q_learning, title="Q-Learning Convergence"):
    """
    绘制Q值收敛图
    
    Args:
        q_learning: QLearning对象
        title: 图表标题
    """
    plt.figure(figsize=(12, 5))
    
    # 子图1：Q值收敛
    plt.subplot(1, 2, 1)
    plt.plot(q_learning.episode_q_values, linewidth=1, alpha=0.7)
    plt.title('Q-Table Value Convergence')
    plt.xlabel('Episode')
    plt.ylabel('Max Q-Value')
    plt.grid(True, alpha=0.3)
    
    # 计算并显示移动平均
    window_size = min(100, len(q_learning.episode_q_values) // 10)
    if window_size > 1:
        moving_avg = np.convolve(q_learning.episode_q_values, 
                                 np.ones(window_size)/window_size, 
                                 mode='valid')
        plt.plot(range(window_size-1, len(q_learning.episode_q_values)), 
                moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size})')
        plt.legend()
    
    # 子图2：Episode奖励
    plt.subplot(1, 2, 2)
    plt.plot(q_learning.episode_rewards, linewidth=1, alpha=0.7, color='green')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    # 计算并显示移动平均
    if window_size > 1:
        moving_avg_reward = np.convolve(q_learning.episode_rewards, 
                                       np.ones(window_size)/window_size, 
                                       mode='valid')
        plt.plot(range(window_size-1, len(q_learning.episode_rewards)), 
                moving_avg_reward, 'r-', linewidth=2, label=f'Moving Average ({window_size})')
        plt.legend()
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=14)
    return plt.gcf()


def visualize_q_table(q_learning, title="Q-Table Visualization"):
    """
    可视化Q表
    
    Args:
        q_learning: QLearning对象
        title: 图表标题
    """
    env = q_learning.env
    q_table = q_learning.q_table
    
    # 为每个动作创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']  # FrozenLake动作顺序
    
    for action in range(4):
        ax = axes[action // 2, action % 2]
        
        # 将Q值重塑为网格形状
        q_grid = q_table[:, action].reshape(env.height, env.width)
        
        # 绘制热力图
        im = ax.imshow(q_grid, cmap='viridis', aspect='auto')
        
        # 标记Holes（危险区域）
        for i in range(env.height):
            for j in range(env.width):
                if env.map_array[i, j] == 'H':
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                              fill=True, color='red', 
                                              alpha=0.5))
        
        # 标记起点和终点（使用用户坐标）
        start_x, start_y = env.start_pos_user
        goal_x, goal_y = env.goal_pos_user
        # 转换为绘图坐标（用户y需要翻转显示）
        start_y_plot = env.height - start_y + 1
        goal_y_plot = env.height - goal_y + 1
        ax.plot(start_x - 1, start_y_plot - 1, 's', 
               markersize=15, color='blue', label='Start')
        ax.plot(goal_x - 1, goal_y_plot - 1, '*', 
               markersize=20, color='yellow', label='Goal')
        
        # 添加文本标注
        for i in range(env.height):
            for j in range(env.width):
                text = ax.text(j, i, f'{q_grid[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=8)
        
        ax.set_title(f'Q-Values for Action: {action_names[action]}')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_xticks(range(env.width))
        ax.set_yticks(range(env.height))
        ax.legend()
        
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=14)
    return fig


def visualize_path(env, q_learning, title="Optimal Path"):
    """
    可视化最优路径
    
    Args:
        env: 环境对象
        q_learning: QLearning对象
        title: 图表标题
    """
    path = q_learning.get_optimal_path()
    path_positions = [env._state_to_pos(state) for state in path]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 绘制网格
    for i in range(env.height):
        for j in range(env.width):
            if env.map_array[i, j] == 'H':
                # Hole（危险区域）
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                    fill=True, color='red', alpha=0.5)
                ax.add_patch(rect)
            else:
                # 其他区域
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                    fill=False, color='gray', linewidth=0.5)
                ax.add_patch(rect)
    
    # 绘制路径（path_positions返回用户坐标(x, y)）
    if len(path_positions) > 1:
        # 转换为绘图坐标
        path_x = [pos[0] - 1 for pos in path_positions]  # 转换为0-based
        path_y = [env.height - pos[1] for pos in path_positions]  # 翻转y轴
        ax.plot(path_x, path_y, 'b-o', linewidth=2, markersize=8, 
               label='Optimal Path', alpha=0.7)
    
    # 标记起点和终点（使用用户坐标）
    start_x, start_y = env.start_pos_user
    goal_x, goal_y = env.goal_pos_user
    # 转换为绘图坐标（用户y需要翻转显示）
    start_y_plot = env.height - start_y + 1
    goal_y_plot = env.height - goal_y + 1
    start_x_plot = start_x - 1  # 转换为0-based绘图坐标
    goal_x_plot = goal_x - 1
    ax.plot(start_x_plot, start_y_plot - 1, 's', 
           markersize=20, color='green', label='Start', 
           markeredgecolor='black', markeredgewidth=2)
    ax.plot(goal_x_plot, goal_y_plot - 1, '*', 
           markersize=25, color='yellow', label='Goal',
           markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)  # 反转y轴
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(title)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    return fig


def print_q_table(q_learning):
    """打印Q表"""
    print("\n" + "="*80)
    print("Q-Table (formatted by state and action)")
    print("="*80)
    env = q_learning.env
    q_table = q_learning.q_table
    
    action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']  # FrozenLake动作顺序
    
    print(f"\nState format: (x, y) -> State ID, where x is column, y is row")
    print(f"Actions: {action_names}\n")
    
    for state in range(q_learning.n_states):
        x_user, y_user = env._state_to_pos(state, user_coords=True)  # 用户坐标
        x_internal, y_internal = env._state_to_pos(state, user_coords=False)  # 内部坐标
        q_values = q_table[state]
        best_action = np.argmax(q_values)
        
        # 跳过Holes（如果有的话）
        if env.map_array[y_internal, x_internal] == 'H':
            continue
        
        print(f"State {state:2d} (用户坐标: x={x_user}, y={y_user}): ", end="")
        for action in range(4):
            print(f"{action_names[action]:5s}={q_values[action]:7.4f}  ", end="")
        print(f"-> Best: {action_names[best_action]}")
    
    print("="*80 + "\n")


def setup_result_directory():
    """
    创建结果保存目录
    格式：result/年月日时分/
    例如：result/2024-1215-1430/
    
    Returns:
        result_dir: 结果目录路径
    """
    # 获取当前脚本所在目录的父目录（项目根目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # 从code目录到Midterm目录
    
    # 创建result文件夹（如果不存在）
    result_base = os.path.join(project_root, 'result')
    os.makedirs(result_base, exist_ok=True)
    
    # 创建日期文件夹（格式：年月日时分）
    now = datetime.now()
    date_str = now.strftime("%Y-%m%d-%H%M")  # 例如：202412151430
    result_dir = os.path.join(result_base, date_str)
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"\n结果将保存到: {result_dir}")
    print("="*80)
    
    return result_dir


def get_random_start_goal_positions(width, height, frozen_cells, seed=None):
    """
    随机生成起点和终点位置（用户坐标系统）
    
    Args:
        width: 网格宽度
        height: 网格高度
        frozen_cells: Holes列表（用户坐标系统）
        seed: 随机种子
    
    Returns:
        start_pos: 起点坐标 (x, y) 用户坐标
        goal_pos: 终点坐标 (x, y) 用户坐标
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 将所有frozen cells转换为内部坐标（用于检查）
    holes_set = set(frozen_cells) if frozen_cells else set()
    
    # 生成所有安全位置（不在Holes上的位置）
    safe_positions = []
    for x in range(1, width + 1):
        for y in range(1, height + 1):
            if (x, y) not in holes_set:
                safe_positions.append((x, y))
    
    if len(safe_positions) < 2:
        raise ValueError("没有足够的安全位置来设置起点和终点！至少需要2个安全位置。")
    
    # 随机选择起点和终点
    start_pos, goal_pos = random.sample(safe_positions, 2)
    
    return start_pos, goal_pos


def save_training_log(result_dir, env, q_learning, config_name="initial"):
    """
    保存训练日志和Q表数据
    
    Args:
        result_dir: 结果目录
        env: 环境对象
        q_learning: QLearning对象
        config_name: 配置名称（initial或changed）
    """
    log_file = os.path.join(result_dir, f'training_log_{config_name}.txt')
    qtable_file = os.path.join(result_dir, f'qtable_{config_name}.npy')
    
    # 保存Q表数据
    np.save(qtable_file, q_learning.q_table)
    
    # 保存训练日志
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Q-Learning Training Log - {config_name.upper()} Configuration\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Environment Configuration:\n")
        f.write(f"  Grid Size: {env.width} columns × {env.height} rows\n")
        f.write(f"  Start Position (user coords): {env.start_pos_user}\n")
        f.write(f"  Goal Position (user coords): {env.goal_pos_user}\n")
        f.write(f"  Number of States: {env.n_states}\n")
        f.write(f"  Number of Actions: {env.n_actions}\n\n")
        
        f.write(f"Training Configuration:\n")
        f.write(f"  Learning Rate: {q_learning.learning_rate}\n")
        f.write(f"  Discount Factor: {q_learning.discount_factor}\n")
        f.write(f"  Final Epsilon: {q_learning.epsilon:.4f}\n")
        f.write(f"  Epsilon Decay: {q_learning.epsilon_decay}\n")
        f.write(f"  Epsilon Min: {q_learning.epsilon_min}\n")
        f.write(f"  Number of Episodes: {len(q_learning.episode_rewards)}\n\n")
        
        f.write(f"Training Results:\n")
        if len(q_learning.episode_rewards) > 0:
            total_success = sum(1 for r in q_learning.episode_rewards if r > 0)
            success_rate = total_success / len(q_learning.episode_rewards) * 100
            f.write(f"  Success Rate: {success_rate:.2f}% ({total_success}/{len(q_learning.episode_rewards)})\n")
            f.write(f"  Average Reward: {np.mean(q_learning.episode_rewards):.4f}\n")
            f.write(f"  Average Steps: {np.mean(q_learning.episode_lengths):.2f}\n")
            f.write(f"  Max Q-Value: {np.max(q_learning.q_table):.4f}\n")
            f.write(f"  Min Q-Value: {np.min(q_learning.q_table):.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Q-Table Summary:\n")
        f.write("="*80 + "\n")
        f.write(f"Q-table shape: {q_learning.q_table.shape}\n")
        f.write(f"Non-zero Q-values: {np.count_nonzero(q_learning.q_table)}\n")
        f.write(f"Zero Q-values: {np.sum(q_learning.q_table == 0)}\n")
    
    print(f"Saved training log: {log_file}")
    print(f"Saved Q-table data: {qtable_file}")


def main():
    """主函数"""
    # 创建结果目录
    result_dir = setup_result_directory()
    
    print("="*80)
    print("Q-Learning Assignment - Grid Environment")
    print("="*80)
    
    # ========== 任务1-4: 初始配置 ==========
    print("\n" + "="*80)
    print("Task 1-4: Initial Configuration")
    print("Start: (6, 1), Goal: (6, 7)")
    print("="*80)
    
    # 创建环境
    # 注意：坐标格式为(x, y)，其中x是列，y是行
    # 环境是9列7行（width=9, height=7）
    frozen_cells = [(2, 4), (3, 2), (3, 4), (4, 4), (5, 2), (5, 4),
                    (6, 2), (6, 4), (7, 1), (7, 2), (7, 4), (7, 6),
                    (7, 7), (8, 6)]
    
    env1 = GridEnv(width=9, height=7, 
                   frozen_cells=frozen_cells,
                   start_pos=(6, 1),  # 用户坐标系统：左下角为(1,1)
                   goal_pos=(6, 7),   # 用户坐标系统：右上角为(9,7)
                   is_slippery=False,
                   seed=123,
                   render_mode="ansi")
    
    print("\nEnvironment created:")
    env1.render()
    
    # 创建Q-Learning算法
    # 参考frozenlake_q_learning.py的超参数设置
    q_learning1 = QLearning(env1, 
                            learning_rate=0.8,  # 参考代码使用0.8
                            discount_factor=0.95,  # 参考代码使用0.95
                            epsilon=0.1,  # 参考代码使用0.1
                            epsilon_decay=0.995,  # 逐步衰减探索率
                            epsilon_min=0.01)  # 保持最小探索率
    
    # 训练
    print("\nStarting Q-Learning training...")
    print("-"*80)
    q_learning1.train(num_episodes=5000, verbose=True, print_interval=500)
    
    # 打印Q表
    print_q_table(q_learning1)
    
    # 保存训练日志和Q表数据
    print("\nSaving training data...")
    save_training_log(result_dir, env1, q_learning1, config_name="initial")
    plt.close('all')  # 关闭所有图表，释放内存
    
    # 可视化
    print("\nGenerating convergence plot...")
    fig1 = plot_convergence(q_learning1, 
                         title="Q-Learning Convergence - Initial Configuration")
    convergence_path = os.path.join(result_dir, 'convergence_initial.png')
    plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {convergence_path}")
    plt.close(fig1)
    
    print("\nGenerating Q-table visualization...")
    fig2 = visualize_q_table(q_learning1, 
                             title="Q-Table Visualization - Initial Configuration")
    qtable_path = os.path.join(result_dir, 'qtable_initial.png')
    plt.savefig(qtable_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {qtable_path}")
    plt.close(fig2)
    
    print("\nGenerating optimal path visualization...")
    fig3 = visualize_path(env1, q_learning1, 
                         title="Optimal Path - Initial Configuration")
    path_path = os.path.join(result_dir, 'path_initial.png')
    plt.savefig(path_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path_path}")
    plt.close(fig3)
    
    # ========== 任务5: 改变初始和目标状态 ==========
    print("\n" + "="*80)
    print("Task 5: Changed Configuration")
    print("Change Initial and Target States (Randomized)")
    print("="*80)
    
    # 随机生成起点和终点（确保不在Holes上，且不在同一位置）
    # 使用基于时间戳的seed，确保每次运行生成不同的随机位置
    import time
    random_seed = int(time.time()) % 10000  # 使用时间戳作为seed的一部分
    print("\n随机生成新的起点和终点位置（位置A和位置C）...")
    start_pos_A, goal_pos_C = get_random_start_goal_positions(
        width=9, 
        height=7, 
        frozen_cells=frozen_cells,
        seed=random_seed  # 每次运行使用不同的seed，生成不同的随机位置
    )
    
    print(f"\n✓ 随机生成的起点位置 A: {start_pos_A} (用户坐标)")
    print(f"✓ 随机生成的终点位置 C: {goal_pos_C} (用户坐标)")
    print(f"✓ 起点和终点位置不同: {start_pos_A != goal_pos_C}")
    print(f"✓ 起点不在Holes上: {start_pos_A not in frozen_cells}")
    print(f"✓ 终点不在Holes上: {goal_pos_C not in frozen_cells}")
    print(f"\n使用的随机种子: {random_seed} (每次运行会不同)")
    
    env2 = GridEnv(width=9, height=7,
                   frozen_cells=frozen_cells,
                   start_pos=start_pos_A,  # 位置A：随机生成的起点
                   goal_pos=goal_pos_C,    # 位置C：随机生成的终点
                   is_slippery=False,
                   seed=123,
                   render_mode="ansi")
    
    print("\nEnvironment created:")
    env2.render()
    
    # 创建新的Q-Learning算法（使用与初始配置相同的超参数）
    q_learning2 = QLearning(env2,
                            learning_rate=0.8,  # 与初始配置保持一致
                            discount_factor=0.95,
                            epsilon=0.1,
                            epsilon_decay=0.995,
                            epsilon_min=0.01)
    
    # 训练
    print("\nStarting Q-Learning training with new configuration...")
    print("-"*80)
    q_learning2.train(num_episodes=5000, verbose=True, print_interval=500)
    
    # 打印Q表
    print_q_table(q_learning2)
    
    # 保存训练日志和Q表数据
    print("\nSaving training data...")
    save_training_log(result_dir, env2, q_learning2, config_name="changed")
    
    # 可视化
    print("\nGenerating convergence plot for new configuration...")
    fig4 = plot_convergence(q_learning2,
                           title="Q-Learning Convergence - Changed Configuration (A to C)")
    convergence_changed_path = os.path.join(result_dir, 'convergence_changed.png')
    plt.savefig(convergence_changed_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {convergence_changed_path}")
    plt.close(fig4)
    
    print("\nGenerating Q-table visualization for new configuration...")
    fig5 = visualize_q_table(q_learning2,
                            title="Q-Table Visualization - Changed Configuration (A to C)")
    qtable_changed_path = os.path.join(result_dir, 'qtable_changed.png')
    plt.savefig(qtable_changed_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {qtable_changed_path}")
    plt.close(fig5)
    
    print("\nGenerating optimal path visualization for new configuration...")
    fig6 = visualize_path(env2, q_learning2,
                         title="Optimal Path - Changed Configuration (A to C)")
    path_changed_path = os.path.join(result_dir, 'path_changed.png')
    plt.savefig(path_changed_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path_changed_path}")
    plt.close(fig6)
    
    # 保存收敛数据（用于后续分析）
    print("\nSaving convergence data...")
    convergence_data = {
        'episode_q_values_initial': q_learning1.episode_q_values,
        'episode_rewards_initial': q_learning1.episode_rewards,
        'episode_lengths_initial': q_learning1.episode_lengths,
        'episode_q_values_changed': q_learning2.episode_q_values,
        'episode_rewards_changed': q_learning2.episode_rewards,
        'episode_lengths_changed': q_learning2.episode_lengths,
    }
    convergence_file = os.path.join(result_dir, 'convergence_data.npz')
    np.savez(convergence_file, **convergence_data)
    print(f"Saved: {convergence_file}")
    
    # 保存环境地图信息
    print("\nSaving environment map...")
    env_map_file = os.path.join(result_dir, 'environment_map.txt')
    with open(env_map_file, 'w', encoding='utf-8') as f:
        f.write("Initial Configuration Environment Map:\n")
        f.write("="*80 + "\n")
        f.write(f"Grid Size: {env1.width} columns × {env1.height} rows\n")
        f.write(f"Start: {env1.start_pos_user}, Goal: {env1.goal_pos_user}\n")
        f.write("\nMap Description:\n")
        desc = env1.get_map_description()
        for i, row in enumerate(desc):
            f.write(f"Row {i}: {row}\n")
        
        f.write("\n\nChanged Configuration Environment Map:\n")
        f.write("="*80 + "\n")
        f.write(f"Grid Size: {env2.width} columns × {env2.height} rows\n")
        f.write(f"Start: {env2.start_pos_user}, Goal: {env2.goal_pos_user}\n")
        f.write("\nMap Description:\n")
        desc2 = env2.get_map_description()
        for i, row in enumerate(desc2):
            f.write(f"Row {i}: {row}\n")
    print(f"Saved: {env_map_file}")
    
    # 显示所有图表
    print("\n" + "="*80)
    print(f"Training completed! All results saved to: {result_dir}")
    print("="*80)
    print("\n结果文件列表:")
    for file in os.listdir(result_dir):
        file_path = os.path.join(result_dir, file)
        size = os.path.getsize(file_path)
        print(f"  - {file} ({size/1024:.2f} KB)")
    
    # 可选：显示图表（注释掉以便在无GUI环境中运行）
    # plt.show()


if __name__ == "__main__":
    main()

