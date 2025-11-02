"""
可视化环境演示脚本
显示FrozenLake环境的GUI可视化窗口
"""
import time
from environment import GridEnv

def visualize_environment():
    """可视化环境并演示agent移动"""
    print("="*80)
    print("FrozenLake环境可视化演示")
    print("="*80)
    
    # 创建环境，使用human模式显示GUI
    frozen_cells = [(2, 4), (3, 2), (3, 4), (4, 4), (5, 2), (5, 4),
                    (6, 2), (6, 4), (7, 1), (7, 2), (7, 4), (7, 6),
                    (7, 7), (8, 6)]
    
    print("\n创建环境...")
    env = GridEnv(width=9, height=7,
                  frozen_cells=frozen_cells,
                  start_pos=(6, 1),  # 用户坐标系统：左下角为(1,1)
                  goal_pos=(6, 7),   # 用户坐标系统：右上角为(9,7)
                  is_slippery=False,
                  seed=123,
                  render_mode="human")
    
    print("环境已创建！")
    print(f"网格大小: {env.width}列 × {env.height}行")
    print(f"坐标系：左下角为(1,1)，右上角为(9,7)")
    print(f"起点（用户坐标）: {env.start_pos_user}")
    print(f"终点（用户坐标）: {env.goal_pos_user}")
    
    # 重置环境
    state = env.reset()
    print(f"\n初始状态: {state}")
    pos = env._state_to_pos(state, user_coords=True)
    print(f"初始位置（用户坐标）: {pos}")
    
    # 显示初始状态的可视化
    print("\n显示GUI窗口...")
    print("(如果GUI窗口没有出现，可能是因为在远程环境中运行)")
    env.render()
    
    # 演示一些动作
    print("\n演示agent移动...")
    actions_to_try = [
        (env.RIGHT, "向右"),
        (env.DOWN, "向下"),
        (env.DOWN, "向下"),
        (env.LEFT, "向左"),
        (env.LEFT, "向左"),
        (env.UP, "向上"),
    ]
    
    for action, action_name in actions_to_try:
        print(f"\n执行动作: {action_name} ({action})")
        result = env.step(action)
        if len(result) == 4:
            next_state, reward, done, info = result
        else:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        
        pos = env._state_to_pos(next_state, user_coords=True)
        blocked = info.get('blocked', False)
        if blocked:
            print(f"  移动被阻止！位置: {pos}, 奖励: {reward}, 结束: {done} (冰块区域不可进入)")
        else:
            print(f"  新状态: {next_state}, 位置: {pos}, 奖励: {reward}, 结束: {done}")
        
        # 更新可视化
        env.render()
        time.sleep(0.5)  # 暂停以便观察
        
        if done:
            print(f"  Episode结束！最终奖励: {reward}")
            if reward > 0:
                print("  🎉 到达终点！")
            else:
                print("  ⚠️  掉入Hole或超时")
            break
    
    print("\n" + "="*80)
    print("可视化演示完成！")
    print("="*80)
    print("\n提示: 如果GUI窗口没有显示，可能需要:")
    print("  1. 确保在有图形界面的环境中运行")
    print("  2. 设置DISPLAY环境变量（Linux）")
    print("  3. 或者使用render_mode='ansi'或'console'查看文本可视化")

if __name__ == "__main__":
    try:
        visualize_environment()
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n尝试使用文本模式...")
        # 回退到文本模式
        env = GridEnv(width=9, height=7,
                      frozen_cells=[(2, 4), (3, 2), (3, 4), (4, 4), (5, 2), (5, 4),
                                    (6, 2), (6, 4), (7, 1), (7, 2), (7, 4), (7, 6),
                                    (7, 7), (8, 6)],
                      start_pos=(6, 1),  # 用户坐标
                      goal_pos=(6, 7),   # 用户坐标
                      is_slippery=False,
                      seed=123,
                      render_mode="ansi")
        print("\n文本模式可视化:")
        env.render()

