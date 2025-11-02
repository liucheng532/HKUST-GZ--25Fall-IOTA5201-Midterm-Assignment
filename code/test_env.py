"""
测试环境是否正确设置
"""
from environment import GridEnv

def test_environment():
    """测试环境创建和基本功能"""
    print("="*80)
    print("Testing FrozenLake Environment")
    print("="*80)
    
    # 创建环境
    frozen_cells = [(2, 4), (3, 2), (3, 4), (4, 4), (5, 2), (5, 4),
                    (6, 2), (6, 4), (7, 1), (7, 2), (7, 4), (7, 6),
                    (7, 7), (8, 6)]
    
    # 注意：用户坐标系统：左下角为(1,1)，右上角为(9,7)
    # 使用"human"模式可以显示GUI窗口，需要安装pygame
    # 如果GUI不可用，可以使用"ansi"或"console"模式
    try:
        env = GridEnv(width=9, height=7,
                      frozen_cells=frozen_cells,
                      start_pos=(6, 1),  # 用户坐标：左下角为(1,1)
                      goal_pos=(6, 7),   # 用户坐标：右上角为(9,7)
                      is_slippery=False,
                      seed=123,
                      render_mode="human")  # 尝试使用GUI模式
    except Exception as e:
        print(f"GUI模式不可用 ({e})，使用文本模式")
        env = GridEnv(width=9, height=7,
                      frozen_cells=frozen_cells,
                      start_pos=(6, 1),  # 用户坐标
                      goal_pos=(6, 7),   # 用户坐标
                      is_slippery=False,
                      seed=123,
                      render_mode="ansi")  # 使用文本模式
    
    print("\n环境信息:")
    print(f"网格大小: {env.width}列 × {env.height}行")
    print(f"坐标系：左下角为(1,1)，右上角为(9,7)")
    print(f"起点坐标（用户坐标）: {env.start_pos_user}")
    print(f"终点坐标（用户坐标）: {env.goal_pos_user}")
    print(f"状态总数: {env.n_states}")
    print(f"动作总数: {env.n_actions}")
    print(f"动作: {env.ACTION_NAMES}")
    
    print("\n环境可视化 (文本模式):")
    env.render(mode="console")
    
    # 如果有GUI支持，也可以显示可视化窗口
    if env.render_mode == "human":
        print("\n尝试显示GUI可视化窗口...")
        state = env.reset()
        env.env.render()  # 直接调用gymnasium的render
        print("(GUI窗口应该已经打开)")
    
    print("\n地图描述 (desc):")
    desc = env.get_map_description()
    for i, row in enumerate(desc):
        print(f"Row {i}: {row}")
    
    print("\n测试重置功能:")
    state = env.reset()
    print(f"初始状态: {state}")
    pos = env._state_to_pos(state, user_coords=True)
    print(f"初始位置（用户坐标）: {pos}")
    
    print("\n测试移动:")
    print(f"当前位置状态: {state}")
    
    # 测试向右移动（RIGHT = 2）
    print("\n1. 向右移动 (RIGHT = 2):")
    result = env.step(env.RIGHT)
    if len(result) == 4:
        next_state, reward, done, info = result
    else:
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
    pos = env._state_to_pos(next_state, user_coords=True)
    print(f"   下一个状态: {next_state}")
    print(f"   位置（用户坐标）: {pos}")
    print(f"   奖励: {reward}")
    print(f"   是否结束: {done}")
    
    # 如果有GUI，显示更新后的状态
    if env.render_mode == "human":
        print("\n更新GUI可视化...")
        env.env.render()
    
    # 测试向下移动（DOWN = 1）
    print("\n2. 向下移动 (DOWN = 1):")
    result = env.step(env.DOWN)
    if len(result) == 4:
        next_state, reward, done, info = result
    else:
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
    pos = env._state_to_pos(next_state, user_coords=True)
    print(f"   下一个状态: {next_state}")
    print(f"   位置（用户坐标）: {pos}")
    print(f"   奖励: {reward}")
    print(f"   是否结束: {done}")
    
    # 如果有GUI，最后再显示一次
    if env.render_mode == "human":
        print("\n最终GUI可视化:")
        env.env.render()
        print("(按任意键关闭窗口或等待自动关闭)")
    
    print("\n" + "="*80)
    print("环境测试完成!")
    print("="*80)

if __name__ == "__main__":
    test_environment()

