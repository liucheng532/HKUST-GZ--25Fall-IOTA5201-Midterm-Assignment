# Q-Learning 作业实现

## 项目结构

```
code/
├── environment.py    # 自定义Grid环境类（参考FrozenLake）
├── q_learning.py     # Q-Learning算法实现
├── main.py           # 主程序（训练和可视化）
└── requirements.txt  # 依赖包
```

## 环境说明

- **网格大小**: 9列 × 7行（width=9, height=7）
- **环境**: 基于Gymnasium的FrozenLake环境
- **Holes**: 危险区域（Frozen cells），agent进入会结束episode并获得0奖励
- **起点**: (6, 1) - 坐标格式为(x, y)，其中x是列索引，y是行索引
- **终点**: (6, 7) - 坐标格式为(x, y)，其中x是列索引，y是行索引

## 环境设置

### 使用Conda（推荐）

已创建名为 `qlearning_env` 的conda环境，包含所有必要的依赖。

**激活环境：**
```bash
conda activate qlearning_env
```

**如果还没有创建环境，可以运行：**
```bash
conda create -n qlearning_env python=3.9 -y
conda activate qlearning_env
conda install numpy matplotlib -y
pip install gymnasium
```

或者使用pip直接安装所有依赖：
```bash
conda activate qlearning_env
pip install -r requirements.txt
```

### 使用pip

如果不想使用conda，也可以使用pip安装依赖：

```bash
pip install -r requirements.txt
```

## 运行程序

### 运行主程序（训练Q-Learning）
```bash
cd code
python main.py
```

### 查看环境可视化

**方法1: GUI可视化（推荐，需要图形界面）**
```bash
python visualize_env.py
```
这将打开一个GUI窗口显示环境，类似OpenAI Gym的FrozenLake可视化。

**方法2: 测试环境（文本模式）**
```bash
python test_env.py
```
这将以文本形式显示环境地图。

**方法3: 在代码中使用可视化**
```python
from environment import GridEnv

# 创建环境（GUI模式）
env = GridEnv(..., render_mode="human")
env.render()  # 显示GUI窗口

# 或使用文本模式
env = GridEnv(..., render_mode="ansi")
env.render()  # 打印文本地图
```

## 输出文件

运行后会生成以下可视化图片：
- `convergence_initial.png`: 初始配置的Q值收敛图
- `qtable_initial.png`: 初始配置的Q表可视化
- `path_initial.png`: 初始配置的最优路径
- `convergence_changed.png`: 改变配置后的Q值收敛图
- `qtable_changed.png`: 改变配置后的Q表可视化
- `path_changed.png`: 改变配置后的最优路径

## 主要功能

1. **环境构建**: 实现了9×7的grid环境，包含frozen cells
2. **Q-Learning算法**: 实现了标准的Q-learning算法
3. **可视化**: 
   - Q值收敛过程
   - Q表热力图
   - 最优路径可视化
4. **配置变更**: 支持改变起点和终点位置

## 注意事项

- 坐标系统：用户输入的坐标格式为(x, y)，其中x是列索引，y是行索引
- 系统内部转换为(row, col)格式使用
- 如果起点/终点坐标需要修改，请在`main.py`中修改相应的参数

## 作业要求完成情况

- ✅ 任务1: 实现Q-Learning算法
- ✅ 任务2: 环境设置（9×7 grid，frozen cells）
- ✅ 任务3: 训练（5000 episodes）
- ✅ 任务4: 收敛图生成和分析
- ✅ 任务5: 改变初始和目标状态（A到C）

