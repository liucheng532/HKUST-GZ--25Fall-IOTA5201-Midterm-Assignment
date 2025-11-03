# Q-Learning 实验报告使用说明

## 文件说明

- `report.tex`: 主LaTeX报告文件

## 在Overleaf中使用

### 步骤1：上传文件

1. 登录 [Overleaf](https://www.overleaf.com/)
2. 创建新项目或打开已有项目
3. 将 `report.tex` 文件上传到项目根目录

### 步骤2：上传图片资源

需要在Overleaf项目中创建以下目录结构，并上传相应的图片文件：

```
your-overleaf-project/
├── report.tex
└── result/
    └── 2025-1103-0021/
        ├── convergence_initial.png
        ├── convergence_changed.png
        ├── qtable_initial.png
        ├── qtable_changed.png
        ├── path_initial.png
        └── path_changed.png
```

**方法1：直接上传文件夹**
- 在Overleaf中创建 `result/2025-1103-0021/` 目录
- 从本地 `result/2025-1103-0021/` 文件夹上传所有PNG图片

**方法2：修改图片路径**
- 如果图片路径不同，可以修改 `report.tex` 中的图片路径
- 查找 `\includegraphics` 命令，修改路径为你自己的路径

### 步骤3：设置编译器

1. 在Overleaf项目设置中，选择编译器为 **XeLaTeX** 或 **pdfLaTeX**
2. 由于使用了 `ctex` 包，推荐使用 **XeLaTeX** 编译器以获得更好的中文支持

### 步骤4：编译

1. 点击 "Recompile" 按钮
2. 如果遇到编译错误，检查：
   - 图片文件路径是否正确
   - 所有必需的LaTeX包是否都已安装（Overleaf通常已包含）

## 需要的LaTeX包

报告使用了以下LaTeX包（Overleaf通常已包含）：

- `ctex` - 中文支持
- `amsmath` - 数学公式
- `graphicx` - 图片插入
- `hyperref` - 超链接
- `geometry` - 页面设置
- `float` - 浮动体控制
- `subfigure` - 子图（如果需要修改为subcaption，可能需要更新）
- `listings` - 代码显示
- `xcolor` - 颜色支持
- `booktabs` - 表格美化

## 图片文件说明

报告引用了以下图片（位于 `result/2025-1103-0021/`）：

1. `convergence_initial.png` - 初始配置的收敛图
2. `convergence_changed.png` - 改变配置的收敛图
3. `qtable_initial.png` - 初始配置的Q表可视化
4. `qtable_changed.png` - 改变配置的Q表可视化
5. `path_initial.png` - 初始配置的最优路径
6. `path_changed.png` - 改变配置的最优路径

## 自定义修改

### 修改图片路径

如果图片路径不同，搜索并替换 `report.tex` 中的路径：

```latex
% 原始路径
\includegraphics[width=0.9\textwidth]{result/2025-1103-0021/qtable_initial.png}

% 修改为你的路径，例如：
\includegraphics[width=0.9\textwidth]{figures/qtable_initial.png}
```

### 使用不同日期的结果

如果使用不同日期生成的结果（例如 `2025-1103-0009`），批量替换路径：

1. 在Overleaf的编辑器中，使用查找替换功能（Ctrl+H 或 Cmd+H）
2. 查找：`result/2025-1103-0021/`
3. 替换为：`result/2025-1103-0009/`（你的结果文件夹）

## 报告内容结构

报告包含以下主要部分：

1. **引言** - Q-Learning算法介绍及其意义
2. **环境与状态描述** - 网格环境的详细说明
3. **实验配置** - 两种配置和超参数设置
4. **Q表分析** - Q值分布和收敛前后对比
5. **收敛性分析** - 两种配置的收敛过程
6. **最优路径分析** - 学习到的最优路径可视化
7. **结果分析与观察** - 实验结果总结和结论

## 注意事项

1. **中文支持**：报告使用了 `ctex` 包，确保使用XeLaTeX编译器
2. **图片格式**：确保图片是PNG或PDF格式，分辨率足够（建议至少300 DPI）
3. **编译时间**：第一次编译可能需要较长时间，因为需要处理图片和中文
4. **版本控制**：Overleaf支持Git，可以连接GitHub进行版本控制

## 编译问题排查

如果编译失败，常见原因：

1. **缺少图片文件**：检查所有图片路径是否正确，文件是否存在
2. **编码问题**：确保LaTeX文件使用UTF-8编码
3. **包缺失**：检查是否有缺少的包（Overleaf通常会自动安装）
4. **编译器选择**：尝试切换XeLaTeX和pdfLaTeX

## 联系信息

如有问题，请联系：
- 作者：liucheng
- 邮箱：cliu425@connect.hkust-gz.edu.cn
