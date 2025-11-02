# IOTA5201 Midterm Assignment - Q-Learning for Grid Environment

This repository contains the implementation of Q-Learning algorithm for a custom 9×7 grid environment, based on Gymnasium's FrozenLake environment.

## 📋 Project Structure

```
.
├── code/              # Source code
│   ├── environment.py    # Custom grid environment
│   ├── q_learning.py     # Q-Learning algorithm
│   ├── main.py          # Main training script
│   ├── test_env.py      # Environment testing script
│   ├── visualize_env.py  # Visualization script
│   ├── requirements.txt  # Dependencies
│   └── README.md        # Code documentation
├── frozenlake/        # Reference code from OpenAI Gym
│   ├── frozenlake_q_learning.py
│   └── frozenlake_q_learning.ipynb
├── doc/              # Assignment document
│   └── IOTA_Assignment.pdf
└── result/           # Training results (organized by date)
    └── YYYY-MMDD-HHMM/
        ├── convergence_initial.png
        ├── convergence_changed.png
        ├── qtable_initial.png
        ├── qtable_changed.png
        ├── path_initial.png
        ├── path_changed.png
        ├── qtable_initial.npy
        ├── qtable_changed.npy
        ├── convergence_data.npz
        ├── training_log_initial.txt
        ├── training_log_changed.txt
        └── environment_map.txt
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Conda (recommended)

### Installation

1. **Create conda environment:**
```bash
conda create -n qlearning_env python=3.9
conda activate qlearning_env
```

2. **Install dependencies:**
```bash
cd code
pip install -r requirements.txt
```

Or install directly:
```bash
pip install numpy matplotlib gymnasium gymnasium[toy-text] pandas seaborn tqdm
```

### Run Training

```bash
cd code
python main.py
```

This will:
- Train Q-Learning on initial configuration (Start: (6,1), Goal: (6,7))
- Train Q-Learning on randomly generated configuration (Task 5)
- Generate visualizations and save all results to `result/` folder with timestamp

## 📖 Assignment Tasks

### Task 1-4: Initial Configuration
- **Start Position**: (6, 1) - User coordinate system (bottom-left origin)
- **Goal Position**: (6, 7) - User coordinate system (top-right is (9,7))
- **Grid Size**: 9 columns × 7 rows
- **Frozen Cells (Holes)**: 14 cells acting as impassable walls

### Task 5: Changed Configuration
- **Random Initial State (A)**: Randomly generated from safe positions
- **Random Target State (C)**: Randomly generated from safe positions
- Ensures A and C are different and not on Holes

## 🎯 Key Features

### Environment
- Custom 9×7 grid based on Gymnasium FrozenLake
- **Coordinate System**: User-friendly 1-based coordinates with bottom-left origin
  - Bottom-left corner: (1, 1)
  - Top-right corner: (9, 7)
- **Frozen Cells**: Act as impassable walls (cannot enter)
- **Deterministic Actions**: No slippery movement

### Q-Learning Algorithm
- Epsilon-greedy exploration strategy
- Hyperparameters:
  - Learning rate: 0.8
  - Discount factor: 0.95
  - Epsilon: 0.1 (with decay)
- Convergence tracking and visualization

### Results Organization
- Each run creates a timestamped folder: `result/YYYY-MMDD-HHMM/`
- Contains:
  - Convergence plots (Q-values and rewards)
  - Q-table visualizations
  - Optimal path visualizations
  - Q-table data (`.npy` format)
  - Training logs
  - Environment maps

## 📊 Results

The algorithm typically achieves:
- **Success Rate**: 100% after convergence
- **Average Steps**: ~14-16 steps for optimal path
- **Convergence**: ~200 episodes to reach 90% success rate

## 🛠️ Code Files

- **`environment.py`**: Custom grid environment wrapper for FrozenLake
- **`q_learning.py`**: Q-Learning algorithm implementation
- **`main.py`**: Main training script with visualization
- **`test_env.py`**: Environment testing utilities
- **`visualize_env.py`**: GUI visualization demo

## 📝 Notes

- **Coordinate System**: All user-facing coordinates use 1-based system with bottom-left origin
- **Frozen Cells**: Treated as walls (blocked), not terminal states
- **Random Seed**: Can be set for reproducibility
- **Results**: Automatically organized by timestamp for easy comparison

## 📚 References

- Based on Gymnasium FrozenLake environment: https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- Reference implementation in `frozenlake/` folder

## 👤 Author

HKUST-GZ IOTA5201 Midterm Assignment

## 📄 License

MIT License
