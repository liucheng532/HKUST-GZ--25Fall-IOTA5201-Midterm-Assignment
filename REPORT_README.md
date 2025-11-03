# Q-Learning Experiment Report Usage Guide

## File Description

- `report.tex`: Main LaTeX report file (English)

## Using in Overleaf

### Step 1: Upload Files

1. Log in to [Overleaf](https://www.overleaf.com/)
2. Create a new project or open an existing project
3. Upload the `report.tex` file to the project root directory

### Step 2: Upload Image Resources

Create the following directory structure in your Overleaf project and upload the corresponding image files:

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

**Method 1: Upload folder directly**
- Create `result/2025-1103-0021/` directory in Overleaf
- Upload all PNG images from the local `result/2025-1103-0021/` folder

**Method 2: Modify image paths**
- If the image paths are different, modify the image paths in `report.tex`
- Find `\includegraphics` commands and modify paths to your own paths

### Step 3: Set Compiler

1. In Overleaf project settings, select the compiler as **pdfLaTeX** or **XeLaTeX**
2. Since the report doesn't use Chinese (no `ctex` package), **pdfLaTeX** is recommended

### Step 4: Compile

1. Click the "Recompile" button
2. If compilation errors occur, check:
   - Whether image file paths are correct
   - Whether all required LaTeX packages are installed (Overleaf usually includes them)

## Required LaTeX Packages

The report uses the following LaTeX packages (usually included in Overleaf):

- `amsmath` - Mathematical formulas
- `graphicx` - Image insertion
- `hyperref` - Hyperlinks
- `geometry` - Page settings
- `float` - Float control
- `subfigure` - Subfigures (if you need to modify to subcaption, may need to update)
- `listings` - Code display
- `xcolor` - Color support
- `booktabs` - Table beautification

## Image Files Description

The report references the following images (located in `result/2025-1103-0021/`):

1. `convergence_initial.png` - Convergence plot for initial configuration
2. `convergence_changed.png` - Convergence plot for changed configuration
3. `qtable_initial.png` - Q-table visualization for initial configuration
4. `qtable_changed.png` - Q-table visualization for changed configuration
5. `path_initial.png` - Optimal path for initial configuration
6. `path_changed.png` - Optimal path for changed configuration

## Customization

### Modifying Image Paths

If image paths are different, search and replace paths in `report.tex`:

```latex
% Original path
\includegraphics[width=0.9\textwidth]{result/2025-1103-0021/qtable_initial.png}

% Modify to your path, for example:
\includegraphics[width=0.9\textwidth]{figures/qtable_initial.png}
```

### Using Results from Different Dates

If using results from a different date (e.g., `2025-1103-0009`), batch replace paths:

1. In Overleaf editor, use find and replace (Ctrl+H or Cmd+H)
2. Find: `result/2025-1103-0021/`
3. Replace with: `result/2025-1103-0009/` (your results folder)

## Report Content Structure

The report includes the following main sections:

1. **Introduction** - Q-Learning algorithm introduction and significance
2. **Environment and State Description** - Detailed description of the grid environment
3. **Experimental Configuration** - Two configurations and hyperparameter settings
4. **Q-Table Analysis** - Q-value distribution and before/after convergence comparison
5. **Convergence Analysis** - Convergence process for both configurations
6. **Optimal Path Analysis** - Visualization of learned optimal paths
7. **Results Analysis and Observations** - Summary of experimental results and conclusions

## Notes

1. **Language**: The report is written in English
2. **Image format**: Ensure images are in PNG or PDF format with sufficient resolution (recommended at least 300 DPI)
3. **Compilation time**: First compilation may take longer due to image processing
4. **Version control**: Overleaf supports Git, can connect to GitHub for version control

## Troubleshooting

If compilation fails, common reasons:

1. **Missing image files**: Check if all image paths are correct and files exist
2. **Encoding issues**: Ensure LaTeX file uses UTF-8 encoding
3. **Missing packages**: Check if any packages are missing (Overleaf usually auto-installs)
4. **Compiler selection**: Try switching between XeLaTeX and pdfLaTeX

## Contact Information

For questions, please contact:
- Author: liucheng
- Email: cliu425@connect.hkust-gz.edu.cn
