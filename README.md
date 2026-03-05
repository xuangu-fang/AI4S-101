# AI4S 公开课 Lesson 1：流体场自编码器

**AI4S-101 公开课 - 表征学习**

本仓库实现基于自编码器 (Autoencoder) 的流体涡量场低维表征学习，使用 **Re=100 圆柱绕流** 数据展示卡门涡街 (Kármán Vortex Street) 的压缩、重构、插值与异常检测。

---

## 目录

- [项目结构](#项目结构)
- [各文件说明](#各文件说明)
- [输出结果一览](#输出结果一览)
- [运行步骤 (Step-by-Step)](#运行步骤-step-by-step)
- [数据来源](#数据来源)
- [依赖与环境](#依赖与环境)

---

## 项目结构

```
AI4S-101/
├── README.md                    # 本说明文档
├── requirements.txt             # Python 依赖
├── generate_data.py             # 数据下载与预处理
├── models.py                    # 自编码器模型定义
├── train.py                     # 训练与可视化主脚本（命令行）
├── AI4S_Lesson1_Fluid_AE.ipynb  # Jupyter 交互式 AE 教程
├── AI4S_Lesson1_Fluid_VAE.ipynb # Jupyter 交互式 VAE 生成模型教程
├── data/                        # 数据目录
│   ├── flow_field.npy          # 预处理后的流场数据（仓库已自带，可直接使用；也可通过 generate_data.py 从原始数据重新生成）
│   ├── flow_field_params.npy   # 归一化参数（可选）
│   └── DATA/FLUIDS/            # 原始 MAT 数据（可选，本仓库默认不包含，需自行下载 DATA.zip 并解压）
│       └── CYLINDER_ALL.mat
└── outputs/                     # 所有可视化与模型输出（运行 train.py 后生成）
    ├── flow_fields.png
    ├── vortex_animation.gif
    ├── training_curves.png
    ├── reconstruction.png
    ├── interpolation.png
    ├── interpolation_animation.gif
    ├── anomaly_detection.png
    ├── pca_vs_ae.png
    └── flow_ae_model.pth
```

---

## 各文件说明

### 1. `generate_data.py` — 数据准备

| 功能 | 说明 |
|------|------|
| **作用** | 从 MAT 文件加载圆柱绕流涡量场，重采样到 64×128，归一化到 [-1, 1]，并保存为 NumPy 数组。 |
| **主要函数** | `load_cylinder_data()` 加载 MAT；`resample_to_target_size()` 重采样；`normalize_data()` 归一化；`prepare_flow_data()` 一键执行并保存。 |
| **输入** | `data/DATA/FLUIDS/CYLINDER_ALL.mat`（需先下载 [DATA.zip](http://dmdbook.com/DATA.zip) 并解压到 `data/`）。 |
| **输出** | `data/flow_field.npy`（形状 `(151, 64, 128)`），以及可选的 `data/flow_field_params.npy`。 |
| **备用** | 若无 MAT 文件，可设置 `use_synthetic=True` 调用 `generate_synthetic_vortex_data()` 生成合成涡街数据。 |

### 2. `models.py` — 模型定义

| 功能 | 说明 |
|------|------|
| **作用** | 定义流体场自编码器：Encoder（3 层 Conv2d + Linear）、Decoder（Linear + 3 层 ConvTranspose2d）、以及 MSE / 物理约束损失。 |
| **类与函数** | `Encoder`, `Decoder`, `FlowAE`；`mse_loss()`, `physics_informed_loss()`。 |
| **输入/输出** | 输入 `(B, 1, 64, 128)` → 潜在向量 `(B, 16)` → 重构 `(B, 1, 64, 128)`。 |
| **直接运行** | `python models.py` 会做一次前向与损失测试，不读写数据。 |

### 3. `train.py` — 训练与可视化（命令行一键跑全流程）

| 功能 | 说明 |
|------|------|
| **作用** | 加载 `flow_field.npy`，训练 FlowAE，并生成所有图表、GIF 和保存模型。 |
| **流程概览** | 环境设置 → 加载数据 → 流场静态图 → 涡街动画 → 划分 train/test → 建模型 → 训练 → 训练曲线 → 重构对比 → 潜在空间插值（图+动画）→ 异常检测/修复 → PCA vs AE → 保存 `flow_ae_model.pth`。 |
| **输入** | `data/flow_field.npy`（须先运行 `generate_data.py` 生成）。 |
| **输出** | 全部在 `outputs/` 下，见 [输出结果一览](#输出结果一览)。 |
| **用法** | `python train.py`（默认 CPU；若需 GPU 可自行改 `setup_environment()` 中的 `device`）。 |

### 4. `AI4S_Lesson1_Fluid_AE.ipynb` — AE 交互式教程

| 功能 | 说明 |
|------|------|
| **作用** | 与 `train.py` 逻辑一致，但拆成多节：环境、数据加载、流场与动画、模型、训练、重构、潜在空间插值、滑块交互、异常检测、PCA vs AE、总结。 |
| **依赖** | 需先有 `data/flow_field.npy`（仓库默认已包含；如需从原始数据重新生成，可先运行 `generate_data.py`）。 |
| **输出** | 图表在 Notebook 内显示；若在代码里保存到 `outputs/`，则与 `train.py` 输出位置一致。模型可保存为 `outputs/flow_ae_model.pth`。 |

### 5. `AI4S_Lesson1_Fluid_VAE.ipynb` — VAE 生成模型交互式教程

| 功能 | 说明 |
|------|------|
| **作用** | 使用变分自编码器 (VAE) 对同一 Re=100 圆柱绕流涡量场进行建模，展示潜在空间采样与**生成新流场**的能力。整体 pipeline 与 AE 版本类似，包括数据加载、训练、重构、潜在空间插值、随机采样与滑块交互等。 |
| **特点** | Latent 空间为高斯分布 (`mu`, `logvar`)，通过重参数化技巧采样；所有图表标题、坐标轴和颜色条全部为英文，避免中文字体兼容问题。 |
| **依赖** | 需要 `data/flow_field.npy`（仓库默认已包含），不强制要求本地存在原始 `CYLINDER_ALL.mat`。 |
| **输出** | 主要为 Notebook 内联图像，包括 VAE 训练曲线、重构对比、潜在插值以及从先验随机采样得到的流场示例。 |

### 6. `requirements.txt` — 依赖列表

列出运行本课程代码所需的 Python 包及最低版本，用于 `pip install -r requirements.txt`。

---

## 输出结果一览

运行 `train.py`（或按相同流程跑完 Notebook）后，在 **`outputs/`** 目录下会得到以下文件：

| 文件名 | 内容说明 |
|--------|----------|
| `flow_fields.png` | 6 张涡量场快照（t=0 及等间隔时刻），展示卡门涡街随时间的演化。 |
| `vortex_animation.gif` | 涡量场时间序列动画，展示涡街脱落过程。 |
| `training_curves.png` | 训练/测试 MSE 随 epoch 变化（线性 + 对数尺度）。 |
| `reconstruction.png` | 4 组「原始 vs 重构 vs 误差图」，展示自编码器重构质量。 |
| `interpolation.png` | 8 张图：两帧流场之间在潜在空间线性插值得到的中间状态。 |
| `interpolation_animation.gif` | 上述插值过程的动画。 |
| `anomaly_detection.png` | 异常检测与修复：遮挡修复（上排）、加噪后去噪（下排），各带误差图。 |
| `pca_vs_ae.png` | 4 组对比：原始 / PCA 重构 / AE 重构，展示线性 vs 非线性降维。 |
| `flow_ae_model.pth` | 训练好的 FlowAE 模型权重，可用于 `model.load_state_dict(torch.load(...))`。 |

---

## 运行步骤 (Step-by-Step)

### 环境准备

```bash
cd /path/to/AI4S-101
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 1：准备数据

> **快速开始提示**：本仓库已经自带预处理好的 `data/flow_field.npy`，默认情况下你可以**直接跳到 Step 2 训练模型**，无需下载原始 DATA.zip。

1. **快速开始（推荐）**  
   - 克隆仓库后，检查 `data/flow_field.npy` 是否存在（默认已经包含）。  
   - 若存在，可直接执行 Step 2：`python train.py` 或运行 Notebook。

2. **从原始数据重新处理（可选，高级用法）**  
   仅当你希望完整体验「从原始数据到预处理」全流程时，才需要下面两步：

   1) **下载原始数据（若尚未下载）**  
      - 打开 [http://dmdbook.com/DATA.zip](http://dmdbook.com/DATA.zip) 下载并解压。  
      - 将解压得到的 `DATA` 文件夹放到项目下的 `data/` 目录，使路径为：  
        `data/DATA/FLUIDS/CYLINDER_ALL.mat`  

   2) **运行数据预处理脚本**  

      ```bash
      python generate_data.py
      ```

      - 会读取 `data/DATA/FLUIDS/CYLINDER_ALL.mat`，重采样并归一化，生成/覆盖 **`data/flow_field.npy`**。  
      - 若无 MAT 文件，可在 `generate_data.py` 的 `prepare_flow_data()` 调用中设 `use_synthetic=True`，改用合成数据。

### Step 2：训练模型并生成所有结果

```bash
python train.py
```

- 会自动：加载 `data/flow_field.npy` → 划分训练/测试 → 训练 FlowAE → 生成 `outputs/` 下所有图表、GIF 和 **`outputs/flow_ae_model.pth`**。  
- 默认使用 CPU；若需 GPU，需在 `train.py` 的 `setup_environment()` 中修改 `device`。

### Step 3（可选）：在 Jupyter 中逐步学习

1. 启动 Jupyter：  
   `jupyter notebook`  
   或  
   `jupyter lab`  

2. 打开 **`AI4S_Lesson1_Fluid_AE.ipynb`**。  

3. 确保已先执行 Step 1，使 `data/flow_field.npy` 存在。  

4. 从上到下依次运行各单元格，即可复现与 `train.py` 相同的流程（含插值、滑块、异常检测、PCA vs AE）。  

5. 若在 Notebook 中保存模型，建议路径为 **`outputs/flow_ae_model.pth`**，与 `train.py` 一致。

---

## 数据来源

- **数据集**：Fluid flow past a circular cylinder at **Re=100**（层流涡脱落）。  
- **来源**：Steven L. Brunton, [DMD book](http://dmdbook.com/) 配套 [DATA.zip](http://dmdbook.com/DATA.zip)。  
- **内容**：151 个涡量场快照，原始分辨率 449×199（沿流向×展向）。  
- **说明**：`data/DATA/FLUIDS/README.txt` 中有简要描述；本课程中重采样为 64×128 并归一化到 [-1, 1]。

---

## 依赖与环境

- **Python**：建议 3.8+。  
- **主要依赖**：见 `requirements.txt`（如 `torch`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `tqdm`, `ipywidgets`, `jupyter` 等）。  
- **GPU**：可选；默认脚本使用 CPU，可在代码中改为 CUDA 以加速训练。

---

## 快速检查清单

- [ ] 已安装依赖：`pip install -r requirements.txt`  
- [ ] 确认 `data/flow_field.npy` 存在（仓库默认已包含；若你选择重新处理，则由 `generate_data.py` 生成/覆盖）  
- [ ] （可选，高级）已下载并解压 DATA.zip，且存在 `data/DATA/FLUIDS/CYLINDER_ALL.mat`  
- [ ] （可选，高级）已运行 `python generate_data.py`，从原始数据重新生成 `data/flow_field.npy`  
- [ ] 已运行 `python train.py`，且 `outputs/` 下生成全部图表与 `flow_ae_model.pth`  

完成以上步骤即表示从数据到结果的全流程已跑通。
