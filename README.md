# PROPBR

**方向反射与漫反射物理一致的可微分辐射传输模型（叶片 BRF 前向模拟与生化参数反演）**

PROPBR 将计算机图形学的物理渲染框架（PBR，Smith–GGX 微面模型）与经典叶片辐射传输模型
PROSPECT-D 耦合，构建物理一致、全可微的叶片双向反射因子（BRF）模型：

- **PROPBR**：显式耦合「表面镜面反射 + 内部漫反射」。镜面反射核 `K_s` 用预计算 2D-LUT
  经 KAN 网络算子化为处处可微的代理；基于 PyTorch 自动微分实现 GPU 批量反演。
- **sPROPBR**：PROPBR 的简化版，用一个等效表面反射系数 μ 代替几何项，适用于缺少完整
  观测几何的近端高光谱影像（像素级叶绿素制图）。

本仓库只包含**前向模拟 BRF** 与**反演参数**所需的最少代码。

---

## 目录结构

```
.
├── README.md
├── environment.yml      # conda 环境
├── func.py              # PROPBR / sPROPBR 前向、参考镜面扣除、Adam/AdamW 批量反演
├── prospect.py          # numpy 版 PROSPECT-D / PROSPECT-PRO（func.py 依赖）
├── prospect_h.py        # 面向 HSI 的 PROSPECT 封装
├── kan_model.py         # KsKAN：镜面反射核的 KAN 代理（2-4-1, B-spline, grid=10）
├── trf_solver.py        # torch GPU 批量 Trust-Region（dogleg）求解器（可选反演器）
├── metric.py            # RMSE / R² 评估
├── verify_fixes.py      # 回归检查：光学参考 / AD / 边界 / 最优解 / 两种模型反演
├── prosailt/            # PROSPECT-5/D/PRO 的 PyTorch 实现（kan_torch 的依赖）
└── lut/                 # 模型权重与预计算查找表
    ├── ks_kan_weights1.pt   # KsKAN 权重
    ├── DLM_nr.npy           # 折射率光谱
    └── Resampler.npy        # 2101 → 196 波段重采样矩阵
```

数据不随仓库发布，见「数据说明」。

---

## 环境安装

```bash
conda env create -f environment.yml
conda activate propbr

# GPU用户按PyTorch官方安装页选择本机对应的CUDA构建：
# https://pytorch.org/get-started/locally/
```

也可在已有Python环境中安装依赖：

```bash
python -m pip install -r requirements.txt
```

PyTorch的CPU或CUDA构建可按[官方安装页](https://pytorch.org/get-started/locally/)选择。官方prosail包为可选依赖：`python -m pip install prosail`。

> 设备选择：模块加载时按 `cuda`（可用时）→ `cpu` 的顺序自动决定，可用环境变量
> `PROPBR_DEVICE=cpu` 强制覆盖；所有前向与反演函数都接受 `device=` 参数，可显式指定。
> 纯 CPU 环境可直接 `import func` 并使用 `device='cpu'`。
>
> 若要用 `torch.func.jacfwd` / `vmap` 直接对前向函数求导（绕过 `invert_mse`），请先调用
> `func.warmup(device)`：functorch 变换内不允许构建模块（`nn.Module.to` 会就地改写参数）。
>
> 装好后运行 `python verify_fixes.py --device all`，检查本机可用的 CPU/GPU。
> 可分别用 `--device cpu` 或 `--device cuda` 指定设备，也可设置 `PROPBR_DEVICE=cpu`。
> `func.py` 的模型资源按模块位置定位，导入和自检不要求当前目录为仓库根目录。
> 光谱库优先取自官方 `prosail` 包；若未安装，`func.py` / `prospect.py` 会自动回退到仓库内的
> `prosailt` 副本（同一套数组，float32 存储），因此也可以不装 `prosail`。
> 已在本机验证：Windows + Python 3.10 + torch 2.1.0 (CUDA) + prosail 2.0.5。

---

## 快速开始

### 1) 前向模拟 BRF

PROPBR 的参数为 `[N, Cab, Car, Anth, Cw, Cm, Cbrown, σ]`，使用经验关系
`f_s = 0.66·σ^-0.63`；9参数版本在末尾增加自由的 `f_s`。
sPROPBR 的参数为 `[N, Cab, Car, Anth, Cw, Cm, Cbrown, μ]`，由 μ 表示有效表面反射系数。
单角度输出为 `(B, 2101)`，对应400–2500 nm、1 nm间隔。

| 前向入口 | 模型 | 几何与输出 |
|---|---|---|
| `kan_torch` | 8参数PROPBR，fs由σ确定 | 相位角15°；`(B, 2101)` |
| `kan_torch_lF` | 9参数PROPBR，fs自由 | 相位角35°；`(B, 2101)` |
| `full_torch` | 9参数PROPBR，解析镜面核 | 相位角35°；`(B, 2101)` |
| `full_torch_l` | 9参数PROPBR，解析镜面核 | 24个方向、12个相位角10–65°；450–1650 nm；展平为`(B, 24*1201)` |
| `spropbr_torch` | 8参数sPROPBR，以μ替代σ | 不需要几何输入；`(B, 2101)` |

`Kf_torch` 保留为 `spropbr_torch` 的兼容别名。

PROPBR 使用 `DHR − ρ_ref(σ) + Ks·F_Schlick`，其中
`ρ_ref(σ) = F0(n_internal)·exp(−1.26σ^1.79)`，内部折射率取自PROSPECT光谱库，
表面折射率取自 `lut/DLM_nr.npy`。按论文式（9），
`F_Schlick = F0(n_surface) + (1−F0(n_surface))·(1−cos(phase/2))^5`。
KAN输出的Ks已包含经验系数fs；自由fs版本通过 `Ks·(fs/f(sigma))` 调整幅度。

sPROPBR 使用 `DHR − [1−tav(40°, n_internal)] + μ·F0(n_surface)`。
零吸收附近的PROSPECT层叠采用无损极限的稳定展开，保留吸收参数的导数。

```python
import torch, func

device = 'cuda' if torch.cuda.is_available() else 'cpu'
p8 = torch.tensor([[1.5, 40.0, 10.0, 0.0, 0.01, 0.005, 0.0, 0.3]], device=device)
brf = func.kan_torch(p8)          # (1, 2101)，PROPBR，15°

# 纯 CPU 用法
p8_cpu = p8.cpu()
brf_cpu = func.kan_torch(p8_cpu)  # 或显式 func.kan_torch(p8_cpu, device='cpu')
print(brf.shape, float(brf.mean()))

p9 = torch.cat([p8, p8.new_tensor([[1.5]])], dim=1)
brf9 = func.kan_torch_lF(p9)      # (1, 2101)，PROPBR（f_s 自由）

q8 = p8.clone()
q8[:, 7] = 1.2                   # 第8列为μ
sbrf = func.spropbr_torch(q8)     # (1, 2101)，sPROPBR
```

参数物理范围（与论文一致）：

| 参数 | N | Cab | Car | Anth | Cw | Cm | Cbrown | σ | f_s |
|---|---|---|---|---|---|---|---|---|---|
| 下界 | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 0.6 |
| 上界 | 3.5 | 120 | 30 | 40 | 0.06 | 0.05 | 1.0 | 1.0 | 3.5 |

sPROPBR前7个参数的范围相同，μ的反演范围为0–30。

### 2) 反演参数

`func.invert_mse` 用 sigmoid 有界重参数化 + 反向自动微分，批量反演：

```python
import numpy as np, torch, func

low  = torch.tensor([1., 0, 0, 0, 0, 0, 0, 0.01], device=device)
high = torch.tensor([3.5, 120, 30, 40, 0.06, 0.05, 1., 1.], device=device)

obs = brf.detach()                # 可直接运行的合成示例；实测BRF同样使用(B, 2101)
p0  = np.array([0.2, 0.33, 0.33, 0.0, 0.17, 0.1, 0.0, 0.3])   # 归一化初值

paras = func.invert_mse(obs=obs, p0_norm=p0, low=low, high=high,
                        n_iters=500, lr=0.05, batch_size=1000,
                        optimizer='adam', device=device)
# paras: (B, 8)，物理参数
# CPU：把 low/high/obs 建在 CPU 上，并传 device='cpu'，其余不变
```

- 上例显式使用Adam、500步、学习率0.05，默认余弦退火到0.0025；函数默认值仍为
  AdamW、400步和学习率0.02。AdamW采用PyTorch默认权重衰减0.01，Adam的权重衰减为0。
- 对sPROPBR反演时，传入`forward_model=func.spropbr_torch`，将最后一项的边界改为0和30。
  对多角度联合反演，传入`forward_model=func.full_torch_l`及对应9参数边界；同一叶片的24个方向
  按10–65°两组顺序展平为一行。每个样本保留所有迭代与最后一步更新中的最低MSE解。
- 备选反演器：`trf_solver.batch_trf`（GPU 批量 dogleg 信赖域，逐样本更快收敛，但性状精度
  在多角度联合反演中低于 Adam，见论文）。

---

## 运行时对比（2,244 条果园光谱 × 2,101 波段）

下表为论文记录的实验结果。本文档修正后的实现使用Schlick项及新的最优解保存逻辑，
相应实测性状指标和运行时间需要使用原始数据、实验配置重新测定。

协议：9 参数 PROPBR（参考镜面按 `F0(n)·exp(−1.26σ^1.79)` 扣除，35°）；Adam 500 迭代、
batch 1000；`curve_fit` 为 scipy TRF，前向使用仓库内 numpy 版 PROSPECT。单位：Cab/Car 为
μg/cm²，EWT/LMA 为 mg/cm²。

| 方案 | 时间 | 每谱 | 池化 RMSE | 中位 RMSE | Cab | Car | EWT | LMA |
|---|---|---|---|---|---|---|---|---|
| **GPU-Adam**（RTX 4070） | **27.6 s** | **12.3 ms** | 0.00449 | 0.00386 | **13.52** | 5.31 | **1.86** | 3.09 |
| CPU-Adam（20 核） | 322.3 s | 143.6 ms | 0.00449 | 0.00386 | **13.52** | 5.31 | **1.86** | 3.09 |
| curve_fit 单核 | 823.9 s | 367.1 ms | **0.00396** | **0.00327** | 13.89 | **4.60** | 1.90 | **2.74** |
| curve_fit 20 核 | 191.3 s | 85.2 ms | **0.00396** | **0.00327** | 13.89 | **4.60** | 1.90 | **2.74** |

要点：GPU 批量反演相对 CPU 同算法快 11.7×、相对单核 scipy TRF 快 29.9×；TRF 在单角度
数据上谱拟合略好，Adam 在 Cab/EWT 上更好——两条路线的差异在 2–13%，属求解器落点差异。

---

## 数据说明

本仓库**不含数据**。论文使用的数据集：

- **#LOTUS**（多角度 BRF，255 叶 × 24 方向）：Federated Research Data Repository，
  Peters & Noble (2022, 2024)；
- **#Global / #Orchard**（固定角度叶片 BRF）：汇编自公开数据库（EcoSIS 等），见论文引用；
- **#Apple / #Grape**（近端 HSI）：作者实测 / Data INRAE 公开数据。

使用时可把光谱存为 `(N, 2101)` 的 float32 数组（400–2500 nm，1 nm），直接送入
`func.invert_mse`；真值性状单位与论文一致即可。

---

## 引用与许可

代码沿用本仓库的GPL-3.0许可，全文见[LICENSE](LICENSE)。efficient-kan的MIT许可及其他来源说明见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。数据遵循各原始数据集许可。软件引用信息见[CITATION.cff](CITATION.cff)。
