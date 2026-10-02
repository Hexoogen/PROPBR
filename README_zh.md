# PROPBR 与 sPROPBR 代码

本目录可以作为独立 GitHub 仓库上传，包含前向模型、可微反演、局部与全局敏感性分析、KAN 代理及验证脚本。没有实测数据集；示例在运行时生成合成光谱。`assets` 中的折射率、吸收系数和网络权重是运行模型必需的数值常量。

## 安装与完整运行

在 Python 3.10 或更新版本的独立环境中，进入本目录后运行：

```bash
python -m pip install -e ".[full]"
python run_pipeline.py
```

GPU 环境使用 `python run_pipeline.py --device cuda`。默认 CPU、双精度、2 个合成叶片、10 nm 光谱采样、1200 次迭代、3 个初值；EFAST 的 N=1000，每个参数分配 1000 个采样点，总模型计算数为 N×参数数。

输出在 `outputs/demo`，包括模拟分量、反演参数与误差、逐波长 Jacobian、参数响应相关矩阵、EFAST 一阶/总效应指数、图件以及配置、依赖版本和文件哈希。

## 运行内容

- PROPBR 固定角度反演。
- PROPBR 多角度联合反演，同一叶片的各角度共享一组参数。
- sPROPBR 无几何输入的独立光谱反演。
- 物理参数导数、参数范围缩放导数和 Sigmoid 潜变量导数。
- 由完整 Jacobian 计算每个叶片的归一化响应相关矩阵。
- KAN 对解析 Ks 的训练格点及错位细格点评价。
- 参考半球镜面分量近似与数值积分的比较。
- PROPBR 与 sPROPBR 的 EFAST 全局敏感性分析。

完整 1 nm 采样可用 `--step 1`；全局分析的采样间隔由 `--global-step` 控制。较短的执行检查可用 `--iterations 200 --restarts 1 --skip-global`，但短迭代不保证反演收敛。

## 与论文和旧代码的对应

按论文公式，默认 PROPBR 有 8 个自由参数：N、Cab、Car、Anth、Cw、Cm、Cbrown、sigma；fsurf 由 sigma 的经验关系确定。仅在标定模式下将 fsurf 作为第 9 个自由参数。sPROPBR 有 8 个自由参数，末项为 mu。

新代码保证：Ks 中只乘一次 fsurf；方向性镜面项使用完整 Schlick 因子；PROPBR 扣除随粗糙度变化的参考半球镜面分量；sPROPBR 按公式 21 扣除 `1−tav(40,n_in)`。这些修正会影响与旧脚本的数值一致性，不能直接沿用旧数据实验的精度与加速比。

反演接口支持外部 NPZ 光谱、初值及有效波段掩码；说明与例子见英文 README。实测数据拟合、跨数据集校准、论文图表中的生化精度评价仍需自行提供数据。

## 上传

上传本目录中的源码、README、配置、测试、许可和 `src/propbr/assets`；不要上传 `.venv`、`outputs`、`__pycache__`、实测光谱或论文 Word 文件。`.gitignore` 已排除运行输出和数据目录。

本实现保留 PROSPECT 的 GPLv3 许可及 efficient-kan 的 MIT 声明。第三方来源和数值常量哈希见 `THIRD_PARTY_NOTICES.md` 与 `assets/manifest.json`。
