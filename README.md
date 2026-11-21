# 单目平面尺寸测量：以 A4 纸为度量基准（免标定）

用一张标准 A4 纸（297 mm × 210 mm）作为平面度量基准，从**单幅手机照片**中测出桌面的长和宽，
**不需要相机内参标定**。本仓库只包含测量程序。

## 原理

A4 纸与被测桌面共面，因此桌面平面与图像之间是一个 3×3 的射影变换（单应）：

```
s · (u, v, 1)^T  =  H · (X, Y, 1)^T        H = K · [ r1  r2  t ]
```

`H` 有 8 个自由度，纸的四个角提供 8 个方程即可唯一确定；`H` 的两端分别是毫米与像素，
所以平面上任意两点的真实距离可以直接算出。内参 `K` 被吸收进 `H`，无需单独求解。

绝对尺度来自 A4 的已知边长——**只知道“是矩形”还不够，必须知道它多大**，否则 `H` 只能定到相差一个相似变换。

## 文件

| 文件 | 作用 |
|---|---|
| `code/geom.py` | 计算层：单应求解（DLT + Hartley 归一化 + SVD）、坐标变换、四边形量算、蒙特卡洛误差预测 |
| `code/measure_table.py` | 测量主程序：点选纸角与桌角 → 解 H → 测距 → 标注图 / 俯视校正图 / CSV |
| `code/analyze.py` | 误差统计：均值、标准差、MAE/RMSE、相对误差、95% 区间，并绘制三联图 |
| `code/calib_k.py` | 附加实验：由多张图像的自标定求解内参 K |

依赖：Python ≥ 3.9，`numpy`、`opencv-python`、`pillow`、`matplotlib`。


## 文档

`docs/` 下的说明文件：

| 文件 | 内容 |
|---|---|
| 任务清单.docx | 整体步骤与先后顺序 |
| 任务指南.docx | 每一步的具体操作、要改第几行 |
| 原理推导.docx | 从相机内参矩阵到单应测距的完整推导 |
| 内容清单.md | 目录索引：每样东西是什么、怎么用、改动的规矩 |

## 使用

```bash
pip install numpy opencv-python pillow matplotlib

# 算法自检（合成数据，不需要照片）
python code/measure_table.py --selftest

# 单张 / 批量测量（data/ 下放照片）
python code/measure_table.py --img data/01.jpg
python code/measure_table.py --batch "data/*.jpg"

# 误差统计与出图
python code/analyze.py

# 附加：解内参
python code/calib_k.py
```

点选顺序：先点 A4 四角，再点桌面四角，回车确认、`u` 撤销、`ESC` 重来。
结果写入 `data/results/`：`result.csv`、`picks.json`、标注图、俯视校正图、`stats/`、`figures/`。

**从零开始做一遍**：

1. `pip install numpy opencv-python pillow matplotlib`
2. 拍摄照片放进 `data/`（命名随意，程序按文件名排序）
3. `python code/measure_table.py --batch "data/*.jpg"` 逐张点选，程序自动解算
4. 打开 `code/analyze.py`，把**第 19、20 行**的 `TRUE_L`、`TRUE_W` 换成你用卷尺量到的桌面真值
5. `python code/analyze.py` —— 输出统计表与三联长图
6. （可选）`python code/calib_k.py` 解内参作为对照实验

## 精度

25 组独立测量，真值 600 mm × 400 mm：

| 指标 | 长 | 宽 |
|---|---|---|
| 均值 / mm | 581.53 | 402.74 |
| 标准差 / mm | 19.40 | 19.51 |
| 平均绝对相对误差 / % | 3.59 | 3.95 |
| 最大相对误差 / % | 9.33 | 11.18 |

- 长存在约 −3% 的系统性偏小，主要来自桌沿为圆角、角点定义模糊；宽基本无偏。
- 点选噪声理论上只对应 0.35% 的离散度，实测 3.2%～4.9%，说明误差主体是“点在哪”而非“点得多准”。
- 多视图自标定在本组数据上呈病态（主点落于画面外），反证了免标定路线的稳健性。

## 目录说明

- `code/` 四个程序，全部从仓库根目录执行，例如 `python code/measure_table.py --selftest`
- `data/` 自己放照片（01.jpg …），以及可选的 meta.csv；程序会自动建 `data/results/`
- `docs/` 任务清单、任务指南、原理推导、内容清单

## 说明

- 全部处理在本地完成，不依赖任何在线服务。
- 出于隐私考虑，仓库未附带原始照片与实验报告；算法本身与图像内容无关，用任意同场景照片即可复现。
