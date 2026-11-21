# geom.py -- 几何层: 单应求解 + 米制测距 (只依赖 numpy, 不读写图片)
#
# 坐标约定
#   图像 (u, v): 像素, 原点在左上角, u 向右, v 向下
#   纸面 (X, Y): 毫米, 原点在 A4 左上角, X 沿 297 边, Y 沿 210 边
# H 的方向: 像素 -> 毫米

import numpy as np

PAPER_W, PAPER_H = 297.0, 210.0
# A4 四角在纸面坐标下的值, 顺序: 左上 右上 右下 左下
PAPER_PLANE = np.array([[0.0, 0.0], [PAPER_W, 0.0],
                        [PAPER_W, PAPER_H], [0.0, PAPER_H]])


# 任意顺序的 4 点 -> 左上 右上 右下 左下
# 做法: 先绕质心按角度排成一圈(对转得很斜的四边形也稳), 再从最靠左上的点起头
# 注意: 早先用 x+y / y-x 直接找角, 桌面接近 45 度时会把同一个点判成两个角,
#       导致四个角变成三个、面积塌成 0, 所以改成按角度排序
def order_quad(pts):
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(ang)]                        # 顺时针排成一圈
    p = np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)   # 从左上角开始
    return p


# Hartley 归一化: 平移到质心, 缩放到平均距离 sqrt(2)
# 不做这一步, 像素(1e3)和毫米(1e2)混在一起, 矩阵条件数很差
def _normalize(pts):
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / d if d > 1e-12 else 1.0
    T = np.array([[s, 0.0, -s * c[0]],
                  [0.0, s, -s * c[1]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return T, (pts - c) * s


# DLT 求单应矩阵: 像素 -> 毫米, 需要 >= 4 对点
def homography(img_pts, plane_pts):
    img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    pln = np.asarray(plane_pts, dtype=np.float64).reshape(-1, 2)
    if len(img) != len(pln) or len(img) < 4:
        raise ValueError("至少需要 4 对且数量相等的点对")

    T, pn = _normalize(img)
    U, qn = _normalize(pln)

    # 每对点 2 个方程, 消掉尺度因子后是齐次的: A h = 0
    A = np.zeros((2 * len(img), 9), dtype=np.float64)
    for i in range(len(img)):
        x, y = pn[i]
        X, Y = qn[i]
        A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, X * x, X * y, X]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, Y * x, Y * y, Y]

    # ||Ah|| 最小且 ||h||=1: 取最小奇异值对应的右奇异向量
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(U) @ Hn @ T        # 反归一化
    if abs(H[2, 2]) > 1e-12:
        return H / H[2, 2]
    return H / np.linalg.norm(H)


# 像素 -> 毫米 (除以第三分量, 即"去齐次")
def to_plane(H, p):
    v = H @ np.array([p[0], p[1], 1.0], dtype=np.float64)
    return v[:2] / v[2]


# 两个像素点在桌面上的真实距离 (mm)
def dist(H, p, q):
    a = to_plane(H, p)
    b = to_plane(H, q)
    return float(np.linalg.norm(a - b))


# 独立验证点: 用没参与求解的点检查 H 准不准, 返回 (预测坐标, 误差 mm)
def check_point(H, img_pt, true_plane_pt):
    pred = to_plane(H, img_pt)
    err = float(np.linalg.norm(pred - np.asarray(true_plane_pt, dtype=np.float64)))
    return pred, err


# 四边形 -> 四条边长 / 两条对角线 / 面积 (quad 需按 左上右上右下左下 排序)
# long_side / short_side: 较长与较短的那条邻边, 用来当"桌面长、宽"
#   (照片里桌子可能横放也可能竖放, 用长短边区分, 免得同一张桌子的长宽颠倒)
def quad_props(H, quad_img):
    q = np.asarray(quad_img, dtype=np.float64)
    P = np.array([to_plane(H, p) for p in q])
    sides = [float(np.linalg.norm(P[(i + 1) % 4] - P[i])) for i in range(4)]
    diags = [float(np.linalg.norm(P[2] - P[0])), float(np.linalg.norm(P[3] - P[1]))]
    x, y = P[:, 0], P[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    a, b = sides[0], sides[1]
    return {
        "sides": sides,
        "long_side": max(a, b),
        "short_side": min(a, b),
        "diagonals": diags,
        "area_mm2": float(area),
        "rect_error_mm": float(abs(sides[0] - sides[2]) + abs(sides[1] - sides[3])),
    }


# 蒙特卡洛: 给 4 个角点加 sigma 像素的噪声, 重解 H, 统计测距的离散度
# 用来回答"点选差 1.5 像素, 最后的长宽会差多少 mm"
def mc_sigma(img_pts, plane_pts, p, q, sigma_px=1.5, trials=1000, seed=0):
    rng = np.random.default_rng(seed)
    ip = np.asarray(img_pts, dtype=np.float64)
    pl = np.asarray(plane_pts, dtype=np.float64)
    vals = np.empty(trials)
    for k in range(trials):
        Hk = homography(ip + rng.normal(0.0, sigma_px, ip.shape), pl)
        vals[k] = dist(Hk, p, q)
    return {"mean": float(vals.mean()), "std": float(vals.std())}


# 修 Windows 终端中文乱码, 各入口脚本开头调一下
# UTF-8 终端(VS Code 默认)用 utf-8, 老式 cmd 用当前代码页; 输出重定向时用 utf-8
def fix_console():
    import os
    import sys
    if os.name != "nt":
        return
    try:
        enc = "utf-8"
        if sys.stdout.isatty():
            import ctypes
            cp = ctypes.windll.kernel32.GetConsoleOutputCP() or 936
            enc = "utf-8" if cp == 65001 else "cp%d" % cp
        sys.stdout.reconfigure(encoding=enc, errors="replace")
        sys.stderr.reconfigure(encoding=enc, errors="replace")
    except Exception:
        pass
