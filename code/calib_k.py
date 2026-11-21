# calib_k.py -- 用多张照片解相机内参 K (加分项; 测长宽用不到它)
#
# 前提: 已经跑过 measure_table.py, data/results/picks.json 里有点选坐标
# 用法: python code/calib_k.py
# 输出: 终端打印 K 和四个内参 + 合理性检查, 同时写 data/results/K.txt
#
# 这个脚本不用改任何东西: 照片尺寸自动从 data/ 里读

import glob
import json
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    print("没装 opencv, 先执行:  pip install opencv-python")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom

PICKS = "data/results/picks.json"
OUT_TXT = "data/results/K.txt"
N_USE = 20                    # 用几张; 张数越多越稳, 但每张点 4 个角比较费事
# A4 四角在纸面坐标下的毫米值, 顺序: 左上 右上 右下 左下
OBJ = np.array([[0, 0, 0], [297, 0, 0], [297, 210, 0], [0, 210, 0]], np.float32)


def read_image_size(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    return w, h


def pick_names(db, n):
    """均匀挑 n 张, 让姿态尽量分散(照片是按距离/角度排的号)"""
    names = sorted(db)
    if len(names) <= n:
        return names
    step = len(names) / float(n)
    return [names[int(i * step)] for i in range(n)]


def main():
    geom.fix_console()

    if not os.path.exists(PICKS):
        print("找不到 %s, 先把 measure_table.py 跑一遍" % PICKS)
        return
    with open(PICKS, "r", encoding="utf-8") as f:
        db = json.load(f)

    names = pick_names(db, N_USE)
    if len(names) < 3:
        print("只找到 %d 张点选记录, 至少要 3 张才能解内参" % len(names))
        return

    img_files = sorted(glob.glob("data/*.jpg") + glob.glob("data/*.png")
                       + glob.glob("data/*.jpeg"))
    if not img_files:
        print("data/ 里没有图片, 读不到照片尺寸")
        return
    W, H = read_image_size(img_files[0])

    obj_pts = [OBJ for _ in names]
    img_pts = [np.array(db[n]["paper"], np.float32) for n in names]

    ok, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, (W, H), None, None,
        flags=cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        | cv2.CALIB_ZERO_TANGENT_DIST)

    if not ok:
        print("标定没收敛, 换更多张照片试试(把脚本里的 N_USE 改大)")
        return

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print("=" * 62)
    print("用的 %d 张: %s" % (len(names), ", ".join(names)))
    print("图像尺寸: %d x %d" % (W, H))
    print("K =")
    print(K)
    print("fx = %.1f   fy = %.1f   cx = %.1f   cy = %.1f" % (fx, fy, cx, cy))
    print("-" * 62)
    print("[1] fx 和 fy 接近吗: %.2f 倍" % (max(fx, fy) / min(fx, fy)))
    print("[2] cx 接近 W/2 吗: %.1f vs %.1f" % (cx, W / 2.0))
    print("[3] cy 接近 H/2 吗: %.1f vs %.1f" % (cy, H / 2.0))
    print("[4] 等效焦距约 %.1f 像素, 折算 %.1f mm(按 36mm 画幅算)"
          % (fx, fx * 36.0 / W))
    print("=" * 62)
    print("说明: 这一步只验证'多张照片能标定出内参', 测长宽用不到 K")

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("照片: %s\n" % ", ".join(names))
        f.write("图像尺寸: %d x %d\n" % (W, H))
        f.write("fx=%.3f fy=%.3f cx=%.3f cy=%.3f\n" % (fx, fy, cx, cy))
        f.write("K=\n%s\n" % np.array2string(K, precision=3))
    print("已写入 %s" % OUT_TXT)


if __name__ == "__main__":
    main()
