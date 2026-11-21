# measure_table.py -- 用一张 A4 纸测桌面长宽, 不需要标定相机
#
# 用法(都在项目根目录执行)
#   自检: python code/measure_table.py --selftest
#   单张: python code/measure_table.py --img data/01.jpg
#   批量: python code/measure_table.py --batch "data/*.jpg"
#
# 操作
#   先点 A4 四个角, 再点桌面四个角(沿桌沿顺时针走一圈)
#   回车 = 确认    u = 撤销上一点    ESC = 清空重来
#
# 输出(默认写进 data/results/)
#   <名>_annotated.jpg   标注图(纸框/桌框/测量线)
#   <名>_rectified.jpg   俯视校正图(带 100mm 网格, 用来目视核对)
#   picks.json           点选坐标缓存, 便于复现, 也是标定内参的输入
#   result.csv           测量结果(只有测量值, 不含真值)
#
# 依赖: pip install opencv-python numpy      (pillow 可选, 用于摆正手机照片)

import argparse
import csv
import glob
import json
import os

import numpy as np

try:
    import cv2
except ImportError:          # 没装 opencv 也能跑 --selftest
    cv2 = None

import geom


# cv2.imread 读不了中文路径, 用 fromfile + imdecode
def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise IOError("读不到文件: %s" % path)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("解码失败: %s" % path)
    return img


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


# 手机照片带 EXIF 旋转标记, 先用 PIL 摆正; 没装 PIL 就退回 cv2
def load_image(path):
    try:
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        return imread_unicode(path)


# cv2 的窗口标题只认 ASCII, 中文文件名会乱码甚至建不出窗口
def ascii_title(name):
    s = "".join(c if 32 <= ord(c) < 127 else "_" for c in name)
    return s or "pick"


# 点选 4 个点(窗口按比例缩放), 返回原图坐标
# cv2.putText 只支持 ASCII, 所以界面提示写英文
class QuadPicker:
    def __init__(self, img, title, scale=1.0):
        self.scale = scale
        self.disp = (cv2.resize(img, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
                     if scale != 1.0 else img.copy())
        self.base = self.disp.copy()
        self.pts = []
        self.title = ascii_title(title)
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.pts) < 4:
            self.pts.append((x, y))
            self._redraw()

    def _redraw(self):
        d = self.base.copy()
        n = min(len(self.pts), 4)
        if n >= 2:
            poly = np.array(self.pts[:n], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(d, [poly], n == 4, (0, 255, 0), 2)
        for i in range(n):
            x, y = self.pts[i]
            cv2.circle(d, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(d, "P%d" % (i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(d, "click 4 corners | Enter=OK  u=undo  ESC=clear",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(d, "%d/4" % n, (10, self.disp.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow(self.title, d)

    def _closed(self):
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1
        except Exception:
            return True

    def run(self):
        self._redraw()
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 10) and len(self.pts) == 4:
                break
            if k == 27:                            # ESC 清空
                self.pts = []
                self._redraw()
            if k in (ord("u"), 8) and self.pts:    # 撤销
                self.pts.pop()
                self._redraw()
            if self._closed():                     # 直接点窗口右上角关掉
                break
        try:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)
        except Exception:
            pass
        s = self.scale
        return [(x / s, y / s) for x, y in self.pts]


def display_scale(img, target=1500):
    h, w = img.shape[:2]
    return min(1.0, target / float(max(h, w)))


# 标注图: 绿=纸, 黄=桌面, 品红=被测量的两条边
def draw_result(img, paper_quad, table_quad, L, Wd):
    out = img.copy()
    pq = np.array(paper_quad, dtype=np.int32).reshape(-1, 1, 2)
    tq = np.array(table_quad, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pq], True, (0, 255, 0), 3)
    cv2.polylines(out, [tq], True, (0, 255, 255), 3)
    for i, j in ((0, 1), (1, 2)):
        p = tuple(int(v) for v in tq[i][0])
        q = tuple(int(v) for v in tq[j][0])
        cv2.line(out, p, q, (255, 0, 255), 3)
    h = out.shape[0]
    cv2.putText(out, "L = %.1f mm" % L, (15, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 3)
    cv2.putText(out, "W = %.1f mm" % Wd, (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 3)
    return out


# 俯视校正图: 按平面坐标重采样, 画 100mm 网格, 用于目视核对纸是不是矩形
def rectified_view(img, H, table_quad, px_per_mm=0.6, pad_mm=30.0):
    q = np.array([geom.to_plane(H, p) for p in table_quad])
    x0, y0 = q.min(axis=0)
    x1, y1 = q.max(axis=0)
    w = (x1 - x0) + 2 * pad_mm
    h = (y1 - y0) + 2 * pad_mm
    M = np.array([[1.0, 0.0, -x0 + pad_mm],
                  [0.0, 1.0, -y0 + pad_mm],
                  [0.0, 0.0, 1.0]]) @ H
    size = (int(round(w * px_per_mm)), int(round(h * px_per_mm)))
    out = cv2.warpPerspective(img, M, size, flags=cv2.INTER_AREA)
    step = int(round(100 * px_per_mm))
    for x in range(0, size[0], step):
        cv2.line(out, (x, 0), (x, size[1]), (200, 200, 200), 1)
        cv2.putText(out, "%d" % int(round(x / px_per_mm + x0)), (x + 3, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for y in range(0, size[1], step):
        cv2.line(out, (0, y), (size[0], y), (200, 200, 200), 1)
    return out


# 单张图的完整流程: 取点 -> 解 H -> 量桌面 -> 出图 -> 返回一行结果
def process_one(path, args, picks_db):
    name = os.path.splitext(os.path.basename(path))[0]
    img = load_image(path)
    scale = display_scale(img)

    if name in picks_db and not args.reclick:
        paper_px = picks_db[name]["paper"]
        table_px = picks_db[name]["table"]
        print("[%s] 用缓存的点选坐标" % name)
    else:
        print("[%s] 请点 A4 四角" % name)
        paper_px = QuadPicker(img, name + " - PAPER (TL,TR,BR,BL)", scale).run()
        if len(paper_px) != 4:
            print("  跳过: 没点满 4 点")
            return None
        print("[%s] 请点桌面四角" % name)
        table_px = QuadPicker(img, name + " - TABLE (TL,TR,BR,BL)", scale).run()
        if len(table_px) != 4:
            print("  跳过: 没点满 4 点")
            return None
        picks_db[name] = {"paper": paper_px, "table": table_px}

    paper_q = geom.order_quad(paper_px)
    table_q = geom.order_quad(table_px)
    H = geom.homography(paper_q, geom.PAPER_PLANE)
    props = geom.quad_props(H, table_q)

    # 四点退化(有两个角重合、或塌成一条线)时结果没有意义, 明确提示重来
    tq = np.asarray(table_q, dtype=float)
    dmin = min(np.linalg.norm(tq[i] - tq[j])
               for i in range(4) for j in range(i + 1, 4))
    if props["area_mm2"] < 1000 or dmin < 5:
        print("  !! 这张的四个角有问题(有两角几乎重合或共线), 数值不可信; 用 --reclick 重拍这张")
    os.makedirs(args.out, exist_ok=True)
    imwrite_unicode(os.path.join(args.out, name + "_annotated.jpg"),
                    draw_result(img, paper_q, table_q,
                                props["long_side"], props["short_side"]))
    imwrite_unicode(os.path.join(args.out, name + "_rectified.jpg"),
                    rectified_view(img, H, table_q, args.px_per_mm))

    print("  -> L = %.1f mm, W = %.1f mm"
          % (props["long_side"], props["short_side"]))
    return {
        "编号": name,
        "file": os.path.basename(path),
        "长(mm)": round(props["long_side"], 1),
        "宽(mm)": round(props["short_side"], 1),
        "对角线(mm)": round(props["diagonals"][0], 1),
        "面积(cm2)": round(props["area_mm2"] / 100.0, 1),
        "矩形度残差(mm)": round(props["rect_error_mm"], 1),
    }


# 合成数据自检: 用已知位姿造像素, 再只用纸的 4 个角反解, 看能不能还原桌面
# 不需要照片, 也不需要 opencv; 数字全是模拟的, 只为验证代码
def selftest():
    F, CX, CY = 3200.0, 2000.0, 1500.0
    a = np.deg2rad(55.0)
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, np.cos(a), -np.sin(a)],
                  [0.0, np.sin(a), np.cos(a)]])
    T = np.array([-600.0, -400.0, 1150.0])
    K = np.array([[F, 0.0, CX], [0.0, F, CY], [0.0, 0.0, 1.0]])

    def project(pts):
        out = []
        for X, Y in pts:
            uvw = K @ (R @ np.array([X, Y, 0.0]) + T)
            out.append(uvw[:2] / uvw[2])
        return out

    paper_px = project(geom.PAPER_PLANE.tolist())
    table_px = project([[0, 0], [1200, 0], [1200, 600], [0, 600]])

    H = geom.homography(paper_px, geom.PAPER_PLANE)
    props = geom.quad_props(H, table_px)
    pred, err = geom.check_point(H, project([[148.5, 105.0]])[0], (148.5, 105.0))

    print("=" * 62)
    print("[自检] 下面的数字全部来自程序内部的模拟相机, 不是你的桌子")
    print("[自检] 不需要照片, 也不需要你量任何东西; 它只检验代码对不对")
    print("=" * 62)
    print("合成数据自检 (4000x3000, F=3200, 俯角55度)")
    print("  纸四角像素: %s" % [tuple(round(float(v), 3) for v in p) for p in paper_px])
    print("  量得桌面长: %.4f mm  (设定值 1200)" % props["long_side"])
    print("  量得桌面宽: %.4f mm  (设定值  600)" % props["short_side"])
    print("  纸中心验证点: (%.3f, %.3f), 误差 %.6f mm"
          % (pred[0], pred[1], err))

    ok = (abs(props["long_side"] - 1200.0) < 1e-3
          and abs(props["short_side"] - 600.0) < 1e-3
          and err < 1e-3)
    print("  结果: %s" % ("通过 (数值误差 < 0.001 mm)" if ok else "不通过, 代码有问题"))
    print("[自检] 通过只说明公式和代码没问题; 你的桌子准不准, 要看你怎么拍照和点角点")
    return ok


def main():
    geom.fix_console()
    ap = argparse.ArgumentParser(description="用 A4 纸测桌面长宽(免标定)")
    ap.add_argument("--img", help="单张图片")
    ap.add_argument("--batch", help="通配符, 如 \"data/*.jpg\"")
    ap.add_argument("--out", default="data/results", help="输出目录")
    ap.add_argument("--px-per-mm", type=float, default=0.6, help="俯视图分辨率")
    ap.add_argument("--reclick", action="store_true", help="忽略缓存重新点选")
    ap.add_argument("--selftest", action="store_true", help="跑合成数据自检, 不需要照片")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if cv2 is None:
        print("没装 opencv, 先执行:  pip install opencv-python")
        return

    if not args.img and not args.batch:
        ap.print_help()
        return

    files = ([args.img] if args.img else []) + \
            (sorted(glob.glob(args.batch)) if args.batch else [])
    if not files:
        print("没找到图片, 检查路径或通配符(Windows 下要加引号)")
        return
    os.makedirs(args.out, exist_ok=True)

    picks_path = os.path.join(args.out, "picks.json")
    picks_db = {}
    if os.path.exists(picks_path):
        with open(picks_path, "r", encoding="utf-8") as f:
            picks_db = json.load(f)

    csv_path = os.path.join(args.out, "result.csv")

    # 先把已有的结果读进来: 这样只重拍某一张时, 不会把其他组冲掉
    all_rows = {}
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    key = r.get("编号") or os.path.splitext(r.get("file", ""))[0]
                    if key:
                        all_rows[key] = r
        except OSError:
            print("提示: 读不了已有的 result.csv, 这次只写新算的组")

    warn_picks = False
    for path in files:
        try:
            row = process_one(path, args, picks_db)
            if row:
                all_rows[row["编号"]] = row
            try:
                with open(picks_path, "w", encoding="utf-8") as f:
                    json.dump(picks_db, f, ensure_ascii=False, indent=1)
            except OSError:
                if not warn_picks:
                    print("  !! 写不了 picks.json(被占用?), 本次点选不会缓存")
                    warn_picks = True
        except Exception as e:
            print("  [失败] %s : %s" % (path, e))

    if not all_rows:
        return
    rows = [all_rows[k] for k in sorted(all_rows)]

    # utf-8-sig 让 Excel 打开不乱码
    try:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\n共 %d 组, 结果写入 %s" % (len(rows), csv_path))
    except OSError:
        print("\n共 %d 组已算完, 但写不进 %s" % (len(rows), csv_path))
        print("   多半是 Excel 正开着它。关掉 Excel 后再跑一次同样的命令即可(点选坐标已缓存, 不用重点)")

    lens = np.array([float(r["长(mm)"]) for r in rows], dtype=float)
    wid = np.array([float(r["宽(mm)"]) for r in rows], dtype=float)
    dd = 1 if len(rows) > 1 else 0
    print("长: 均值 %.1f, 标准差 %.1f mm" % (lens.mean(), lens.std(ddof=dd)))
    print("宽: 均值 %.1f, 标准差 %.1f mm" % (wid.mean(), wid.std(ddof=dd)))


if __name__ == "__main__":
    main()
