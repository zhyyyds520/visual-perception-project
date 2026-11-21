# analyze.py -- 误差统计: 拿 result.csv + 你量的真值, 算误差、分组、出图
#
# 用法: python code/analyze.py
# 前置: measure_table.py 已经跑完, data/results/result.csv 存在
# 输出: 终端打印 + data/results/stats/*.csv + data/results/figures/三张图 (+ 装了 matplotlib 再存 3 张 png)

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom

# ==================== 只改这两行 ====================
TRUE_L = 600.0      # ← 填 X: 你量到的桌面长(mm), 例如 1200.0
TRUE_W = 400.0      # ← 填 Y: 你量到的桌面宽(mm), 例如 600.0
# ===================================================

RESULT_CSV = "data/results/result.csv"
PICKS_JSON = "data/results/picks.json"
META_CSV = "data/meta.csv"       # 可选: 编号,距离,俯角,纸位置; 没有就不分组
OUT_DIR = "data/results/stats"
SIGMA_PX = 1.5                   # 假设点选误差 1.5 像素, 用于蒙特卡洛预测


def line_of(name):
    """找到某个配置在本文件里的行号, 报错时告诉用户改第几行"""
    src = open(os.path.abspath(__file__), encoding="utf-8").read().splitlines()
    for i, s in enumerate(src, 1):
        if s.strip().startswith(name + " "):
            return i
    return 0


def load_results():
    if not os.path.exists(RESULT_CSV):
        print("找不到 %s, 先把 measure_table.py 跑一遍" % RESULT_CSV)
        return []
    rows = []
    with open(RESULT_CSV, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "编号": r.get("编号") or os.path.splitext(r["file"])[0],
                    "file": r["file"],
                    "长": float(r["长(mm)"]),
                    "宽": float(r["宽(mm)"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def load_meta():
    if not os.path.exists(META_CSV):
        return {}
    meta = {}
    with open(META_CSV, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r.get("编号") or "").strip()
            if key:
                meta[key] = (r.get("距离", ""), r.get("俯角", ""))
    return meta


def stat_block(meas, truth):
    meas = np.asarray(meas, dtype=float)
    err = meas - truth
    rel = err / truth * 100.0
    return {
        "样本数": len(meas),
        "均值(mm)": meas.mean(),
        "标准差(mm)": meas.std(ddof=1) if len(meas) > 1 else 0.0,
        "平均绝对误差(mm)": np.abs(err).mean(),
        "RMSE(mm)": float(np.sqrt((err ** 2).mean())),
        "平均相对误差(%)": rel.mean(),
        "平均绝对相对误差(%)": np.abs(rel).mean(),
        "最大相对误差(%)": np.abs(rel).max(),
        "95%区间(±mm)": 1.96 * (meas.std(ddof=1) if len(meas) > 1 else 0.0),
    }


def group_table(detail, key, truth_l, truth_w):
    groups = {}
    for d in detail:
        g = d.get(key) or "未知"
        groups.setdefault(g, []).append(d)
    out = []
    for g in sorted(groups):
        items = groups[g]
        rel_l = np.mean([abs(i["长"] - truth_l) / truth_l * 100 for i in items])
        rel_w = np.mean([abs(i["宽"] - truth_w) / truth_w * 100 for i in items])
        out.append([g, len(items), round(rel_l, 2), round(rel_w, 2)])
    return out


def mc_predict():
    """蒙特卡洛: 只考虑点选噪声时, 长宽应该有多的相对离散度"""
    if not os.path.exists(PICKS_JSON):
        return None
    with open(PICKS_JSON, "r", encoding="utf-8") as f:
        db = json.load(f)
    rels = []
    for name in sorted(db):
        # 必须先排序: 原始点选顺序不一定首尾相接, 直接取 (0,1)/(1,2) 可能取到的不是边
        paper = geom.order_quad(db[name]["paper"])
        table = geom.order_quad(db[name]["table"])
        try:
            for i, j in ((0, 1), (1, 2)):
                r = geom.mc_sigma(paper, geom.PAPER_PLANE, table[i], table[j],
                                  sigma_px=SIGMA_PX, trials=400)
                if r["mean"] > 0:
                    rels.append(r["std"] / r["mean"] * 100.0)
        except Exception:
            continue
    return float(np.mean(rels)) if rels else None


def save_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def make_plots(detail, truth_l, truth_w):
    """长图: (a)逐组误差折线 / (b)误差结构 / (c)累计误差分布, 上中下三段"""
    try:
        import math
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import MultipleLocator
        plt.rcParams.update({
            "font.sans-serif": ["Microsoft YaHei", "SimHei"],
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.linewidth": 0.9,
        })
    except Exception:
        print("没装 matplotlib, 跳过画图")
        return

    fig_dir = "data/results/figures"
    os.makedirs(fig_dir, exist_ok=True)

    names = [d["编号"] for d in detail]
    L = np.array([d["长"] for d in detail], dtype=float)
    W = np.array([d["宽"] for d in detail], dtype=float)
    rl = (L - truth_l) / truth_l * 100.0
    rw = (W - truth_w) / truth_w * 100.0
    C_L, C_W = "#2F5D8A", "#B4553F"

    def tidy(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#777")
        ax.tick_params(direction="out", length=2.5, color="#777", labelsize=7)

    def phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    fig = plt.figure(figsize=(7.0, 10.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.85, 1.0])
    fig.subplots_adjust(left=0.105, right=0.975, top=0.972, bottom=0.052,
                        hspace=0.40)

    # ---------------- (a) 逐组误差折线
    ax = fig.add_subplot(gs[0, 0])
    idx = np.arange(len(names))
    ax.plot(idx, rl, "-o", color=C_L, lw=1.3, ms=4.2, mfc="w", mew=1.2)
    ax.plot(idx, rw, "-s", color=C_W, lw=1.3, ms=3.8, mfc="w", mew=1.2)
    ax.axhline(0, color="#555", lw=0.8, zorder=1)
    for t, ls in ((1.0, ":"), (3.0, "--")):
        for s in (1, -1):
            ax.axhline(s * t, color="#AAAAAA", lw=0.7, ls=ls, zorder=0)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, fontsize=6.2)
    ax.set_xlim(-0.8, len(idx) - 0.2)
    ax.set_xlabel("组号", fontsize=8, labelpad=2)
    ax.set_ylabel("相对误差 (%)", fontsize=8.5)
    ax.grid(axis="y", color="#EDEDED", lw=0.5)
    handles = [Line2D([], [], marker="o", ls="-", color=C_L, ms=4, label="长"),
               Line2D([], [], marker="s", ls="-", color=C_W, ms=3.8, label="宽"),
               Line2D([], [], color="#AAAAAA", ls=":", lw=1.1, label="±1%"),
               Line2D([], [], color="#AAAAAA", ls="--", lw=1.1, label="±3%")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.03),
              ncol=4, frameon=False, fontsize=7.2, handletextpad=0.35,
              columnspacing=1.4)
    ax.set_title("(a) 各组测量相对误差", fontsize=9.5, loc="left", pad=6)
    tidy(ax)

    # ---------------- (b) 误差结构
    ax = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(0)
    for k, (tag, r, c) in enumerate((("长", rl, C_L), ("宽", rw, C_W))):
        y0 = 1 - k
        jit = rng.uniform(-0.13, 0.13, len(r))
        ax.scatter(r, y0 + jit, s=13, color=c, alpha=0.72,
                   edgecolors="w", linewidths=0.35, zorder=3)
        m, s = float(r.mean()), float(r.std(ddof=1))
        ax.plot([m, m], [y0 - 0.20, y0 + 0.20], color=c, lw=2.0, zorder=4)
        ax.errorbar([m], [y0], xerr=[[1.96 * s], [1.96 * s]], fmt="none",
                    ecolor=c, elinewidth=1.0, capsize=2.5, zorder=4)
        ax.text(0.015, y0 + 0.26,
                "%s  偏差 %+.2f%%　σ = %.2f%%　|偏差|/σ = %.2f"
                % (tag, m, s, abs(m) / s), transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=7, color=c)
    ax.axvline(0, color="#555", lw=0.8, zorder=1)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["长", "宽"], fontsize=9)
    ax.set_ylim(-0.42, 1.62)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel("相对误差 (%)", fontsize=8, labelpad=2)
    ax.grid(axis="x", color="#EDEDED", lw=0.5, zorder=0)
    ax.set_title("(b) 误差结构：系统偏差与随机离散", fontsize=9.5, loc="left",
                 pad=6)
    tidy(ax)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="x", which="minor", length=1.5, color="#999")

    # ---------------- (c) 累计误差分布(实测点 + 拟合)
    ax = fig.add_subplot(gs[2, 0])
    handles = []
    xmax = max(float(np.abs(rl).max()), float(np.abs(rw).max())) * 1.02
    box = []
    for rel, c, mk, tag in ((rl, C_L, "o", "长"), (rw, C_W, "s", "宽")):
        a = np.sort(np.abs(rel))
        n = len(a)
        ax.plot(a, np.arange(1, n + 1) / float(n), mk, ms=3.2, color=c,
                alpha=0.55, ls="none", zorder=3)
        mu, sd = float(rel.mean()), float(rel.std(ddof=1))
        xs = np.linspace(0, xmax, 300)
        ys = [phi((x - mu) / sd) - phi((-x - mu) / sd) for x in xs]
        ax.plot(xs, ys, "-", color=c, lw=1.7, zorder=4)
        box.append("%s  中位 %.1f%%　90%% 分位 %.1f%%　最大 %.1f%%"
                   % (tag, float(np.percentile(a, 50)),
                      float(np.percentile(a, 90)), float(a.max())))
        handles.append(Line2D([], [], marker=mk, ls="none", color=c, ms=4,
                              label="%s 实测" % tag))
        handles.append(Line2D([], [], ls="-", color=c, lw=1.6,
                              label="%s 拟合" % tag))
    for t in (3, 5):
        ax.axvline(t, color="#AAAAAA", lw=0.7, ls="--", zorder=0)
        ax.text(t, 0.02, " %d%%" % t, fontsize=6.5, color="#888")
    ax.text(0.03, 0.97, "\n".join(box), transform=ax.transAxes, va="top",
            ha="left", fontsize=7, color="#333", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDD", lw=0.5))
    ax.set_xlabel("相对误差绝对值 (%)", fontsize=8, labelpad=2)
    ax.set_ylabel("累计比例", fontsize=8.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.grid(color="#EDEDED", lw=0.5)
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.8,
              ncol=2, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.3)
    ax.set_title("(c) 累计误差分布", fontsize=9.5, loc="left", pad=6)
    tidy(ax)

    out = os.path.join(fig_dir, "图_误差分析长图.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("长图已存到 %s" % out)


def main():
    geom.fix_console()

    ap = argparse.ArgumentParser(description="误差统计")
    ap.add_argument("--group", action="store_true",
                    help="按 data/meta.csv 的距离/俯角分组(要先把 meta.csv 改成你实际的拍摄条件)")
    args = ap.parse_args()

    if TRUE_L <= 0 or TRUE_W <= 0:
        print("还没填真值。打开 code/analyze.py:")
        print("  第 %d 行  TRUE_L = ...   填你量的桌面长(mm)" % line_of("TRUE_L"))
        print("  第 %d 行  TRUE_W = ...   填你量的桌面宽(mm)" % line_of("TRUE_W"))
        return

    rows = load_results()
    if not rows:
        return

    meta = load_meta() if args.group else {}
    if args.group and not meta:
        print("data/meta.csv 里没有可用的距离/俯角, 跳过分组")
    detail = []
    for r in rows:
        m = meta.get(r["编号"], ("", ""))
        detail.append({"编号": r["编号"], "距离": m[0], "俯角": m[1],
                       "长": r["长"], "宽": r["宽"]})

    sl = stat_block([d["长"] for d in detail], TRUE_L)
    sw = stat_block([d["宽"] for d in detail], TRUE_W)

    print("=" * 58)
    print("真值: 长 %.1f mm, 宽 %.1f mm; 样本 %d 组" % (TRUE_L, TRUE_W, len(detail)))
    print("-" * 58)
    print("%-16s %12s %12s" % ("指标", "长", "宽"))
    for k in sl:
        if k == "样本数":
            print("%-16s %12d %12d" % (k, sl[k], sw[k]))
        else:
            print("%-16s %12.2f %12.2f" % (k, sl[k], sw[k]))
    print("-" * 58)

    pred = mc_predict()
    if pred is not None:
        print("蒙特卡洛预测: 只算点选误差(%.1f 像素)时, 相对标准差约 %.2f%%"
              % (SIGMA_PX, pred))
        print("实测相对标准差: 长 %.2f%%, 宽 %.2f%%"
              % (sl["标准差(mm)"] / TRUE_L * 100, sw["标准差(mm)"] / TRUE_W * 100))
        print("两者接近 -> 点选是主因; 实测大很多 -> 还有系统误差(畸变/纸不平)")
    print("=" * 58)

    os.makedirs(OUT_DIR, exist_ok=True)
    save_csv(os.path.join(OUT_DIR, "detail.csv"),
             ["编号", "距离", "俯角", "测长", "测宽", "误差长", "误差宽", "相对长%", "相对宽%"],
             [[d["编号"], d["距离"], d["俯角"],
               round(d["长"], 1), round(d["宽"], 1),
               round(d["长"] - TRUE_L, 1), round(d["宽"] - TRUE_W, 1),
               round((d["长"] - TRUE_L) / TRUE_L * 100, 2),
               round((d["宽"] - TRUE_W) / TRUE_W * 100, 2)] for d in detail])
    save_csv(os.path.join(OUT_DIR, "summary.csv"),
             ["指标", "长", "宽"],
             [[k, round(sl[k], 3), round(sw[k], 3)] for k in sl if k != "样本数"])

    for key, fname in (("距离", "by_distance.csv"), ("俯角", "by_angle.csv")):
        table = group_table(detail, key, TRUE_L, TRUE_W)
        if args.group and len(table) > 1:
            save_csv(os.path.join(OUT_DIR, fname),
                     [key, "组数", "长平均相对误差%", "宽平均相对误差%"], table)
            print("\n按%s分组:" % key)
            for row in table:
                print("  %-6s %3d 组  长 %5.2f%%  宽 %5.2f%%"
                      % (row[0], row[1], row[2], row[3]))
    if not args.group:
        print("\n(没分组。要按距离/俯角分组: 先把 data/meta.csv 填成你实际的拍摄条件, "
              "再加 --group 重跑)")

    make_plots(detail, TRUE_L, TRUE_W)
    print("\n结果表已写入 %s/" % OUT_DIR)


if __name__ == "__main__":
    main()
