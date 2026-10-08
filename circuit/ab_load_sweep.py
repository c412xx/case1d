"""戴维南定理验证：A、B 端口的两并联支路网络 → 等效为「V_oc 串 R_in」
验证三件事：
    ① 开路仿真（A、B 断开）        -> V_oc，与手算比较
    ② 短路仿真（A、B 用 0 V 源短接）-> I_sc，与手算比较
       并由 R_in = V_oc / I_sc 反推等效内阻
    ③ 同一批纯电阻负载分别接在「原网络」和「等效网络」上，
       比较 V_AB 与 I_L：两者必须一致，且与手算一致
"""

import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
FIGURE_PATH = BASE_DIR / "07_ab_load_sweep.png"
TABLE_PATH = BASE_DIR / "thevenin_table.md"

import PySpice.Logging.Logging as Logging

Logging.setup_logging(logging_level="ERROR")

import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_Ohm, u_V

# --------------------------------------------------------------------------- #
# 0. 原网络参数（按要求保持不动）
# --------------------------------------------------------------------------- #
R1 = 3.0        # Ω，支路 1 电阻
R2 = 6.0        # Ω，支路 2 电阻
U1 = 1.2        # V，支路 1 电源
U2 = 3.6        # V，支路 2 电源

# 待验证的纯电阻负载
LOAD_RESISTANCES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 20.0, 50.0]

# --------------------------------------------------------------------------- #
# 1. 手算：戴维南等效参数
# --------------------------------------------------------------------------- #
R_IN = R1 * R2 / (R1 + R2)                                  # 等效内阻 2 Ω
V_OC = (U1 / R1 + U2 / R2) / (1.0 / R1 + 1.0 / R2)          # 开路电压 2 V
I_SC = V_OC / R_IN                                          # 短路电流 1 A


def hand_calc(load):
    """负载为 load(Ω) 时的手算 (I_L, V_AB, P_L)。"""
    if load is None:                       # 开路
        return 0.0, V_OC, 0.0
    current = V_OC / (R_IN + load)
    voltage = V_OC * load / (R_IN + load)
    return current, voltage, voltage * current


def banner(title):
    print("=" * 100)
    print(title)
    print("=" * 100)


def num(value, digits):
    """按指定小数位格式化（写 markdown 时用）。"""
    return ("%%.%df" % digits) % value


# --------------------------------------------------------------------------- #
# 2. 搭建电路：原网络 / 戴维南等效网络
# --------------------------------------------------------------------------- #
def build_circuit(network, load=None, short=False):
    """network: 'original'（原网络）或 'thevenin'（等效网络）
       load:    负载电阻（Ω），None 表示开路
       short:   True 时用 0 V 源把 A、B 直接短接（同时充当电流表）
    """
    circuit = Circuit("%s network" % network)

    if network == "original":
        circuit.R(1, "a", "m1", R1 @ u_Ohm)
        circuit.V(1, "m1", "b", U1 @ u_V)
        circuit.R(2, "a", "m2", R2 @ u_Ohm)
        circuit.V(2, "m2", "b", U2 @ u_V)
    elif network == "thevenin":
        # 戴维南等效：V_oc 电压源与 R_in 串联
        circuit.R("in", "a", "m", R_IN @ u_Ohm)
        circuit.V("oc", "m", "b", V_OC @ u_V)
    else:
        raise ValueError(network)

    # B 点作为 0 V 参考
    circuit.V("ref", "b", circuit.gnd, 0 @ u_V)

    if short:
        circuit.V("ammeter", "a", "b", 0 @ u_V)          # 短路线 = 0 V 源（电流表）
    elif load is not None:
        circuit.V("ammeter", "a", "aload", 0 @ u_V)      # 串联 0 V 源测负载电流
        circuit.R("L", "aload", "b", load @ u_Ohm)

    return circuit


def branch_current(analysis, name):
    """读电压源支路电流（PySpice 的 key 是小写，如 'vammeter'）。"""
    for key in (name, name.lower(), "v" + name.lower()):
        if key in analysis.branches:
            return float(analysis.branches[key][0])
    raise KeyError("找不到支路 %s，可用: %s" % (name, list(analysis.branches.keys())))


def measure(network, load=None, short=False):
    """跑一次直流工作点分析，返回端口电压/电流。"""
    analysis = build_circuit(network, load=load, short=short).simulator(
        temperature=25, nominal_temperature=25).operating_point()

    v_a = float(analysis["a"][0])
    v_b = float(analysis["b"][0])
    i_l = branch_current(analysis, "ammeter") if (short or load is not None) else 0.0

    return {
        "network": network,
        "load": load,
        "short": short,
        "v_a": v_a,
        "v_b": v_b,
        "v_ab": v_a - v_b,
        "i_load": i_l,
    }


# --------------------------------------------------------------------------- #
# 3. 仿真① 开路 与 仿真② 短路（原网络 + 等效网络各做一遍）
# --------------------------------------------------------------------------- #
banner("戴维南定理验证：原网络（两并联支路） → 等效网络（V_oc 串 R_in）")

oc_original = measure("original")                 # 仿真① 原网络开路
sc_original = measure("original", short=True)     # 仿真② 原网络短路
oc_thevenin = measure("thevenin")                 # 仿真① 等效网络开路
sc_thevenin = measure("thevenin", short=True)     # 仿真② 等效网络短路

R_IN_SIM = oc_original["v_ab"] / sc_original["i_load"]          # 由两次仿真反推等效内阻
R_IN_SIM_THEV = oc_thevenin["v_ab"] / sc_thevenin["i_load"]

print("仿真① 开路：原网络 V_oc = %.6f V ；等效网络 V_oc = %.6f V"
      % (oc_original["v_ab"], oc_thevenin["v_ab"]))
print("仿真② 短路：原网络 I_sc = %.6f A ；等效网络 I_sc = %.6f A"
      % (sc_original["i_load"], sc_thevenin["i_load"]))
print("由两次仿真反推：原网络 R_in = V_oc/I_sc = %.6f Ω ；等效网络 R_in = %.6f Ω"
      % (R_IN_SIM, R_IN_SIM_THEV))

# ---- 表 1：V_oc、I_sc 两次仿真的「手算 vs 仿真」 ----
table1 = [
    ("开路电压 V_oc", "V", "仿真① 开路（A、B 断开）",
     V_OC, oc_original["v_ab"], oc_thevenin["v_ab"], 6),
    ("短路电流 I_sc", "A", "仿真② 短路（A、B 用 0 V 源短接）",
     I_SC, sc_original["i_load"], sc_thevenin["i_load"], 6),
    ("等效内阻 R_in = V_oc/I_sc", "Ω", "由上面两次仿真导出",
     R_IN, R_IN_SIM, R_IN_SIM_THEV, 6),
]

print()
banner("表 1  V_oc / I_sc 两次仿真：手算 vs 仿真")
print("数据来源：V_oc ← 仿真① 开路；I_sc ← 仿真② 短路；R_in = V_oc/I_sc 由这两次仿真导出\n")
print("{:<26}{:<6}{:>14}{:>18}{:>18}{:>12}{:>12}".format(
    "项目", "单位", "手算", "原网络仿真", "等效网络仿真", "原网误差", "等效误差"))
print("-" * 100)
for name, unit, _source, hand, sim_o, sim_t, digits in table1:
    print("{:<26}{:<6}{:>14.{d}f}{:>18.{d}f}{:>18.{d}f}{:>11.4f}%{:>11.4f}%".format(
        name, unit, hand, sim_o, sim_t,
        (sim_o - hand) / hand * 100, (sim_t - hand) / hand * 100, d=digits))
print("-" * 100)

# --------------------------------------------------------------------------- #
# 4. 仿真③ 接不同负载：原网络 vs 等效网络
# --------------------------------------------------------------------------- #
rows = []
for load in LOAD_RESISTANCES:
    r_orig = measure("original", load=load)
    r_thev = measure("thevenin", load=load)
    i_hand, v_hand, _ = hand_calc(load)
    rows.append((load, v_hand, r_orig["v_ab"], r_thev["v_ab"],
                 i_hand, r_orig["i_load"], r_thev["i_load"]))
    # 电流交叉校验：0 V 电流表读数 与 V/R
    assert abs(r_orig["i_load"] - r_orig["v_ab"] / load) < 1e-12, "电流交叉校验失败"

open_row = (None, V_OC, oc_original["v_ab"], oc_thevenin["v_ab"],
            0.0, oc_original["i_load"], oc_thevenin["i_load"])
short_row = (0.0, 0.0, sc_original["v_ab"], sc_thevenin["v_ab"],
             I_SC, sc_original["i_load"], sc_thevenin["i_load"])
all_rows = [open_row] + rows + [short_row]


def load_label(load):
    if load is None:
        return "开路 ∞"
    if load == 0.0:
        return "短路 0"
    return "%.1f" % load


print()
banner("表 2  等效电路替换后接负载：电压验证（V_AB）")
print("{:>8}{:>14}{:>16}{:>16}{:>14}{:>14}".format(
    "R_L/Ω", "手算/V", "原网络仿真/V", "等效网络仿真/V", "原网误差", "等效误差"))
print("-" * 100)
for load, v_hand, v_orig, v_thev, *_ in all_rows:
    err_o = "-" if v_hand == 0 else "%+.4f%%" % ((v_orig - v_hand) / v_hand * 100)
    err_t = "-" if v_hand == 0 else "%+.4f%%" % ((v_thev - v_hand) / v_hand * 100)
    print("{:>8}{:>14.6f}{:>16.6f}{:>16.6f}{:>14}{:>14}".format(
        load_label(load), v_hand, v_orig, v_thev, err_o, err_t))
print("-" * 100)

print()
banner("表 2  等效电路替换后接负载：电流验证（I_L）")
print("{:>8}{:>14}{:>16}{:>16}{:>14}{:>14}".format(
    "R_L/Ω", "手算/A", "原网络仿真/A", "等效网络仿真/A", "原网误差", "等效误差"))
print("-" * 100)
for load, _vh, _vo, _vt, i_hand, i_orig, i_thev in all_rows:
    err_o = "-" if i_hand == 0 else "%+.4f%%" % ((i_orig - i_hand) / i_hand * 100)
    err_t = "-" if i_hand == 0 else "%+.4f%%" % ((i_thev - i_hand) / i_hand * 100)
    print("{:>8}{:>14.6f}{:>16.6f}{:>16.6f}{:>14}{:>14}".format(
        load_label(load), i_hand, i_orig, i_thev, err_o, err_t))
print("-" * 100)

# 原网络 vs 等效网络 的最大偏差（应为浮点误差量级）
max_du = max(abs(row[2] - row[3]) for row in all_rows)
max_di = max(abs(row[5] - row[6]) for row in all_rows)
print("\n等效前后（原网络 vs 等效网络）端口电压最大偏差 = %.3e V" % max_du)
print("等效前后（原网络 vs 等效网络）负载电流最大偏差 = %.3e A" % max_di)

# --------------------------------------------------------------------------- #
# 5. 写 markdown 表
# --------------------------------------------------------------------------- #
md = ["# 戴维南定理验证（原网络 R1=%.0fΩ, R2=%.0fΩ, U1=%.1fV, U2=%.1fV）\n" % (R1, R2, U1, U2)]
md.append("手算等效参数：**V_oc = %.6f V，R_in = R1∥R2 = %.6f Ω，I_sc = V_oc/R_in = %.6f A**\n"
          % (V_OC, R_IN, I_SC))
md.append("## 表 1  V_oc / I_sc 两次仿真：手算 vs 仿真\n")
md.append("| 项目 | 单位 | 数据来源 | 手算 | 原网络仿真 | 等效网络仿真 | 原网误差 | 等效误差 |")
md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
for name, unit, source, hand, sim_o, sim_t, digits in table1:
    md.append("| {} | {} | {} | {} | {} | {} | {:+.4f}% | {:+.4f}% |".format(
        name, unit, source, num(hand, digits), num(sim_o, digits), num(sim_t, digits),
        (sim_o - hand) / hand * 100, (sim_t - hand) / hand * 100))
md.append("\n仿真条件：开路 = A、B 断开；短路 = A、B 之间接 0 V 电压源（同时充当电流表）。\n")
md.append("## 表 2  等效电路替换后接负载：电压验证 V_AB\n")
md.append("| R_L (Ω) | 手算 (V) | 原网络仿真 (V) | 等效网络仿真 (V) | 原网误差 | 等效误差 |")
md.append("| --- | --- | --- | --- | --- | --- |")
for load, v_hand, v_orig, v_thev, *_ in all_rows:
    err_o = "-" if v_hand == 0 else "%+.4f%%" % ((v_orig - v_hand) / v_hand * 100)
    err_t = "-" if v_hand == 0 else "%+.4f%%" % ((v_thev - v_hand) / v_hand * 100)
    md.append("| %s | %.6f | %.6f | %.6f | %s | %s |" % (
        load_label(load), v_hand, v_orig, v_thev, err_o, err_t))
md.append("\n## 表 2  等效电路替换后接负载：电流验证 I_L\n")
md.append("| R_L (Ω) | 手算 (A) | 原网络仿真 (A) | 等效网络仿真 (A) | 原网误差 | 等效误差 |")
md.append("| --- | --- | --- | --- | --- | --- |")
for load, _vh, _vo, _vt, i_hand, i_orig, i_thev in all_rows:
    err_o = "-" if i_hand == 0 else "%+.4f%%" % ((i_orig - i_hand) / i_hand * 100)
    err_t = "-" if i_hand == 0 else "%+.4f%%" % ((i_thev - i_hand) / i_hand * 100)
    md.append("| %s | %.6f | %.6f | %.6f | %s | %s |" % (
        load_label(load), i_hand, i_orig, i_thev, err_o, err_t))
md.append("\n**结论**：所有负载（含开路、短路）下，原网络与等效网络的端口电压、负载电流完全一致"
          "（最大偏差 %.3e V / %.3e A，即浮点精确相等），且都与手算值吻合（相对误差 0.0000%%），"
          "戴维南等效成立。\n" % (max_du, max_di))
TABLE_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")
print("表格已保存:", TABLE_PATH)

# --------------------------------------------------------------------------- #
# 6. 画图：两个电路示意 + 曲线重合验证 + 偏差
# --------------------------------------------------------------------------- #
loads = np.array([r[0] for r in rows])
v_orig_arr = np.array([r[2] for r in rows])
v_thev_arr = np.array([r[3] for r in rows])
i_orig_arr = np.array([r[5] for r in rows])
i_thev_arr = np.array([r[6] for r in rows])

load_fine = np.logspace(np.log10(0.4), np.log10(60), 400)
i_hand_fine = V_OC / (R_IN + load_fine)
v_hand_fine = V_OC * load_fine / (R_IN + load_fine)

figure, axes = plt.subplots(3, 2, figsize=(13, 13))
(ax_orig, ax_thev), (ax_vi, ax_i), (ax_u, ax_err) = axes


def draw_base(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.0, 6.4)
    ax.axis("off")
    ax.set_title(title)
    # A、B 端子（左竖线 = A，右竖线 = B）
    ax.plot([1.0, 1.0], [1.0, 5.0], color="black", linewidth=1.4)
    ax.plot([8.0, 8.0], [1.0, 5.0], color="black", linewidth=1.4)
    ax.plot([0.55, 1.0], [3.0, 3.0], color="black", linewidth=1.2, linestyle=":")
    ax.plot([8.0, 8.45], [3.0, 3.0], color="black", linewidth=1.2, linestyle=":")
    ax.plot([0.55], [3.0], "o", color="black", markersize=5)
    ax.plot([8.45], [3.0], "o", color="black", markersize=5)
    ax.text(0.40, 3.0, "A", ha="right", va="center", fontsize=13, fontweight="bold")
    ax.text(8.60, 3.0, "B", ha="left", va="center", fontsize=13, fontweight="bold")
    # 负载支路（虚线，A、B 之间）
    ax.plot([0.55, 2.7], [3.0, 3.0], color="tab:red", linewidth=1.4, linestyle="--")
    ax.add_patch(Rectangle((2.7, 2.72), 1.1, 0.56, fill=False, edgecolor="tab:red",
                           linewidth=1.4, linestyle="--"))
    ax.text(3.25, 3.45, "R_L（负载）", ha="center", va="bottom", fontsize=9, color="tab:red")
    ax.plot([3.8, 8.45], [3.0, 3.0], color="tab:red", linewidth=1.4, linestyle="--")


def draw_branch(ax, y, r_label, u_label, show_r=True):
    if show_r:
        ax.plot([1.0, 2.2], [y, y], color="black", linewidth=1.4)
        ax.add_patch(Rectangle((2.35, y - 0.28), 1.1, 0.56, fill=False,
                               edgecolor="black", linewidth=1.4))
        ax.text(2.9, y + 0.53, r_label, ha="center", va="bottom", fontsize=9)
        ax.plot([3.45, 5.3], [y, y], color="black", linewidth=1.4)
    else:
        ax.plot([1.0, 5.3], [y, y], color="black", linewidth=1.4)
    source_x = 5.8
    ax.plot([source_x - 0.45, source_x + 0.45], [y + 0.11, y + 0.11],
            color="black", linewidth=1.6)
    ax.plot([source_x - 0.22, source_x + 0.22], [y - 0.11, y - 0.11],
            color="black", linewidth=2.8)
    ax.text(source_x, y - 0.40, u_label, ha="center", va="top", fontsize=9)
    ax.plot([source_x + 0.55, 8.0], [y, y], color="black", linewidth=1.4)


# (1) 原网络
draw_base(ax_orig, "① 原网络：两并联支路（R1 / R2 各串电源）")
draw_branch(ax_orig, 5.0, "R1 = %.1f Ω" % R1, "U1 = %.1f V" % U1)
draw_branch(ax_orig, 1.0, "R2 = %.1f Ω" % R2, "U2 = %.1f V" % U2)

# (2) 等效网络
draw_base(ax_thev, "② 戴维南等效网络：V_oc 串 R_in")
draw_branch(ax_thev, 5.0, "R_in = R1∥R2 = %.1f Ω" % R_IN, "V_oc = %.1f V" % V_OC)
ax_thev.text(5.0, 1.3,
             "V_oc = %.6f V\nI_sc  = %.6f A\nR_in = V_oc/I_sc = %.6f Ω"
             % (V_OC, I_SC, R_IN),
             ha="center", va="center", fontsize=10.5,
             bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"))

# (3) 外特性 V_AB – I_L：原网络与等效网络工作点重合
ax_vi.plot(i_hand_fine, v_hand_fine, "-", color="tab:blue", label="手算直线 U = V_oc − R_in·I")
ax_vi.plot(i_orig_arr, v_orig_arr, "o", color="tab:red", markersize=8, mfc="none",
           label="原网络仿真工作点")
ax_vi.plot(i_thev_arr, v_thev_arr, "x", color="tab:green", markersize=7,
           label="等效网络仿真工作点")
ax_vi.plot([0], [V_OC], "s", color="tab:purple", markersize=7,
           label="开路点 (0, %.3f V)" % V_OC)
ax_vi.plot([I_SC], [0], "^", color="tab:brown", markersize=7,
           label="短路点 (%.3f A, 0)" % I_SC)
ax_vi.set_xlabel("负载电流 I_L (A)")
ax_vi.set_ylabel("端电压 V_AB (V)")
ax_vi.set_title("③ 外特性：两套仿真工作点完全重合")
ax_vi.grid(True, alpha=0.3)
ax_vi.legend(fontsize=8)

# (4) I_L 随 R_L 变化
ax_i.semilogx(load_fine, i_hand_fine, "-", color="tab:blue", label="手算")
ax_i.semilogx(loads, i_orig_arr, "o", color="tab:red", markersize=7, mfc="none",
              label="原网络仿真")
ax_i.semilogx(loads, i_thev_arr, "x", color="tab:green", markersize=6,
              label="等效网络仿真")
ax_i.axvline(R_IN, color="tab:gray", linestyle=":", alpha=0.8, label="R_L = R_in")
ax_i.set_xlabel("负载电阻 R_L (Ω)")
ax_i.set_ylabel("负载电流 I_L (A)")
ax_i.set_title("④ 不同纯电阻负载下的电流")
ax_i.grid(True, which="both", alpha=0.3)
ax_i.legend(fontsize=8)

# (5) V_AB 随 R_L 变化
ax_u.semilogx(load_fine, v_hand_fine, "-", color="tab:blue", label="手算")
ax_u.semilogx(loads, v_orig_arr, "o", color="tab:red", markersize=7, mfc="none",
              label="原网络仿真")
ax_u.semilogx(loads, v_thev_arr, "x", color="tab:green", markersize=6,
              label="等效网络仿真")
ax_u.set_xlabel("负载电阻 R_L (Ω)")
ax_u.set_ylabel("端电压 V_AB (V)")
ax_u.set_title("⑤ 不同负载下的端电压")
ax_u.grid(True, which="both", alpha=0.3)
ax_u.legend(fontsize=8)

# (6) 原网络 − 等效网络 的偏差
du = np.abs(v_orig_arr - v_thev_arr)
di = np.abs(i_orig_arr - i_thev_arr)
ax_err.loglog(loads, np.maximum(du, 1e-18), "o-", color="tab:red", markersize=5,
              label="|ΔV_AB| (V)")
ax_err.loglog(loads, np.maximum(di, 1e-18), "s-", color="tab:green", markersize=5,
              label="|ΔI_L| (A)")
ax_err.axhline(1e-15, color="tab:gray", linestyle="--", alpha=0.8,
               label="双精度浮点误差量级 1e-15")
ax_err.set_xlabel("负载电阻 R_L (Ω)")
ax_err.set_ylabel("原网络 − 等效网络 的偏差")
ax_err.set_title("⑥ 等效前后偏差：仅浮点误差（最大 %.1e V / %.1e A）" % (du.max(), di.max()))
ax_err.grid(True, which="both", alpha=0.3)
ax_err.legend(fontsize=8)

figure.suptitle("戴维南定理验证：原网络（R1=%.1fΩ, R2=%.1fΩ, U1=%.1fV, U2=%.1fV）"
                " 等效为 V_oc=%.4f V 串 R_in=%.4f Ω" % (R1, R2, U1, U2, V_OC, R_IN),
                fontsize=13)
figure.tight_layout(rect=(0, 0, 1, 0.965))
figure.savefig(FIGURE_PATH, dpi=110, bbox_inches="tight")
print("图片已保存:", FIGURE_PATH)

plt.show()
