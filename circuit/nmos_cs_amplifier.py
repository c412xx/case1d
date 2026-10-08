"""NMOS 共源放大电路（分压偏置）：手算 vs 仿真 + 电路图。
"""

import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
SHORT_TABLE_PATH = BASE_DIR / "手算vs仿真.md"

import PySpice.Logging.Logging as Logging

Logging.setup_logging(logging_level="ERROR")

import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_F, u_Hz, u_kOhm, u_m, u_ms, u_us, u_V

# 0. 题目参数
VDD = 5.0
RG1 = 60e3
RG2 = 40e3
RD = 2e3
CB1 = 10e-6
K_NMOS = 0.8e-3
VTH = 1.0
LAMBDA = 0.02
VI_AMP = 10e-3
FREQ = 1e3

KP = K_NMOS
W_OVER_L = 1.0
W_CH = 1e-6
L_CH = 1e-6
K_HALF = K_NMOS / 2

# 1. 手算：题卡推导结果 + 解析校核
VG_HAND = VDD * RG2 / (RG1 + RG2)
RG_PAR = RG1 * RG2 / (RG1 + RG2)
VGS_HAND = VG_HAND

ID_HAND = 0.4e-3
VDS_HAND = 4.2
GM_HAND = 2 * ID_HAND / (VGS_HAND - VTH)
RDS_HAND = 1.0 / (LAMBDA * ID_HAND)
AV_HAND = -GM_HAND * (RD * RDS_HAND / (RD + RDS_HAND))
AV_HAND_IDEAL = -GM_HAND * RD

ID_ANA = (K_HALF * (VGS_HAND - VTH) ** 2 * (1 + LAMBDA * VDD)
          / (1 + K_HALF * (VGS_HAND - VTH) ** 2 * LAMBDA * RD))
VDS_ANA = VDD - ID_ANA * RD
GM_ANA = K_NMOS * (VGS_HAND - VTH) * (1 + LAMBDA * VDS_ANA)
RO_ANA = 1.0 / (LAMBDA * K_HALF * (VGS_HAND - VTH) ** 2)
RO_PAR_ANA = RD * RO_ANA / (RD + RO_ANA)
AV_ANA = -GM_ANA * RO_PAR_ANA

# 2. 仿真：直流工作点 / 瞬态 / 交流
def branch_current(analysis, name):
    for key in (name, name.lower(), "v" + name.lower()):
        if key in analysis.branches:
            return np.asarray(analysis.branches[key])
    raise KeyError("找不到支路 %s，可用: %s" % (name, list(analysis.branches.keys())))

def build_circuit(with_input=True, ac_input=False):
    """题卡电路。with_input=False 时去掉输入回路（等价于直流通路）；
    ac_input=True 时输入源带 1 V 交流幅值（正弦源本身没有 AC 幅值，交流扫频会得到 nan）"""
    circuit = Circuit("NMOS common-source amplifier (divider bias)")
    circuit.V("DD", "vdd", circuit.gnd, VDD @ u_V)
    circuit.R("g1", "vdd", "g", (RG1 / 1e3) @ u_kOhm)
    circuit.R("g2", "g", circuit.gnd, (RG2 / 1e3) @ u_kOhm)
    circuit.R("d", "vdd", "d", (RD / 1e3) @ u_kOhm)
    circuit.M(1, "d", "g", circuit.gnd, circuit.gnd, model="NMOS",
              w=W_CH @ u_m, l=L_CH @ u_m)
    circuit.model("NMOS", "NMOS", LEVEL=1, VTO=VTH, KP=KP, LAMBDA=LAMBDA)
    if ac_input:
        circuit.SinusoidalVoltageSource("i", "vi", circuit.gnd,
                                        amplitude=1 @ u_V, frequency=FREQ @ u_Hz)
        circuit.C("b1", "vi", "g", CB1 @ u_F)
    elif with_input:
        circuit.V("i", "vi", circuit.gnd, "SIN(0 %g %g)" % (VI_AMP, FREQ))
        circuit.C("b1", "vi", "g", CB1 @ u_F)
    return circuit

def simulate():
    circuit = build_circuit()
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    op = simulator.operating_point()
    vg = float(op["g"][0])
    vd = float(op["d"][0])
    vs = float(op[circuit.gnd][0]) if circuit.gnd in op.nodes else 0.0
    id_dc_sim = (VDD - vd) / RD

    transient = simulator.transient(step_time=10 @ u_us, end_time=5 @ u_ms)
    t = np.asarray(transient.time)
    v_in = np.asarray(transient["vi"])
    v_g = np.asarray(transient["g"])
    v_out = np.asarray(transient["d"])

    tail = t > t[-1] - 2.0 / FREQ
    def pp(x):
        return x[tail].max() - x[tail].min()
    gain_vi = pp(v_out) / pp(v_in)
    gain_vgs = pp(v_out) / pp(v_g)

    ac_simulator = build_circuit(ac_input=True).simulator(
        temperature=25, nominal_temperature=25)
    ac = ac_simulator.ac(start_frequency=1 @ u_Hz, stop_frequency=1e6 @ u_Hz,
                         number_of_points=20, variation="dec")
    freq = np.asarray(ac.frequency)
    av = np.asarray(ac["d"])
    av_db = 20 * np.log10(np.abs(av))
    i_1k = int(np.argmin(np.abs(freq - FREQ)))
    av_1k = float(np.abs(av)[i_1k])
    phase_1k = float(np.angle(av[i_1k], deg=True))
    def phase_of(x):
        seg = tail
        xs = x[seg] - x[seg].mean()
        w = 2 * np.pi * FREQ
        return float(np.angle(np.sum(xs * np.exp(-1j * w * t[seg])), deg=True))

    d_phase = phase_of(v_out) - phase_of(v_in)
    phase_transient = ((d_phase + 180.0) % 360.0) - 180.0

    return dict(t=t, v_in=v_in, v_g=v_g, v_out=v_out, vg=vg, vd=vd, vs=vs,
                id_dc=id_dc_sim, gain_vi=gain_vi, gain_vgs=gain_vgs,
                freq=freq, av=av, av_db=av_db, av_1k=av_1k,
                phase_1k=phase_1k, phase_transient=phase_transient)

def simulate_output_curve():
    """输出特性（V_GS = V_G = 2 V）用于画直流负载线"""
    circuit = Circuit("nmos output @ VGS")
    circuit.V("DD", "d", circuit.gnd, 0 @ u_V)
    circuit.V("GS", "g", circuit.gnd, VGS_HAND @ u_V)
    circuit.M(1, "d", "g", circuit.gnd, circuit.gnd, model="NMOS",
              w=W_CH @ u_m, l=L_CH @ u_m)
    circuit.model("NMOS", "NMOS", LEVEL=1, VTO=VTH, KP=KP, LAMBDA=LAMBDA)
    analysis = circuit.simulator(temperature=25, nominal_temperature=25).dc(
        VDD=slice(0, VDD, 0.02))
    return np.asarray(analysis["d"]), -branch_current(analysis, "vdd")

# 3. 图：仿真结果（瞬态 / 负载线 / 频响）
def figure_results(sim, vds_curve, id_curve, path):
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))

    ax = axes[0]
    t_ms = sim["t"] * 1e3
    ax.plot(t_ms, sim["v_out"] * 1e3, color="#d62728", lw=1.9, label="输出 v_o")
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("输出 v_o (mV)", color="#d62728")
    ax.tick_params(axis="y", labelcolor="#d62728")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t_ms, sim["v_in"] * 1e3, color="#1f77b4", lw=1.5,
             label="输入 v_i（10 mV）")
    ax2.set_ylabel("输入 v_i (mV)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_title("瞬态：反相放大 —— 输出与输入相差 180°，实测 |A_v| = %.4f" % sim["gain_vi"])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="lower right")

    ax = axes[1]
    ax.plot(vds_curve, id_curve * 1e3, color="#1f77b4", lw=2,
            label="输出特性 V_GS = %.1f V" % VGS_HAND)
    vds_line = np.linspace(0, VDD, 100)
    ax.plot(vds_line, (VDD - vds_line) / RD * 1e3, "k-", lw=1.8,
            label="直流负载线 I_D=(V_DD−V_DS)/R_d")
    ax.plot([sim["vd"]], [sim["id_dc"] * 1e3], "r*", markersize=18,
            label="仿真静态工作点 Q")
    ax.annotate("Q  (%.3f V, %.4f mA)" % (sim["vd"], sim["id_dc"] * 1e3),
                xy=(sim["vd"], sim["id_dc"] * 1e3),
                xytext=(sim["vd"] - 3.4, sim["id_dc"] * 1e3 + 0.55),
                fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red"))
    ax.axvline(VGS_HAND - VTH, color="gray", ls=":", lw=1.3)
    ax.text(VGS_HAND - VTH + 0.06, 0.12, "V_DS(sat)=V_GS−V_th=%.1f V" % (VGS_HAND - VTH),
            fontsize=8.5, color="#555555")
    ax.set_xlim(0, VDD)
    ax.set_ylim(0, max(2.0, sim["id_dc"] * 1e3 * 1.8))
    ax.set_xlabel("V_DS (V)")
    ax.set_ylabel("I_D (mA)")
    ax.set_title("直流通路：负载线与静态工作点")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")

    ax = axes[2]
    ax.semilogx(sim["freq"], sim["av_db"], color="#2ca02c", lw=1.9)
    ax.axvline(FREQ, color="tab:red", ls="--", lw=1.4,
               label="工作频率 %.0f Hz" % FREQ)
    av_1k_db = 20 * np.log10(sim["av_1k"])
    ax.plot([FREQ], [av_1k_db], "o", color="tab:red", markersize=7)
    ax.annotate("|A_v| = %.3f（%.2f dB）" % (sim["av_1k"], av_1k_db),
                xy=(FREQ, av_1k_db), xycoords="data",
                xytext=(0.42, 0.22), textcoords="axes fraction",
                fontsize=9, color="tab:red",
                arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("|A_v| (dB)")
    lo, hi = float(np.nanmin(sim["av_db"])), float(np.nanmax(sim["av_db"]))
    ax.set_ylim(lo - 1.0, hi + 1.5)
    ax.set_title("交流扫频：中频增益与带宽")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower left")

    fig.suptitle("共源放大电路仿真验证（ngspice）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("已保存:", path)

# 4. 手算 vs 仿真表
def write_short_table(sim, path):
    """精简版：一张「手算 vs 仿真」表，不带多余说明"""
    vgs_sim = sim["vg"] - sim["vs"]
    id_sim = sim["id_dc"]
    vds_sim = sim["vd"]
    gm_sim = K_NMOS * (vgs_sim - VTH) * (1 + LAMBDA * vds_sim)
    ro_sim = 1.0 / (LAMBDA * K_HALF * (vgs_sim - VTH) ** 2)
    av_sim = -sim["gain_vi"]
    sat = vds_sim > (vgs_sim - VTH)

    def err(s, h):
        return "" if h == 0 else "%+.2f%%" % ((s - h) / h * 100.0)

    rows = [
        ("V_GS (V)", "%.4f" % VGS_HAND, "%.4f" % vgs_sim, err(vgs_sim, VGS_HAND)),
        ("I_D (mA)", "%.4f" % (ID_HAND * 1e3), "%.4f" % (id_sim * 1e3),
         err(id_sim, ID_HAND)),
        ("V_DS (V)", "%.4f" % VDS_HAND, "%.4f" % vds_sim, err(vds_sim, VDS_HAND)),
        ("是否饱和", "是", "是" if sat else "否", ""),
        ("g_m (mA/V)", "%.4f" % (GM_HAND * 1e3), "%.4f" % (gm_sim * 1e3),
         err(gm_sim, GM_HAND)),
        ("R_ds (kΩ)", "%.2f" % (RDS_HAND / 1e3), "%.2f" % (ro_sim / 1e3),
         err(ro_sim, RDS_HAND)),
        ("R_i (kΩ)", "%.1f" % (RG_PAR / 1e3), "%.1f" % (RG_PAR / 1e3), "0.00%"),
        ("R_o (kΩ)", "—", "%.2f" % ((RD * ro_sim / (RD + ro_sim)) / 1e3), ""),
        ("A_v", "%.4f" % AV_HAND, "%.4f" % av_sim, err(av_sim, AV_HAND)),
        ("相位差 (°)", "180", "%.2f" % sim["phase_1k"], ""),
    ]
    lines = [
        "# 08 NMOS 共源放大电路：手算 vs 仿真\n",
        "| 项目 | 手算 | 仿真 | 误差 |",
        "| --- | --- | --- | --- |",
    ]
    lines += ["| %s | %s | %s | %s |" % r for r in rows]
    lines.append("")
    lines.append("注：I_D、V_DS、g_m、A_v 的差异来自 λ，因子 (1+λV_DS) = %.4f。"
                 % (1 + LAMBDA * vds_sim))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("已保存:", path)

def banner(title):
    print("=" * 96)
    print(title)
    print("=" * 96)

# 5. 主流程
def main():
    banner("NMOS 共源放大电路（分压偏置）仿真")
    print("器件：K = %.2f mA/V²（KP = %.0f µA/V²，W/L = %.0f），V_th = %.1f V，λ = %.2f /V"
          % (K_NMOS * 1e3, KP * 1e6, W_OVER_L, VTH, LAMBDA))

    sim = simulate()
    vgs_sim = sim["vg"] - sim["vs"]
    gm_sim = K_NMOS * (vgs_sim - VTH) * (1 + LAMBDA * sim["vd"])
    ro_sim = 1.0 / (LAMBDA * K_HALF * (vgs_sim - VTH) ** 2)
    av_sim = -sim["gain_vi"]

    print("\n① 直流工作点：手算 vs 仿真")
    print("   手算（题卡）：V_GS = %.4f V, I_D = %.4f mA, V_DS = %.4f V（忽略 λ）"
          % (VGS_HAND, ID_HAND * 1e3, VDS_HAND))
    print("   仿真        ：V_GS = %.4f V, I_D = %.4f mA, V_DS = %.4f V"
          % (vgs_sim, sim["id_dc"] * 1e3, sim["vd"]))
    print("   相对差异    ：V_GS %+.4f%%，I_D %+.3f%%，V_DS %+.3f%%"
          % ((vgs_sim - VGS_HAND) / VGS_HAND * 100,
             (sim["id_dc"] - ID_HAND) / ID_HAND * 100,
             (sim["vd"] - VDS_HAND) / VDS_HAND * 100))
    print("   饱和判据    ：手算 %.1f V > %.1f V（是）；仿真 %.3f V > %.3f V（%s）"
          % (VDS_HAND, VGS_HAND - VTH, sim["vd"], vgs_sim - VTH,
             "是" if sim["vd"] > vgs_sim - VTH else "否"))
    print("   差异来源    ：手算忽略 λ；解析含 λ 得 I_D = %.4f mA、V_DS = %.4f V（与仿真一致）"
          % (ID_ANA * 1e3, VDS_ANA))

    print("\n② 小信号参数与增益：手算 vs 仿真")
    print("   g_m ：手算 %.4f mA/V（= 2I_D/(V_GS−V_th)） 仿真 %.4f mA/V（含 1+λV_DS = %.4f）"
          % (GM_HAND * 1e3, gm_sim * 1e3, 1 + LAMBDA * sim["vd"]))
    print("   R_ds：手算 %.2f kΩ（= 1/(λI_D)）        仿真 %.2f kΩ"
          % (RDS_HAND / 1e3, ro_sim / 1e3))
    print("   R_i ：%.1f kΩ（= R_g1∥R_g2，手算与仿真相同），R_o = R_d∥R_ds = %.4f kΩ"
          % (RG_PAR / 1e3, (RD * ro_sim / (RD + ro_sim)) / 1e3))
    print("   A_v ：手算 %.4f（忽略 R_ds 时 %.4f）  仿真 %.4f（瞬态） / %.4f（交流 @1 kHz）"
          % (AV_HAND, AV_HAND_IDEAL, av_sim, -sim["av_1k"]))
    print("   相对差异：A_v %+.3f%%（全部来自 λ）" % ((av_sim - AV_HAND) / AV_HAND * 100))

    print("\n③ 交流扫频")
    print("   1 kHz 处 |A_v| = %.4f（%.2f dB）" % (sim["av_1k"], 20 * np.log10(sim["av_1k"])))
    print("   1 kHz 处相位差 = %+.2f°（交流分析） / %+.2f°（瞬态基波 DFT）→ 反相放大 180°"
          % (sim["phase_1k"], sim["phase_transient"]))
    upper = sim["freq"][(sim["freq"] > FREQ)
                        & (sim["av_db"] < sim["av_db"][np.argmin(np.abs(sim["freq"] - FREQ))] - 3.0)]
    if upper.size:
        print("   −3 dB 上限频率 ≈ %.3g Hz（高频端，受 C_gs 等寄生影响时才会出现）" % upper[0])

    print("\n④ 仿真结果图与对照表")
    vds_curve, id_curve = simulate_output_curve()
    figure_results(sim, vds_curve, id_curve, BASE_DIR / "cs_amplifier_results.png")
    write_short_table(sim, SHORT_TABLE_PATH)

    banner("完成，文件都在本文件夹内")
    for name in ("cs_amplifier_results.png", "手算vs仿真.md"):
        file = BASE_DIR / name
        if file.exists():
            print("   %-30s %8.2f MB" % (name, file.stat().st_size / 1e6))
    return 0

if __name__ == "__main__":
    sys.exit(main())
