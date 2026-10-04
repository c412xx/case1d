"""带负载的一阶 RC 低通滤波器：方波瞬态 + 波特图 + 手算/仿真对照表
程序做三件事：
    1. 方波输入的瞬态仿真（输入/输出波形），用「一个时间常数到达终值 63.2%（2τ 到达 86.5%）」
       的判据从上升沿读出 τ
    2. 交流分析（波特图：幅频 + 相频），从 -3 dB 点读取 f_c，并用 τ = 1/(2π·f_c) 反向校核
    3. 输出「手算 vs 仿真」对照表
"""

import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
FIGURE_PATH = BASE_DIR / "06_loaded_rc_squarewave_bode.png"
TABLE_PATH = BASE_DIR / "hand_vs_sim_table.md"

import PySpice.Logging.Logging as Logging

Logging.setup_logging(logging_level="ERROR")  # 静音 ngspice 的噪声日志

import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_Hz, u_kOhm, u_ms, u_s, u_uF, u_V

# --------------------------------------------------------------------------- #
# 0. 电路参数与手算值
# --------------------------------------------------------------------------- #
R_VALUE = 20.0      # kΩ
RL_VALUE = 20.0     # kΩ
C_VALUE = 100.0     # µF

REQ_HAND = (R_VALUE * RL_VALUE) / (R_VALUE + RL_VALUE)          # 10 kΩ
TAU_HAND = REQ_HAND * 1e3 * C_VALUE * 1e-6                      # 1 s
FC_HAND = 1.0 / (2.0 * np.pi * TAU_HAND)                        # 0.159155 Hz
DC_GAIN_HAND = RL_VALUE / (R_VALUE + RL_VALUE)                  # 0.5 (-6.0206 dB)
DC_GAIN_HAND_DB = 20.0 * np.log10(DC_GAIN_HAND)

# 方波：±1 V、0.1 Hz（周期 10 s，即每个半周期 5τ），仿真 2 个周期
V_HIGH, V_LOW = 1.0, -1.0
SQ_FREQ = 0.1                       # Hz
SQ_PERIOD = 1.0 / SQ_FREQ           # 10 s
SQ_HALF = SQ_PERIOD / 2.0           # 5 s
T_END = 2.0 * SQ_PERIOD             # 20 s
T_EDGE = SQ_PERIOD                  # 取第 2 个上升沿（t = 10 s）做 τ 辨识，波形已进入稳态

# 方波稳态高电平的解析值：半个周期后电容电压 = V_HIGH·RL/(R+RL)·tanh(T_half/(2τ))
V_HIGH_HAND = DC_GAIN_HAND * V_HIGH * np.tanh(SQ_HALF / (2.0 * TAU_HAND))

# 方波源用「原始 PULSE 串」而不是 circuit.PulseVoltageSource()：
#   后者会生成 "Vinput vin 0 DC 0V PULSE(...)"，而 ngspice 47 会针对该 DC 值往 stderr 打印
#       Note: vinput: dc value used for op instead of transient time=0 value.
#   PySpice 1.5 把 stderr 上任何不以 "Warning:" 开头的行都当致命错误
#   （Shared.py: _send_char -> _error_in_stderr = True），于是 run 直接失败。
#   不带 DC 值时 ngspice 只发 Warning（PySpice 会容忍），而且工作点就用 t=0 的 -1 V，
#   正好让初始输出等于 -0.5 V（直流分压值），更符合理论。
PULSE_SPEC = "PULSE({low:g} {high:g} 0 1m 1m {half:g} {period:g})".format(
    low=V_LOW, high=V_HIGH, half=SQ_HALF, period=SQ_PERIOD)


def build_circuit(source):
    """按需构建电路：source 取 'pulse'（瞬态用）或 'ac'（交流分析用）。"""
    circuit = Circuit("Loaded RC low-pass")
    if source == "pulse":
        circuit.V("input", "vin", circuit.gnd, PULSE_SPEC)
    else:
        # AC 分析需要 ac_magnitude，SinusoidalVoltageSource 默认 ac_magnitude = 1 V
        circuit.SinusoidalVoltageSource(
            "input", "vin", circuit.gnd, amplitude=1 @ u_V, frequency=1 @ u_Hz,
        )
    circuit.R(1, "vin", "vout", R_VALUE @ u_kOhm)       # 串联电阻 R
    circuit.C(1, "vout", circuit.gnd, C_VALUE @ u_uF)   # 电容 C
    circuit.R("L", "vout", circuit.gnd, RL_VALUE @ u_kOhm)  # 负载 RL
    return circuit


def banner(title):
    print("=" * 84)
    print(title)
    print("=" * 84)


# --------------------------------------------------------------------------- #
# 1. 直流工作点：验证低频增益
# --------------------------------------------------------------------------- #
banner("带负载的 RC 低通：R=20k, RL=20k, C=100uF -> Req=10k, τ=1s, f_c=0.159Hz")

dc_circuit = Circuit("DC gain check")
dc_circuit.V("input", "vin", dc_circuit.gnd, 1 @ u_V)
dc_circuit.R(1, "vin", "vout", R_VALUE @ u_kOhm)
dc_circuit.C(1, "vout", dc_circuit.gnd, C_VALUE @ u_uF)
dc_circuit.R("L", "vout", dc_circuit.gnd, RL_VALUE @ u_kOhm)

dc_analysis = dc_circuit.simulator(temperature=25, nominal_temperature=25).operating_point()
dc_gain_sim = float(dc_analysis["vout"][0]) / 1.0

# --------------------------------------------------------------------------- #
# 2. 瞬态：方波输入 / 输出
# --------------------------------------------------------------------------- #
transient_analysis = build_circuit("pulse").simulator(
    temperature=25, nominal_temperature=25
).transient(step_time=1 @ u_ms, end_time=T_END @ u_s)

time = np.asarray(transient_analysis.time)
v_in = np.asarray(transient_analysis.vin)
v_out = np.asarray(transient_analysis.vout)
print("瞬态采样点数 =", len(time))

# --- τ 辨识（用第 2 个上升沿，t = 10 s） ---------------------------------- #
# 判据：一个时间常数到达终值的 63.2%，两个时间常数到达 86.5%
window = (time > T_EDGE + 0.05) & (time < T_EDGE + SQ_HALF * 0.9)
t_win = time[window] - T_EDGE
v_win = v_out[window]

v_start = v_out[time <= T_EDGE][-1]          # 阶跃前的初值（约 -0.493 V）
v_final = DC_GAIN_HAND * V_HIGH              # 阶跃后的终值（0.5 V，直流分压）


def level_time(fraction):
    """阶跃后电压到达 v_start + fraction·(v_final − v_start) 的时刻"""
    target = v_start + fraction * (v_final - v_start)
    index = int(np.argmax(v_win >= target))
    return float(np.interp(target, [v_win[index - 1], v_win[index]],
                           [t_win[index - 1], t_win[index]]))


v_632 = v_start + 0.632 * (v_final - v_start)      # 用于图上标注
TAU_632 = level_time(0.632)                        # 一个时间常数 → τ
T2TAU_632 = level_time(0.865)                      # 两个时间常数 → 2τ

# --- 方波稳态高电平（最后半个周期取最大） --------------------------------- #
last_half = time > T_END - SQ_HALF
V_HIGH_SIM = float(v_out[last_half].max())

# --------------------------------------------------------------------------- #
# 3. 交流分析：波特图
# --------------------------------------------------------------------------- #
ac_analysis = build_circuit("ac").simulator(
    temperature=25, nominal_temperature=25
).ac(start_frequency=0.01 @ u_Hz, stop_frequency=100 @ u_Hz,
     number_of_points=50, variation="dec")

frequency = np.asarray(ac_analysis.frequency)
h = np.asarray(ac_analysis.vout)
gain_db = 20.0 * np.log10(np.abs(h))
phase_deg = np.angle(h, deg=True)

# 相对低频增益下降 3.01 dB 的点即截止频率
# 注意：增益随频率单调下降，np.interp 要求 xp 递增，故把数组反向再插值
target_db = DC_GAIN_HAND_DB - 3.0103
FC_SIM = float(10 ** np.interp(target_db, gain_db[::-1], np.log10(frequency)[::-1]))
PHASE_AT_FC = float(np.interp(np.log10(FC_SIM), np.log10(frequency), phase_deg))
TAU_AC = 1.0 / (2.0 * np.pi * FC_SIM)          # 由交流 −3 dB 点反推的时间常数（交叉校核）

# --------------------------------------------------------------------------- #
# 4. 手算 vs 仿真 对照表
# --------------------------------------------------------------------------- #
rows = [
    ("R_eq = R·RL/(R+RL)", "kΩ", REQ_HAND, TAU_632 / (C_VALUE * 1e-6) / 1e3, 3),
    ("τ（63.2% 判据，瞬态）", "s", TAU_HAND, TAU_632, 4),
    ("τ = 1/(2πf_c)（交流反推）", "s", TAU_HAND, TAU_AC, 4),
    ("f_c = 1/(2πτ)", "Hz", FC_HAND, FC_SIM, 5),
    ("低频增益 RL/(R+RL)", "V/V", DC_GAIN_HAND, dc_gain_sim, 4),
    ("低频增益", "dB", DC_GAIN_HAND_DB, 20 * np.log10(dc_gain_sim), 3),
    ("f_c 处相位", "°", -45.0, PHASE_AT_FC, 2),
    ("方波稳态高电平", "V", V_HIGH_HAND, V_HIGH_SIM, 4),
]


def format_table():
    header = ("| 项目 | 单位 | 手算 | 仿真 | 相对误差 |\n"
              "| --- | --- | --- | --- | --- |\n")
    lines = [header]
    text_rows = []
    for name, unit, hand, sim, digits in rows:
        error = (sim - hand) / hand * 100.0
        lines.append("| {} | {} | {:.{d}f} | {:.{d}f} | {:+.3f}% |\n".format(
            name, unit, hand, sim, error, d=digits))
        text_rows.append((name, unit, hand, sim, error, digits))
    return "".join(lines), text_rows


markdown, table_rows = format_table()

print()
banner("τ / 截止频率：手算 vs 仿真")
print("{:<26}{:<7}{:>14}{:>14}{:>12}".format("项目", "单位", "手算", "仿真", "相对误差"))
print("-" * 84)
for name, unit, hand, sim, error, digits in table_rows:
    print("{:<26}{:<7}{:>14.{d}f}{:>14.{d}f}{:>11.3f}%".format(
        name, unit, hand, sim, error, d=digits))
print("-" * 84)

TABLE_PATH.write_text(markdown, encoding="utf-8")
print("表格已保存:", TABLE_PATH)

# --------------------------------------------------------------------------- #
# 5. 画图：瞬态全景 + 上升沿放大 + 波特图
# --------------------------------------------------------------------------- #
figure, axes = plt.subplots(2, 2, figsize=(13, 8.5))
ax_full, ax_zoom, ax_gain, ax_phase = axes.ravel()

# (1) 瞬态全景
ax_full.plot(time, v_in, "--", color="tab:gray", label="输入方波 vin")
ax_full.plot(time, v_out, color="tab:blue", label="输出 vout")
ax_full.axvline(T_EDGE, color="tab:red", linestyle=":", alpha=0.7)
ax_full.annotate("τ 辨识用的上升沿", xy=(T_EDGE, 0.1), xytext=(T_EDGE + 1.0, 0.30),
                 arrowprops=dict(arrowstyle="->", color="tab:red"), color="tab:red")
ax_full.set_xlabel("时间 (s)")
ax_full.set_ylabel("电压 (V)")
ax_full.set_title("方波响应全景（f = 0.1 Hz，2 个周期）")
ax_full.grid(True, alpha=0.3)
ax_full.legend(loc="upper right")

# (2) 上升沿放大 + τ 标注
zoom = (time > T_EDGE - 0.4) & (time < T_EDGE + 4.0)
t_zoom = time[zoom] - T_EDGE
ax_zoom.plot(t_zoom, v_in[zoom], "--", color="tab:gray", label="输入 vin")
ax_zoom.plot(t_zoom, v_out[zoom], color="tab:blue", label="输出 vout")
ax_zoom.axhline(v_final, color="tab:green", linestyle=":", alpha=0.8,
                label="终值 %.3f V" % v_final)
ax_zoom.axhline(v_632, color="tab:orange", linestyle=":", alpha=0.8,
                label="63.2%% 电平 %.3f V" % v_632)
ax_zoom.axvline(TAU_632, color="tab:orange", alpha=0.9)
ax_zoom.plot([TAU_632], [v_632], "o", color="tab:orange")
ax_zoom.annotate("τ_sim = %.4f s\n（63.2%%，v = %.3f V）" % (TAU_632, v_632),
                 xy=(TAU_632, v_632),
                 xytext=(TAU_632 + 0.45, v_632 - 0.45),
                 arrowprops=dict(arrowstyle="->", color="tab:orange"),
                 color="tab:orange", fontsize=9)
# 2τ 处到达 86.5%，用于验证指数规律（比值应接近 2）
v_865 = v_start + 0.865 * (v_final - v_start)
ax_zoom.axhline(v_865, color="tab:purple", linestyle=":", alpha=0.8,
                label="86.5%% 电平 %.3f V" % v_865)
ax_zoom.axvline(T2TAU_632, color="tab:purple", alpha=0.9)
ax_zoom.plot([T2TAU_632], [v_865], "s", color="tab:purple")
ax_zoom.annotate("2τ = %.4f s\n（比值 %.3f ≈ 2）" % (T2TAU_632, T2TAU_632 / TAU_632),
                 xy=(T2TAU_632, v_865),
                 xytext=(T2TAU_632 + 0.35, v_865 - 0.45),
                 arrowprops=dict(arrowstyle="->", color="tab:purple"),
                 color="tab:purple", fontsize=9)
ax_zoom.set_xlim(-0.4, 4.0)
ax_zoom.set_xlabel("相对上升沿的时间 (s)")
ax_zoom.set_ylabel("电压 (V)")
ax_zoom.set_title("上升沿放大：63.2% / 86.5% 判据直接读 τ（不做曲线拟合）")
ax_zoom.grid(True, alpha=0.3)
ax_zoom.legend(loc="lower right", fontsize=8)

# (3) 幅频特性
ax_gain.semilogx(frequency, gain_db, color="tab:blue")
ax_gain.axhline(DC_GAIN_HAND_DB, color="tab:gray", linestyle=":", alpha=0.8,
                label="低频增益 %.3f dB" % DC_GAIN_HAND_DB)
ax_gain.axhline(target_db, color="tab:green", linestyle=":", alpha=0.8,
                label="−3 dB 电平 %.3f dB" % target_db)
ax_gain.axvline(FC_HAND, color="tab:red", linestyle="--", alpha=0.8,
                label="手算 f_c = %.5f Hz" % FC_HAND)
ax_gain.plot([FC_SIM], [target_db], "o", color="tab:green")
ax_gain.annotate("仿真 f_c = %.5f Hz" % FC_SIM, xy=(FC_SIM, target_db),
                 xytext=(FC_SIM * 1.6, target_db + 4.0),
                 arrowprops=dict(arrowstyle="->", color="tab:green"),
                 color="tab:green", fontsize=9)
ax_gain.set_xlabel("频率 (Hz)")
ax_gain.set_ylabel("增益 (dB)")
ax_gain.set_title("波特图 · 幅频")
ax_gain.grid(True, which="both", alpha=0.3)
ax_gain.legend(loc="lower left", fontsize=8)

# (4) 相频特性
ax_phase.semilogx(frequency, phase_deg, color="tab:orange")
ax_phase.axvline(FC_HAND, color="tab:red", linestyle="--", alpha=0.8,
                 label="手算 f_c = %.5f Hz" % FC_HAND)
ax_phase.axhline(-45.0, color="tab:gray", linestyle=":", alpha=0.8)
ax_phase.plot([FC_SIM], [PHASE_AT_FC], "o", color="tab:purple")
ax_phase.annotate("仿真 %.2f° @ f_c" % PHASE_AT_FC, xy=(FC_SIM, PHASE_AT_FC),
                  xytext=(FC_SIM * 1.6, PHASE_AT_FC + 12),
                  arrowprops=dict(arrowstyle="->", color="tab:purple"),
                  color="tab:purple", fontsize=9)
ax_phase.set_xlabel("频率 (Hz)")
ax_phase.set_ylabel("相位 (°)")
ax_phase.set_title("波特图 · 相频")
ax_phase.grid(True, which="both", alpha=0.3)
ax_phase.legend(loc="lower left", fontsize=8)

figure.suptitle("带负载的一阶 RC 低通：R=20 kΩ, R_L=20 kΩ, C=100 µF", fontsize=13)
figure.tight_layout(rect=(0, 0, 1, 0.97))
figure.savefig(FIGURE_PATH, dpi=120, bbox_inches="tight")
print("图片已保存:", FIGURE_PATH)

plt.show()
