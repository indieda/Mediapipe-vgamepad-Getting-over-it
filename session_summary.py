import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib import gridspec


class SessionSummary:
    def __init__(self):
        self.reset()

    def reset(self):
        # Reps tracked per joint
        self.reps = {"L_elbow": 0, "R_elbow": 0, "L_shoulder": 0, "R_shoulder": 0}
        self.last_state = {"L_elbow": None, "R_elbow": None,
                           "L_shoulder": None, "R_shoulder": None}
        # Store ROM values
        self.rom = {"L_elbow": [], "R_elbow": [],
                    "L_shoulder": [], "R_shoulder": []}

        # Time-series history
        self.history = {
            "time": [],
            "L_elbow": [], "R_elbow": [],
            "L_shoulder": [], "R_shoulder": [],
            "L_speed": [], "R_speed": []
        }

        # Speeds and smoothness
        self.speeds = {"Left": [], "Right": []}
        self.jerk = {"Left": None, "Right": None}
        self.sparc = {"Left": None, "Right": None}

        # Movement classification counts
        self.class_counts = {
            "L_elbow": {"flexion": 0, "extension": 0},
            "R_elbow": {"flexion": 0, "extension": 0},
            "L_shoulder": {"abduction": 0, "adduction": 0},
            "R_shoulder": {"abduction": 0, "adduction": 0}
        }
        
    def update_angles(self, ang_dict, now):
        self.history["time"].append(now)
        for k, v in ang_dict.items():
            if v is None:
                continue
            self.history[k].append(v)
            self.rom[k].append(v)
            self._classify_movement(k, v)
            self._update_reps(k, v)

    def _update_reps(self, joint, angle):
        """Hysteresis-based rep counting"""
        if angle is None:
            return

        state = None
        if "elbow" in joint:
            if angle < 60:
                state = "flexed"
            elif angle > 150:
                state = "extended"
        elif "shoulder" in joint:  # treat as abduction cycles
            if angle < 30:
                state = "down"
            elif angle > 100:
                state = "up"

        last = self.last_state[joint]
        if last and state and state != last:
            if {"flexed", "extended"} == {last, state} or {"down", "up"} == {last, state}:
                self.reps[joint] += 1
        if state:
            self.last_state[joint] = state

    def update_speeds(self, vdict):
        for side in ("Left", "Right"):
            spd = vdict.get(f"{side[0]}_speed")
            if spd is None or np.isnan(spd):
                continue
            self.history[f"{side[0]}_speed"].append(spd)
            self.speeds[side].append(spd)

    def _classify_movement(self, joint, angle):
        if "elbow" in joint:
            if angle < 120:
                self.class_counts[joint]["flexion"] += 1
            else:
                self.class_counts[joint]["extension"] += 1
        elif "shoulder" in joint:
            if angle > 90:
                self.class_counts[joint]["abduction"] += 1
            else:
                self.class_counts[joint]["adduction"] += 1

    def finalize(self):
        self._compute_jerk_and_sparc("Left")
        self._compute_jerk_and_sparc("Right")

        # Robust ROM (5–95 percentile range)
        rom_clean = {}
        for k, vals in self.rom.items():
            if len(vals) > 5:
                smooth = np.convolve(vals, np.ones(5)/5, mode="valid")
                lo, hi = np.percentile(smooth, [5, 95])
                rom_clean[k] = (lo, hi)
            else:
                rom_clean[k] = (None, None)

        # Symmetry score (elbow ROM only)
        if rom_clean["L_elbow"][0] is not None and rom_clean["R_elbow"][0] is not None:
            L_rom = rom_clean["L_elbow"][1] - rom_clean["L_elbow"][0]
            R_rom = rom_clean["R_elbow"][1] - rom_clean["R_elbow"][0]
            sym_index = 200 * abs(R_rom - L_rom) / (R_rom + L_rom + 1e-6)
        else:
            sym_index = None

        return {
            "reps": self.reps,
            "ROM": rom_clean,
            "jerk": self.jerk,
            "sparc": self.sparc,
            "class_counts": self.class_counts,
            "history": self.history,
            "sym_index": sym_index
        }

    def _compute_jerk_and_sparc(self, side):
        spd = np.array(self.speeds[side])
        if len(spd) < 5:
            return
        dt = 1.0 / 30.0
        acc = np.gradient(spd, dt)
        jerk = np.gradient(acc, dt)
        self.jerk[side] = np.nanmean(np.abs(jerk))

        from scipy.fft import rfft, rfftfreq
        f = rfft(spd - np.mean(spd))
        mag = np.abs(f)
        freq = rfftfreq(len(spd), dt)
        mag = mag / (np.max(mag) + 1e-6)
        arc = np.sum(np.sqrt(np.diff(freq)**2 + np.diff(mag)**2))
        self.sparc[side] = -arc

    def _format_rom(self, val):
        if val[0] is None or val[1] is None:
            return "-"
        return f"{val[0]:.1f}-{val[1]:.1f}"

    def _format_num(self, val, fmt=".2f"):
        if val is None:
            return "-"
        return f"{val:{fmt}}"

    def render_summary(self, metrics, pdf_path="session_summary.pdf"):
        fig = plt.figure(figsize=(14, 20), dpi=120)
        gs = gridspec.GridSpec(9, 1, height_ratios=[2, 0.4, 2, 0.4, 1.6, 0.4, 1.6, 0.4, 1.6])

        t = np.array(metrics["history"]["time"], dtype=float)
        L_speed, R_speed = metrics["history"]["L_speed"], metrics["history"]["R_speed"]

        # --- Section 1: Angles (Elbow + Shoulder) ---
        gs01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])

        ax1a = fig.add_subplot(gs01[0, 0])
        ax1a.set_title("Elbow Angles (deg)", fontsize=14, fontweight="bold")
        n = min(len(t), len(metrics["history"]["L_elbow"]), len(metrics["history"]["R_elbow"]))
        if n > 0:
            ax1a.plot(t[:n], metrics["history"]["L_elbow"][:n], label="L_elbow")
            ax1a.plot(t[:n], metrics["history"]["R_elbow"][:n], label="R_elbow")
        ax1a.legend(); ax1a.grid(True)

        ax1b = fig.add_subplot(gs01[0, 1])
        ax1b.set_title("Shoulder Angles (deg)", fontsize=14, fontweight="bold")
        p = min(len(t), len(metrics["history"]["L_shoulder"]), len(metrics["history"]["R_shoulder"]))
        if p > 0:
            ax1b.plot(t[:p], metrics["history"]["L_shoulder"][:p], label="L_shoulder")
            ax1b.plot(t[:p], metrics["history"]["R_shoulder"][:p], label="R_shoulder")
        ax1b.legend(); ax1b.grid(True)

         # --- Section 1 Summary (ROMs) ---
        ax1s = fig.add_subplot(gs[1]); ax1s.axis("off")

        romL_e, romR_e = metrics["ROM"]["L_elbow"], metrics["ROM"]["R_elbow"]
        romL_s, romR_s = metrics["ROM"]["L_shoulder"], metrics["ROM"]["R_shoulder"]

        lines = []

        # Elbow summary
        if all(v is not None for v in (*romL_e, *romR_e)):
            diff_e = abs((romR_e[1] - romR_e[0]) - (romL_e[1] - romL_e[0]))
            elbow_txt = "balanced" if diff_e < 15 else f"asymmetry Δ={diff_e:.1f}°"
            lines.append(
                f"Elbow ROM: {romL_e[0]:.1f}-{romL_e[1]:.1f}° (L), "
                f"{romR_e[0]:.1f}-{romR_e[1]:.1f}° (R), {elbow_txt}"
            )
        else:
            lines.append("Elbow ROM: not enough data")

        # Shoulder summary
        if all(v is not None for v in (*romL_s, *romR_s)):
            diff_s = abs((romR_s[1] - romR_s[0]) - (romL_s[1] - romL_s[0]))
            shoulder_txt = "balanced" if diff_s < 15 else f"asymmetry Δ={diff_s:.1f}°"
            lines.append(
                f"Shoulder ROM: {romL_s[0]:.1f}-{romL_s[1]:.1f}° (L), "
                f"{romR_s[0]:.1f}-{romR_s[1]:.1f}° (R), {shoulder_txt}"
            )
        else:
            lines.append("Shoulder ROM: not enough data")

        # Render with spacing
        y = 0.9
        ax1s.text(0, y, "Summary:", fontsize=12, fontweight="bold", ha="left", va="top")
        for line in lines:
            y -= 0.36  # add vertical spacing between lines
            ax1s.text(0.05, y, line, fontsize=11, ha="left", va="top")



        # --- Section 2: Wrist Speeds ---
        ax2 = fig.add_subplot(gs[2])
        ax2.set_title("Wrist Speeds", fontsize=16, fontweight="bold")
        m = min(len(t), len(L_speed), len(R_speed))
        if m > 0:
            ax2.plot(t[:m], L_speed[:m], label="L speed", linewidth=1.8)
            ax2.plot(t[:m], R_speed[:m], label="R speed", linewidth=1.8)
        ax2.legend(); ax2.grid(True)

        ax2s = fig.add_subplot(gs[3]); ax2s.axis("off")
        txt = "Very low wrist activity." if (m > 0 and np.mean(L_speed) < 0.05 and np.mean(R_speed) < 0.05) else "Active wrist speeds detected."
        ax2s.text(0, 0.5, "Summary: " + txt, fontsize=12)

        # --- Section 3: Elbow Flex/Ext Distribution ---
        ax3 = fig.add_subplot(gs[4])
        labels_e = ["flexion", "extension"]
        L_vals_e = [metrics["class_counts"]["L_elbow"][l] for l in labels_e]
        R_vals_e = [metrics["class_counts"]["R_elbow"][l] for l in labels_e]
        total_e = sum(L_vals_e) + sum(R_vals_e)
        if total_e > 0:
            L_vals_e = [100 * c / total_e for c in L_vals_e]
            R_vals_e = [100 * c / total_e for c in R_vals_e]
        ax3.bar([0, 1], L_vals_e, 0.36, label="Left Elbow")
        ax3.bar([0.36, 1.36], R_vals_e, 0.36, label="Right Elbow")
        ax3.set_xticks([0.18, 1.18]); ax3.set_xticklabels(labels_e)
        ax3.set_title("Elbow Flexion / Extension Distribution"); ax3.legend()
        
        ax3s = fig.add_subplot(gs[5]); ax3s.axis("off")
        if abs(sum(L_vals_e) - sum(R_vals_e)) < 5:
            txt = "Counts suggest balanced elbow usage."
        else:
            txt = "Asymmetric elbow usage."
        ax3s.text(0, 0.5, "Summary: " + txt, fontsize=12)


        # --- Section 4: Shoulder Abd/Add Distribution ---
        ax4 = fig.add_subplot(gs[6])
        labels_s = ["abduction", "adduction"]
        L_vals_s = [metrics["class_counts"]["L_shoulder"][l] for l in labels_s]
        R_vals_s = [metrics["class_counts"]["R_shoulder"][l] for l in labels_s]
        total_s = sum(L_vals_s) + sum(R_vals_s)
        if total_s > 0:
            L_vals_s = [100 * c / total_s for c in L_vals_s]
            R_vals_s = [100 * c / total_s for c in R_vals_s]
        ax4.bar([0, 1], L_vals_s, 0.36, label="Left Shoulder")
        ax4.bar([0.36, 1.36], R_vals_s, 0.36, label="Right Shoulder")
        ax4.set_xticks([0.18, 1.18]); ax4.set_xticklabels(labels_s)
        ax4.set_title("Shoulder Abduction / Adduction Distribution"); ax4.legend()
        ax4s = fig.add_subplot(gs[7]); ax4s.axis("off")
        if abs(sum(L_vals_s) - sum(R_vals_s)) < 5:
            txt = "Shoulder movement balanced."
        else:
            txt = "Shoulder imbalance detected."
        ax4s.text(0, 0.5, "Summary: " + txt, fontsize=12)


        # --- Section 5: Core Metrics ---
        ax5 = fig.add_subplot(gs[8]); ax5.axis("off")
        reps, rom, jerk, sparc, sym = metrics["reps"], metrics["ROM"], metrics["jerk"], metrics["sparc"], metrics["sym_index"]

        core_table = [
            ["Metric", "Left", "Right"],
            ["Elbow Reps", reps["L_elbow"], reps["R_elbow"]],
            ["Shoulder Reps", reps["L_shoulder"], reps["R_shoulder"]],
            ["ROM Elbow", self._format_rom(rom["L_elbow"]), self._format_rom(rom["R_elbow"])],
            ["ROM Shoulder", self._format_rom(rom["L_shoulder"]), self._format_rom(rom["R_shoulder"])],
            ["Jerk Index", self._format_num(jerk["Left"]), self._format_num(jerk["Right"])],
            ["SPARC", self._format_num(sparc["Left"]), self._format_num(sparc["Right"])],
            ["Symmetry %", self._format_num(sym, ".1f"), ""]
        ]
        table = ax5.table(cellText=core_table, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(11)
        
        if sym is not None:
            if sym < 15:
                color = "#b6fcb6"
                comment = "Symmetry good (<15%)."
            elif sym < 30:
                color = "#fff5b6"
                comment = "Mild asymmetry (15–30%). Monitor."
            else:
                color = "#fcb6b6"
                comment = "Marked asymmetry (>30%). Attention needed."
            for j in range(3):
                table[(7, j)].set_facecolor(color)
        else:
            comment = "Not enough data for symmetry."

        # Place the comment below the table
        ax5.text(0, -0.7, "Summary: " + comment, fontsize=12)


        fig.suptitle("Session Summary", fontsize=20, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        # ---- Save PDF ----
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        print(f"[INFO] PDF report saved to {pdf_path}")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img, 1)
        plt.close(fig)
        return img
