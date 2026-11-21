"""
Laser lap-weld thermal simulator - simple desktop app (Tkinter)

Runs the Goldak moving-heat-source solver from laser_weld_goldak.py for up to
5 materials in one go and compares them (pool size, temperatures, cooling time).

Files needed in the SAME folder:
    weld_app.py              (this file)
    laser_weld_goldak.py     (the solver)
    materials.json           (created automatically on first run; edit it to
                              change / add materials, max 5)

Run:   python weld_app.py
Test without a window:   python weld_app.py --selftest
"""
import csv
import json
import math
import os
import queue
import subprocess
import sys
import threading
from dataclasses import replace

import numpy as np
from matplotlib.figure import Figure

import laser_weld_goldak as S

HERE = os.path.dirname(os.path.abspath(__file__))
MAT_FILE = os.path.join(HERE, "materials.json")
MAX_MATERIALS = 5

# ----------------------------------------------------------------------------
# Material library (approximate literature values - replace with your textbook /
# datasheet numbers for the report). Tables: temperature (C), k (W/mK), cp (J/kgK)
# ----------------------------------------------------------------------------
DEFAULT_MATERIALS = [
    {"name": "AISI 304 stainless", "rho": 7800, "T_sol": 1400, "T_liq": 1454, "L_fus": 2.7e5,
     "eta": 0.70, "T_haz": 600,
     "T_tab": [20, 500, 1000, 1400, 1500, 3000],
     "k_tab": [14.9, 20.0, 25.7, 30.0, 32.0, 36.0],
     "cp_tab": [500, 560, 620, 680, 700, 720]},
    {"name": "Mild steel (S235)", "rho": 7850, "T_sol": 1450, "T_liq": 1510, "L_fus": 2.7e5,
     "eta": 0.70, "T_haz": 723,
     "T_tab": [20, 500, 1000, 1500, 3000],
     "k_tab": [50, 38, 28, 30, 35],
     "cp_tab": [480, 590, 650, 690, 700]},
    {"name": "Ti-6Al-4V", "rho": 4430, "T_sol": 1605, "T_liq": 1660, "L_fus": 2.9e5,
     "eta": 0.60, "T_haz": 700,
     "T_tab": [20, 500, 1000, 1650, 3000],
     "k_tab": [7, 12, 19, 30, 33],
     "cp_tab": [530, 610, 700, 830, 830]},
    {"name": "Aluminium 6061", "rho": 2700, "T_sol": 582, "T_liq": 652, "L_fus": 3.9e5,
     "eta": 0.50, "T_haz": 250,
     "T_tab": [20, 300, 582, 652, 1000, 3000],
     "k_tab": [167, 185, 190, 90, 95, 100],
     "cp_tab": [900, 1000, 1100, 1180, 1180, 1200]},
    {"name": "Inconel 718", "rho": 8190, "T_sol": 1260, "T_liq": 1336, "L_fus": 2.9e5,
     "eta": 0.70, "T_haz": 600,
     "T_tab": [20, 500, 1000, 1300, 3000],
     "k_tab": [11.4, 17.5, 25.0, 29.0, 35.0],
     "cp_tab": [435, 520, 610, 680, 700]},
]
REQUIRED = ["name", "rho", "T_sol", "T_liq", "L_fus", "eta", "T_haz", "T_tab", "k_tab", "cp_tab"]


def load_materials():
    """Return (list_of_materials, message). Creates materials.json if missing."""
    if not os.path.exists(MAT_FILE):
        with open(MAT_FILE, "w") as f:
            json.dump(DEFAULT_MATERIALS, f, indent=2)
        return [dict(m) for m in DEFAULT_MATERIALS], ""
    try:
        with open(MAT_FILE) as f:
            mats = json.load(f)
        assert isinstance(mats, list) and mats, "file must contain a list of materials"
        out, seen = [], set()
        for m in mats[:MAX_MATERIALS]:
            for key in REQUIRED:
                assert key in m, f"material '{m.get('name', '?')}' is missing '{key}'"
            n = len(m["T_tab"])
            assert len(m["k_tab"]) == n and len(m["cp_tab"]) == n, \
                f"'{m['name']}': T_tab, k_tab and cp_tab must have the same length"
            assert m["T_liq"] > m["T_sol"], f"'{m['name']}': T_liq must be above T_sol"
            name = m["name"]
            while name in seen:
                name += "*"
            seen.add(name)
            out.append(dict(m, name=name))
        msg = "" if len(mats) <= MAX_MATERIALS else f"Only the first {MAX_MATERIALS} materials are used."
        return out, msg
    except Exception as e:  # fall back to defaults, tell the user
        return [dict(m) for m in DEFAULT_MATERIALS], f"materials.json problem ({e}); using defaults."


def params_for(base, m, eta_override=None):
    return replace(base, name=m["name"], rho=float(m["rho"]), T_sol=float(m["T_sol"]),
                   T_liq=float(m["T_liq"]), L_fus=float(m["L_fus"]),
                   eta=float(m["eta"] if eta_override is None else eta_override),
                   T_haz=float(m["T_haz"]), T_tab=tuple(m["T_tab"]),
                   k_tab=tuple(m["k_tab"]), cp_tab=tuple(m["cp_tab"]))


def penetration_label(r):
    p = r["p"]
    if r["depth"] <= 0:
        return "No melting"
    if r["full_pen"]:
        return "Burn-through"
    if r["depth"] >= p.t1 + 0.25:
        return "Joined (lower sheet fused)"
    return "Not joined"


# ----------------------------------------------------------------------------
# Plot functions (work on any matplotlib Figure, so they are testable without a window)
# ----------------------------------------------------------------------------
def _grid(n):
    ncols = min(3, n)
    return math.ceil(n / ncols), ncols


def draw_contours(fig, results, view):
    """view = 'top' or 'section'"""
    fig.clear()
    if not results:
        return
    nrows, ncols = _grid(len(results))
    for i, (name, r) in enumerate(results.items()):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        p, snap = r["p"], r["snap"]
        T, ix = snap["T"], snap["ix"]
        levels = np.linspace(p.T0, max(T.max(), p.T_liq * 1.1), 20)
        y_full = np.concatenate([-r["ys"][::-1], r["ys"]])
        if view == "top":
            top = T[:, :, 0].T
            Z = np.vstack([top[::-1], top])
            c = ax.contourf(r["xs"], y_full, Z, levels=levels, cmap="inferno")
            if Z.max() >= p.T_liq:
                ax.contour(r["xs"], y_full, Z, levels=[p.T_liq], colors="cyan", linewidths=1.3)
            ax.set_xlabel("x along weld (mm)"); ax.set_ylabel("y (mm)")
        else:
            cs = T[ix, :, :].T
            Z = np.hstack([cs[:, ::-1], cs])
            c = ax.contourf(y_full, r["zs"], Z, levels=levels, cmap="inferno")
            if Z.max() >= p.T_liq:
                ax.contour(y_full, r["zs"], Z, levels=[p.T_liq], colors="cyan", linewidths=1.3)
            ax.axhline(p.t1, color="white", ls="--", lw=0.9)
            ax.invert_yaxis()
            ax.set_xlabel("y (mm)"); ax.set_ylabel("depth z (mm)")
            ax.set_xlim(-6, 6)
        ax.set_aspect("equal" if view == "top" else "auto")
        ax.set_title(f"{name}\npool {r['length']:.1f} x {r['width']:.1f} x {r['depth']:.2f} mm", fontsize=9)
        fig.colorbar(c, ax=ax, label="T (°C)")
    what = "Top surface" if view == "top" else "Cross-section under the laser (white dashes = sheet interface)"
    fig.suptitle(f"{what}  -  cyan line = liquidus", fontsize=10)


def draw_cycles(fig, results):
    fig.clear()
    if not results:
        return
    ax = fig.add_subplot(111)
    for i, (name, r) in enumerate(results.items()):
        col = f"C{i}"
        ax.plot(r["times"], r["hist"]["top, y=0 (centreline)"], color=col, label=name)
        ax.plot(r["times"], r["hist"]["interface, y=0"], color=col, ls="--", lw=1)
    ax.set_xlabel("time (s)"); ax.set_ylabel("temperature (°C)")
    ax.set_title("Thermal cycles at mid-weld  (solid = top centreline, dashed = sheet interface)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend()


def draw_compare(fig, results):
    fig.clear()
    if not results:
        return
    names = list(results)
    cols = [f"C{i}" for i in range(len(names))]
    items = [("Pool depth (mm)", "depth", "%.2f"), ("Pool width (mm)", "width", "%.2f"),
             ("Peak temperature (°C)", "T_max", "%.0f"), ("Cooling time t8/5 (s)", "t85", "%.2f")]
    for k, (title, key, fmt) in enumerate(items):
        ax = fig.add_subplot(2, 2, k + 1)
        vals = [results[n][key] for n in names]
        bars = ax.bar(range(len(names)), np.nan_to_num(vals), color=cols)
        ax.bar_label(bars, labels=[("n/a" if not np.isfinite(v) else fmt % v) for v in vals], fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split(" (")[0] for n in names], rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        if key == "depth":
            p = results[names[0]]["p"]
            ax.axhline(p.t1, color="k", ls="--", lw=1)
            ax.text(len(names) - 0.5, p.t1, " sheet interface", va="bottom", ha="right", fontsize=7)
            ax.axhline(p.t1 + p.t2, color="r", ls=":", lw=1)
            ax.text(len(names) - 0.5, p.t1 + p.t2, " bottom surface", va="bottom", ha="right", fontsize=7)
            ax.set_ylim(0, (p.t1 + p.t2) * 1.15)


TABLE_COLS = [("Material", 150), ("Heat input (J/mm)", 105), ("Length (mm)", 80), ("Width (mm)", 80),
              ("Depth (mm)", 80), ("Result", 160), ("Peak T (°C)", 85), ("HAZ each side (mm)", 115),
              ("t8/5 (s)", 70), ("Run time (s)", 80)]


def table_row(name, r):
    p = r["p"]
    t85 = "n/a" if not np.isfinite(r["t85"]) else f"{r['t85']:.2f}"
    return (name, f"{p.P * p.eta / p.v:.1f}", f"{r['length']:.2f}", f"{r['width']:.2f}",
            f"{r['depth']:.2f}", penetration_label(r), f"{r['T_max']:.0f}",
            f"{r['haz_width'] / 2:.2f}", t85, f"{r['runtime']:.0f}")


# ----------------------------------------------------------------------------
# Parameter fields: (label, key, default)
# ----------------------------------------------------------------------------
FIELDS = [
    ("Laser / Goldak source", [
        ("Laser power (W)", "P", "2500"),
        ("Welding speed (mm/s)", "v", "20"),
        ("a_f  front length (mm)", "af", "1.0"),
        ("a_r  rear length (mm)", "ar", "3.0"),
        ("b  half width (mm)", "b", "1.5"),
        ("c  depth (mm)", "c", "2.0"),
        ("Focus depth below top (mm)", "focus_offset", "0.0")]),
    ("Joint geometry (from your CAD)", [
        ("Sheet 1 thickness (mm)", "t1", "1.5"),
        ("Sheet 2 thickness (mm)", "t2", "1.5"),
        ("Overlap width (mm)", "overlap", "20"),
        ("Block length (mm)", "L", "30"),
        ("Weld start x (mm)", "weld_start", "5"),
        ("Weld end x (mm)", "weld_end", "25")]),
    ("Numerics / environment", [
        ("Grid size dx (mm)", "dx", "0.3"),
        ("Cooling time after weld (s)", "t_cool", "2.0"),
        ("Convection h (W/m²K)", "h_conv", "15"),
        ("Emissivity", "emissivity", "0.5"),
        ("Ambient temperature (°C)", "T0", "20")]),
]


def build_base(vals):
    """vals: dict key -> float. Validates and returns a Params (default material)."""
    need_pos = ["P", "v", "af", "ar", "b", "c", "t1", "t2", "overlap", "L", "dx", "t_cool"]
    for k in need_pos:
        if vals[k] <= 0:
            raise ValueError(f"'{k}' must be greater than zero")
    if not 0 < vals["emissivity"] <= 1:
        raise ValueError("Emissivity must be between 0 and 1")
    if not 0.1 <= vals["dx"] <= 1.0:
        raise ValueError("Grid size dx should be between 0.1 and 1.0 mm")
    if not 0 <= vals["weld_start"] < vals["weld_end"] <= vals["L"]:
        raise ValueError("Need 0 <= weld start < weld end <= block length")
    if vals["focus_offset"] < 0:
        raise ValueError("Focus depth must be zero or positive")
    return S.Params(L=vals["L"], W=vals["overlap"] / 2, t1=vals["t1"], t2=vals["t2"],
                    weld_start=vals["weld_start"], weld_end=vals["weld_end"], dx=vals["dx"],
                    P=vals["P"], v=vals["v"], af=vals["af"], ar=vals["ar"], b=vals["b"], c=vals["c"],
                    focus_offset=vals["focus_offset"], T0=vals["T0"], h_conv=vals["h_conv"],
                    emissivity=vals["emissivity"], t_cool=vals["t_cool"])


# ----------------------------------------------------------------------------
# The window
# ----------------------------------------------------------------------------
def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Laser lap-weld thermal simulator (Goldak heat source)")
            self.geometry("1350x820")
            self.q = queue.Queue()
            self.stop_evt = threading.Event()
            self.worker = None
            self.results = {}
            self.vars, self.labels = {}, {}
            self.mat_vars = []
            self.materials, msg = load_materials()
            self._build_left()
            self._build_right()
            self._build_material_checks()
            if msg:
                self.status.set(msg)
            self.after(100, self._poll)

        # ---------- left panel ----------
        def _build_left(self):
            left = ttk.Frame(self, padding=8)
            left.pack(side="left", fill="y")

            box = ttk.LabelFrame(left, text="1. Materials (tick up to 5)", padding=6)
            box.pack(fill="x")
            self.mat_frame = ttk.Frame(box)
            self.mat_frame.pack(fill="x")
            row = ttk.Frame(box); row.pack(fill="x", pady=(4, 0))
            ttk.Button(row, text="Edit materials file", command=self._open_mat_file).pack(side="left")
            ttk.Button(row, text="Reload", command=self._reload_materials).pack(side="left", padx=4)

            self.use_mat_eta = tk.BooleanVar(value=True)
            ttk.Checkbutton(box, text="Use each material's own absorptivity",
                            variable=self.use_mat_eta, command=self._toggle_eta).pack(anchor="w", pady=(6, 0))
            r = ttk.Frame(box); r.pack(fill="x")
            ttk.Label(r, text="Absorptivity for all:").pack(side="left")
            self.eta_var = tk.StringVar(value="0.70")
            self.eta_entry = ttk.Entry(r, textvariable=self.eta_var, width=8, state="disabled")
            self.eta_entry.pack(side="left", padx=4)

            for title, rows in FIELDS:
                lf = ttk.LabelFrame(left, text=title, padding=6)
                lf.pack(fill="x", pady=(8, 0))
                for i, (label, key, default) in enumerate(rows):
                    ttk.Label(lf, text=label).grid(row=i, column=0, sticky="w")
                    v = tk.StringVar(value=default)
                    ttk.Entry(lf, textvariable=v, width=9).grid(row=i, column=1, padx=(8, 0), pady=1)
                    self.vars[key] = v
                    self.labels[key] = label

            bt = ttk.Frame(left); bt.pack(fill="x", pady=(12, 0))
            self.run_btn = ttk.Button(bt, text="RUN SIMULATION", command=self._start)
            self.run_btn.pack(side="left", fill="x", expand=True)
            self.stop_btn = ttk.Button(bt, text="Stop", command=self._stop, state="disabled")
            self.stop_btn.pack(side="left", padx=(6, 0))
            self.bar = ttk.Progressbar(left, maximum=1.0)
            self.bar.pack(fill="x", pady=(8, 0))
            self.status = tk.StringVar(value="Ready.")
            ttk.Label(left, textvariable=self.status, wraplength=300, foreground="#444").pack(anchor="w", pady=4)

        def _build_material_checks(self):
            for w in self.mat_frame.winfo_children():
                w.destroy()
            self.mat_vars = []
            for i, m in enumerate(self.materials):
                v = tk.BooleanVar(value=(i < 2))
                ttk.Checkbutton(self.mat_frame, text=f"{m['name']}  (η = {m['eta']})", variable=v).pack(anchor="w")
                self.mat_vars.append(v)

        def _reload_materials(self):
            self.materials, msg = load_materials()
            self._build_material_checks()
            self.status.set(msg or "Materials reloaded.")

        def _open_mat_file(self):
            load_materials()  # makes sure the file exists
            try:
                if sys.platform.startswith("win"):
                    os.startfile(MAT_FILE)  # noqa
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", MAT_FILE])
                else:
                    subprocess.Popen(["xdg-open", MAT_FILE])
            except Exception:
                messagebox.showinfo("materials.json", f"Open this file in any text editor:\n{MAT_FILE}")
            self.status.set("Edit and save the file, then press Reload.")

        def _toggle_eta(self):
            self.eta_entry.configure(state="disabled" if self.use_mat_eta.get() else "normal")

        # ---------- right panel ----------
        def _build_right(self):
            nb = ttk.Notebook(self)
            nb.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=8)

            # results table
            tab = ttk.Frame(nb); nb.add(tab, text="Results")
            self.tree = ttk.Treeview(tab, columns=[c for c, _ in TABLE_COLS], show="headings", height=8)
            for c, w in TABLE_COLS:
                self.tree.heading(c, text=c)
                self.tree.column(c, width=w, anchor="center")
            self.tree.pack(fill="x", padx=6, pady=6)
            ttk.Button(tab, text="Export results to CSV", command=self._export_csv).pack(anchor="w", padx=6)
            notes = ("How to read this:  Joined = melt reaches the lower sheet but not through it.  "
                     "Burn-through = melt reaches the bottom surface.  Not joined = melt stays in the top sheet.\n"
                     "Peak temperatures above the boiling point are not physical (conduction-only model, "
                     "no evaporation/keyhole).  Use the plot tabs; the toolbar under each plot saves it as PNG.")
            ttk.Label(tab, text=notes, wraplength=860, foreground="#444").pack(anchor="w", padx=6, pady=10)

            self.figs = {}
            for key, title in (("top", "Top-surface contours"), ("section", "Cross-section"),
                               ("cycles", "Thermal cycles"), ("compare", "Comparison")):
                fr = ttk.Frame(nb); nb.add(fr, text=title)
                fig = Figure(figsize=(9, 6), layout="constrained")
                canvas = FigureCanvasTkAgg(fig, master=fr)
                NavigationToolbar2Tk(canvas, fr).update()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                self.figs[key] = (fig, canvas)

        def _redraw(self):
            draw_contours(self.figs["top"][0], self.results, "top")
            draw_contours(self.figs["section"][0], self.results, "section")
            draw_cycles(self.figs["cycles"][0], self.results)
            draw_compare(self.figs["compare"][0], self.results)
            for fig, canvas in self.figs.values():
                canvas.draw_idle()
            self.tree.delete(*self.tree.get_children())
            for name, r in self.results.items():
                self.tree.insert("", "end", values=table_row(name, r))

        # ---------- running ----------
        def _read_inputs(self):
            vals = {}
            for key, var in self.vars.items():
                try:
                    vals[key] = float(var.get())
                except ValueError:
                    raise ValueError(f"'{self.labels[key]}' must be a number")
            base = build_base(vals)
            eta = None
            if not self.use_mat_eta.get():
                try:
                    eta = float(self.eta_var.get())
                except ValueError:
                    raise ValueError("Absorptivity must be a number")
                if not 0 < eta <= 1:
                    raise ValueError("Absorptivity must be between 0 and 1")
            chosen = [m for m, v in zip(self.materials, self.mat_vars) if v.get()]
            if not chosen:
                raise ValueError("Tick at least one material")
            return [(m["name"], params_for(base, m, eta)) for m in chosen]

        def _start(self):
            try:
                jobs = self._read_inputs()
            except ValueError as e:
                messagebox.showerror("Check your inputs", str(e))
                return
            self.results = {}
            self._redraw()
            self.stop_evt.clear()
            self.run_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.bar["value"] = 0
            self.worker = threading.Thread(target=self._work, args=(jobs,), daemon=True)
            self.worker.start()

        def _stop(self):
            self.stop_evt.set()
            self.status.set("Stopping...")

        def _work(self, jobs):
            try:
                for idx, (name, p) in enumerate(jobs):
                    dt = S.stable_dt(p)
                    n = int(np.ceil(((p.weld_end - p.weld_start) / p.v + p.t_cool) / dt))
                    self.q.put(("status", f"Running {name} ({idx + 1}/{len(jobs)}): {n} time steps..."))

                    def cb(frac, idx=idx):
                        self.q.put(("progress", (idx + frac) / len(jobs)))
                        return self.stop_evt.is_set()

                    res = S.run(p, verbose=False, progress=cb)
                    if res is None:
                        self.q.put(("stopped", None))
                        return
                    self.q.put(("result", (name, res)))
                self.q.put(("finished", None))
            except Exception as e:
                self.q.put(("error", f"{type(e).__name__}: {e}"))

        def _poll(self):
            try:
                while True:
                    kind, data = self.q.get_nowait()
                    if kind == "status":
                        self.status.set(data)
                    elif kind == "progress":
                        self.bar["value"] = data
                    elif kind == "result":
                        self.results[data[0]] = data[1]
                        self._redraw()
                    elif kind in ("finished", "stopped", "error"):
                        self.run_btn.configure(state="normal")
                        self.stop_btn.configure(state="disabled")
                        if kind == "finished":
                            self.bar["value"] = 1
                            self.status.set("Done. Check the Results, contour, cycle and comparison tabs.")
                        elif kind == "stopped":
                            self.status.set("Stopped.")
                        else:
                            self.status.set("Error.")
                            messagebox.showerror("Simulation error", data)
            except queue.Empty:
                pass
            self.after(100, self._poll)

        def _export_csv(self):
            if not self.results:
                messagebox.showinfo("Nothing to export", "Run a simulation first.")
                return
            path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="weld_results.csv",
                                                filetypes=[("CSV", "*.csv")])
            if not path:
                return
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([c for c, _ in TABLE_COLS])
                for name, r in self.results.items():
                    w.writerow(table_row(name, r))
            self.status.set(f"Saved {path}")

    App().mainloop()


# ----------------------------------------------------------------------------
# Headless self-test (no window needed)
# ----------------------------------------------------------------------------
def selftest():
    mats, msg = load_materials()
    print("materials:", [m["name"] for m in mats], msg)
    vals = {k: float(d) for _, rows in FIELDS for _, k, d in rows}
    vals.update(dx=0.5, t_cool=0.6)
    base = build_base(vals)
    results = {}
    for m in mats:
        r = S.run(params_for(base, m), verbose=False)
        results[m["name"]] = r
        print(table_row(m["name"], r))
    for key, fn in (("top", lambda f: draw_contours(f, results, "top")),
                    ("section", lambda f: draw_contours(f, results, "section")),
                    ("cycles", lambda f: draw_cycles(f, results)),
                    ("compare", lambda f: draw_compare(f, results))):
        fig = Figure(figsize=(9, 6), layout="constrained")
        fn(fig)
        fig.savefig(os.path.join(HERE, f"selftest_{key}.png"), dpi=70)
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_gui()
