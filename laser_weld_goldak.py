"""
Transient 3D finite-difference simulation of LASER LAP WELDING
with Goldak's double-ellipsoidal moving heat source.

Material  : AISI 304 stainless steel (temperature-dependent k and cp,
            latent heat through the apparent-heat-capacity method)
Solver    : explicit finite difference on a uniform grid, stability checked
Geometry  : two stacked sheets (lap joint). Only the overlap region is meshed,
            and only HALF of it (symmetry plane along the weld centreline y = 0).

Coordinates (mm)
    x : along the weld path
    y : across the weld, y = 0 is the weld centreline (symmetry plane)
    z : through thickness, z = 0 is the top surface (laser side)
        interface between the sheets is at z = t1

Link with the CAD model
    W  = overlap_width / 2   (weld runs along the centre of the overlap)
    weld_start, weld_end     = weld path coordinates from the CAD drawing
    focus_offset             = focus position below the top surface (mm)

Outputs (saved in the working folder)
    fig1_contours.png   top-surface and cross-section temperature contours
    fig2_thermal_cycles.png
    fig3_sweep.png      parametric sweep + optimisation
    fig4_convergence.png

Run:  python laser_weld_goldak.py
"""
import time
from dataclasses import dataclass, replace

import numpy as np


def _plt():
    """Lazy pyplot import (script mode only, saves PNGs without opening windows)."""
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    return plt

SIGMA = 5.670374e-8  # Stefan-Boltzmann constant, W/m2K4


# ----------------------------------------------------------------------------
# 1. INPUT PARAMETERS
# ----------------------------------------------------------------------------
@dataclass
class Params:
    # geometry (mm)
    L: float = 30.0            # block length along the weld
    W: float = 10.0            # half width (= overlap / 2)
    t1: float = 1.5            # top sheet thickness
    t2: float = 1.5            # bottom sheet thickness
    weld_start: float = 5.0    # weld path start x
    weld_end: float = 25.0     # weld path end x
    dx: float = 0.25           # grid size (mm), isotropic

    # laser / Goldak source
    P: float = 2000.0          # laser power, W
    eta: float = 0.70          # absorptivity (process efficiency)
    v: float = 20.0            # welding speed, mm/s
    af: float = 1.0            # front length, mm
    ar: float = 3.0            # rear length, mm
    b: float = 1.5             # half width, mm
    c: float = 2.0             # depth, mm
    focus_offset: float = 0.0  # source centre depth below top surface, mm

    # material (AISI 304)
    rho: float = 7800.0        # kg/m3
    T_sol: float = 1400.0      # solidus, C
    T_liq: float = 1454.0      # liquidus, C
    L_fus: float = 2.7e5       # latent heat, J/kg
    name: str = "AISI 304"
    T_tab: tuple = (20, 500, 1000, 1400, 1500, 3000)            # C
    k_tab: tuple = (14.9, 20.0, 25.7, 30.0, 32.0, 36.0)         # W/mK
    cp_tab: tuple = (500, 560, 620, 680, 700, 720)              # J/kgK

    # environment
    T0: float = 20.0           # ambient / initial temperature, C
    h_conv: float = 15.0       # convection coefficient, W/m2K
    emissivity: float = 0.5

    # run control
    t_cool: float = 2.5        # extra cooling time after the laser stops, s
    T_haz: float = 600.0       # lower temperature used to define the HAZ edge, C


# ----------------------------------------------------------------------------
# 2. MATERIAL PROPERTIES (tables live in Params; default = AISI 304)
# ----------------------------------------------------------------------------
def k_of_T(T, p):
    return np.interp(T, p.T_tab, p.k_tab)


def cp_eff(T, p):
    """Specific heat + latent heat spread over the melting range."""
    cp = np.interp(T, p.T_tab, p.cp_tab)
    in_range = (T > p.T_sol) & (T < p.T_liq)
    return cp + in_range * p.L_fus / (p.T_liq - p.T_sol)


# ----------------------------------------------------------------------------
# 3. GOLDAK DOUBLE ELLIPSOID HEAT SOURCE (volumetric, W/m3)
# ----------------------------------------------------------------------------
def goldak(xs, ys, zs, x_src, p):
    """
    q_f = 6*sqrt(3)*f_f*Q / (a_f*b*c*pi*sqrt(pi)) * exp(-3x^2/a_f^2 -3y^2/b^2 -3z^2/c^2)   (x >= 0, ahead)
    q_r = same with f_r, a_r                                                                 (x <  0, behind)
    f_f = 2 a_f/(a_f+a_r),  f_r = 2 a_r/(a_f+a_r),  f_f + f_r = 2
    All lengths in metres here.
    """
    Q = p.P * p.eta
    af, ar, b, c = p.af * 1e-3, p.ar * 1e-3, p.b * 1e-3, p.c * 1e-3
    ff = 2 * af / (af + ar)
    fr = 2 * ar / (af + ar)
    pref = 6 * np.sqrt(3) * Q / (b * c * np.pi * np.sqrt(np.pi))

    xi = xs - x_src
    qx = np.where(
        xi >= 0,
        ff / af * np.exp(-3 * xi**2 / af**2),
        fr / ar * np.exp(-3 * xi**2 / ar**2),
    )
    qy = np.exp(-3 * ys**2 / b**2)
    qz = np.exp(-3 * (zs - p.focus_offset * 1e-3) ** 2 / c**2)
    return pref * qx[:, None, None] * qy[None, :, None] * qz[None, None, :]


# ----------------------------------------------------------------------------
# 4. FINITE-DIFFERENCE OPERATORS
# ----------------------------------------------------------------------------
def div_k_grad(T, k, dx):
    """div(k grad T) with face-averaged k; zero flux through all outer faces."""
    out = np.zeros_like(T)
    for ax in range(3):
        Tm = np.moveaxis(T, ax, 0)
        km = np.moveaxis(k, ax, 0)
        flux = 0.5 * (km[1:] + km[:-1]) * (Tm[1:] - Tm[:-1]) / dx
        d = np.zeros_like(Tm)
        d[:-1] += flux
        d[1:] -= flux
        out += np.moveaxis(d, 0, ax) / dx
    return out


def surface_sink(T, p, dx):
    """Convection + radiation loss (W/m3) applied to the cells on every exposed face.
    The y = 0 plane is a symmetry plane, so it is NOT exposed."""
    Tk, T0k = T + 273.15, p.T0 + 273.15
    loss = p.h_conv * (T - p.T0) + p.emissivity * SIGMA * (Tk**4 - T0k**4)
    s = np.zeros_like(T)
    s[:, :, 0] += loss[:, :, 0] / dx     # top (laser side)
    s[:, :, -1] += loss[:, :, -1] / dx   # bottom
    s[0, :, :] += loss[0, :, :] / dx     # x = 0 end
    s[-1, :, :] += loss[-1, :, :] / dx   # x = L end
    s[:, -1, :] += loss[:, -1, :] / dx   # y = W side
    return s


# ----------------------------------------------------------------------------
# 5. STABLE TIME STEP
# ----------------------------------------------------------------------------
def stable_dt(p, safety=0.8):
    """Explicit 3D stability limit:  dt <= dx^2 / (6 alpha_max)."""
    Tg = np.linspace(20, 3000, 600)
    alpha_max = np.max(k_of_T(Tg, p) / (p.rho * cp_eff(Tg, p)))
    dx = p.dx * 1e-3
    return safety * dx**2 / (6 * alpha_max)


# ----------------------------------------------------------------------------
# 6. MAIN SOLVER
# ----------------------------------------------------------------------------
def run(p, verbose=True, progress=None):
    dx = p.dx * 1e-3
    H = p.t1 + p.t2
    nx, ny, nz = round(p.L / p.dx), round(p.W / p.dx), round(H / p.dx)
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dx
    zs = (np.arange(nz) + 0.5) * dx

    v = p.v * 1e-3
    t_weld = (p.weld_end - p.weld_start) / p.v
    t_total = t_weld + p.t_cool
    dt = stable_dt(p)
    n_steps = int(np.ceil(t_total / dt))
    if verbose:
        print(f"  grid {nx}x{ny}x{nz} = {nx*ny*nz} cells, dt = {dt*1e3:.3f} ms, "
              f"steps = {n_steps}, source moves {v*dt*1e3:.4f} mm/step")

    T = np.full((nx, ny, nz), p.T0)
    Tpeak = T.copy()

    # probe points (mm): mid-weld, various positions
    xm = 0.5 * (p.weld_start + p.weld_end)
    iz_int = max(0, round(p.t1 / p.dx - 0.5))
    probes = {
        "top, y=0 (centreline)": (xm, 0.0, 0.0),
        "interface, y=0": (xm, 0.0, p.t1 - 0.5 * p.dx),
        "top, y=2 mm": (xm, 2.0, 0.0),
        "top, y=4 mm": (xm, 4.0, 0.0),
    }
    pidx = {n: (min(nx - 1, int(x / p.dx)), min(ny - 1, int(y / p.dx)),
                min(nz - 1, int(z / p.dx))) for n, (x, y, z) in probes.items()}
    hist = {n: [] for n in probes}
    times, xsrc_hist, metrics = [], [], []
    snap = None

    t0 = time.time()
    for n in range(n_steps):
        t = n * dt
        if progress is not None and n % 25 == 0 and progress(n / n_steps):
            return None  # cancelled
        laser_on = t <= t_weld
        x_src = (p.weld_start * 1e-3 + v * t) if laser_on else None

        k = k_of_T(T, p)
        rcp = p.rho * cp_eff(T, p)
        rhs = div_k_grad(T, k, dx) - surface_sink(T, p, dx)
        if laser_on:
            rhs += goldak(xs, ys, zs, x_src, p)
        T = T + dt * rhs / rcp
        np.maximum(Tpeak, T, out=Tpeak)

        # ---- record ----
        times.append(t)
        xsrc_hist.append(x_src * 1e3 if laser_on else np.nan)
        for name, (i, j, kk) in pidx.items():
            hist[name].append(T[i, j, kk])
        mask = T >= p.T_liq
        if mask.any():
            ii = np.where(mask.any(axis=(1, 2)))[0]
            jj = np.where(mask.any(axis=(0, 2)))[0]
            kk_ = np.where(mask.any(axis=(0, 1)))[0]
            metrics.append(((ii.max() - ii.min() + 1) * p.dx,
                            2 * (jj.max() + 1) * p.dx,       # x2: half model
                            (kk_.max() + 1) * p.dx))
        else:
            metrics.append((0.0, 0.0, 0.0))

        # snapshot when the laser is at mid-weld
        if snap is None and laser_on and x_src * 1e3 >= xm:
            snap = dict(T=T.copy(), ix=min(nx - 1, int(x_src / dx)), t=t)

        if verbose and n % max(1, n_steps // 5) == 0:
            print(f"    step {n:5d}/{n_steps}  t = {t:.3f} s  Tmax = {T.max():.0f} C")

    times = np.array(times)
    xsrc_hist = np.array(xsrc_hist)
    metrics = np.array(metrics)
    hist = {n: np.array(h) for n, h in hist.items()}

    # quasi-steady pool size: take the window where the laser is >6 mm past the start
    win = (xsrc_hist >= p.weld_start + 6.0)
    if not win.any():
        win = np.isfinite(xsrc_hist)
    length, width, depth = metrics[win].max(axis=0)

    # HAZ width at the top surface (mid-weld), from the peak-temperature field
    ix_m = min(nx - 1, int(xm / p.dx))
    top = Tpeak[ix_m, :, 0]
    fz = np.where(top >= p.T_liq)[0]
    hz = np.where(top >= p.T_haz)[0]
    fz_half = (fz.max() + 1) * p.dx if fz.size else 0.0
    hz_half = (hz.max() + 1) * p.dx if hz.size else 0.0

    # cooling time t8/5 at the centreline probe
    t85 = cooling_time(times, hist["top, y=0 (centreline)"], 800.0, 500.0)

    res = dict(
        p=p, times=times, hist=hist, snap=snap, Tpeak=Tpeak, xs=xs * 1e3, ys=ys * 1e3, zs=zs * 1e3,
        length=length, width=width, depth=depth,
        T_max=Tpeak.max(), haz_width=2 * (hz_half - fz_half), fz_width=2 * fz_half,
        t85=t85, dt=dt, runtime=time.time() - t0,
        full_pen=depth >= H - 0.5 * p.dx,
    )
    if verbose:
        print(f"  -> pool L x W x D = {length:.2f} x {width:.2f} x {depth:.2f} mm, "
              f"Tmax = {res['T_max']:.0f} C, t8/5 = {t85:.2f} s, runtime {res['runtime']:.1f} s")
    return res


def cooling_time(t, T, Ta, Tb):
    """Time to cool from Ta to Tb after the peak (NaN if never happens)."""
    ipk = int(np.argmax(T))
    if T[ipk] < Ta:
        return np.nan
    Tc, tc = T[ipk:], t[ipk:]
    ia = np.where(Tc <= Ta)[0]
    ib = np.where(Tc <= Tb)[0]
    if ia.size == 0 or ib.size == 0:
        return np.nan
    return tc[ib[0]] - tc[ia[0]]


# ----------------------------------------------------------------------------
# 7. VALIDATION: ROSENTHAL (point source on a semi-infinite body)
# ----------------------------------------------------------------------------
def rosenthal_peak(p, y_mm, z_mm, k=25.0, cp=600.0):
    """Peak temperature at (y, z) from Rosenthal's thick-plate solution
       T - T0 = Q/(2 pi k R) * exp(-v (xi + R) / (2 alpha)).
    Only a rough check: a 3 mm lap joint is not 'thick', and the real source is
    distributed, so compare far from the pool (y >= 3 mm)."""
    Q, v = p.P * p.eta, p.v * 1e-3
    alpha = k / (p.rho * cp)
    xi = np.linspace(-0.05, 0.05, 20001)
    R = np.sqrt(xi**2 + (y_mm * 1e-3) ** 2 + (z_mm * 1e-3) ** 2)
    T = p.T0 + Q / (2 * np.pi * k * R) * np.exp(-v * (xi + R) / (2 * alpha))
    return T.max()


# ----------------------------------------------------------------------------
# 8. PLOTS
# ----------------------------------------------------------------------------
def plot_contours(res, fname="fig1_contours.png"):
    plt = _plt()
    p, snap = res["p"], res["snap"]
    T, ix = snap["T"], snap["ix"]
    xs, ys, zs = res["xs"], res["ys"], res["zs"]
    levels = np.linspace(p.T0, max(1800, T.max()), 25)

    top = T[:, :, 0].T                       # (ny, nx)
    top_full = np.vstack([top[::-1], top])   # mirror across y = 0
    y_full = np.concatenate([-ys[::-1], ys])
    cs = T[ix, :, :].T                       # (nz, ny)
    cs_full = np.hstack([cs[:, ::-1], cs])

    fig, ax = plt.subplots(2, 1, figsize=(9, 8))
    c0 = ax[0].contourf(xs, y_full, top_full, levels=levels, cmap="inferno")
    ax[0].contour(xs, y_full, top_full, levels=[p.T_liq], colors="cyan", linewidths=1.5)
    ax[0].set_title(f"Top-surface temperature at t = {snap['t']:.2f} s "
                    f"(cyan line = liquidus {p.T_liq:.0f} C)")
    ax[0].set_xlabel("x along weld (mm)"); ax[0].set_ylabel("y (mm)")
    ax[0].set_aspect("equal")
    fig.colorbar(c0, ax=ax[0], label="T (C)")

    c1 = ax[1].contourf(y_full, zs, cs_full, levels=levels, cmap="inferno")
    ax[1].contour(y_full, zs, cs_full, levels=[p.T_liq], colors="cyan", linewidths=1.5)
    ax[1].axhline(p.t1, color="white", ls="--", lw=1)
    ax[1].text(y_full[0] + 0.2, p.t1 - 0.1, "sheet interface", color="white", fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_title(f"Cross-section at x = {xs[ix]:.1f} mm (under the laser)")
    ax[1].set_xlabel("y (mm)"); ax[1].set_ylabel("z depth (mm)")
    ax[1].set_aspect("equal")
    fig.colorbar(c1, ax=ax[1], label="T (C)")
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def plot_cycles(res, fname="fig2_thermal_cycles.png"):
    plt = _plt()
    p = res["p"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, h in res["hist"].items():
        ax.plot(res["times"], h, label=name)
    ax.axhline(p.T_liq, color="k", ls=":", lw=1)
    ax.text(res["times"][-1] * 0.7, p.T_liq + 20, "liquidus", fontsize=8)
    ax.set_xlabel("time (s)"); ax.set_ylabel("temperature (C)")
    ax.set_title("Thermal cycles at mid-weld"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------
# 9. PARAMETRIC SWEEP + OPTIMISATION
# ----------------------------------------------------------------------------
def sweep_and_optimise(base, powers, speeds, dx=0.25, fname="fig3_sweep.png"):
    plt = _plt()
    """Run every (P, v) pair. Target: melt reaches at least 0.5 mm into the
    lower sheet (depth >= t1 + 0.5 mm) WITHOUT burning through.
    Among feasible sets pick the lowest heat input P*eta/v, then narrowest weld."""
    H = base.t1 + base.t2
    target = base.t1 + 0.5
    D = np.zeros((len(powers), len(speeds)))
    Wd = np.zeros_like(D)
    rows = []
    for i, P in enumerate(powers):
        for j, v in enumerate(speeds):
            r = run(replace(base, P=P, v=v, dx=dx, t_cool=0.3), verbose=False)
            D[i, j], Wd[i, j] = r["depth"], r["width"]
            rows.append((P, v, P * base.eta / v, r["depth"], r["width"], r["length"], r["T_max"], r["full_pen"]))
            print(f"  P={P:6.0f} W  v={v:5.1f} mm/s  HI={P*base.eta/v:5.1f} J/mm  "
                  f"depth={r['depth']:.2f}  width={r['width']:.2f}  burn-through={r['full_pen']}")

    feasible = [r for r in rows if r[3] >= target and not r[7]]
    if feasible:
        best = min(feasible, key=lambda r: (round(r[2], 1), r[4]))
    else:
        best = min(rows, key=lambda r: abs(r[3] - target))
        print("  (no set met the target exactly; showing the closest one)")
    print(f"\n  OPTIMUM: P = {best[0]:.0f} W, v = {best[1]:.1f} mm/s, heat input = {best[2]:.1f} J/mm, "
          f"depth = {best[3]:.2f} mm, width = {best[4]:.2f} mm")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for a, Z, ttl in ((ax[0], D, "Pool depth (mm)"), (ax[1], Wd, "Pool width (mm)")):
        m = a.imshow(Z, origin="lower", aspect="auto", cmap="viridis")
        a.set_xticks(range(len(speeds))); a.set_xticklabels(speeds)
        a.set_yticks(range(len(powers))); a.set_yticklabels(powers)
        a.set_xlabel("speed (mm/s)"); a.set_ylabel("power (W)"); a.set_title(ttl)
        for i in range(len(powers)):
            for j in range(len(speeds)):
                a.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
        fig.colorbar(m, ax=a)
    bi, bj = powers.index(best[0]), speeds.index(best[1])
    for a in ax:
        a.plot(bj, bi, "r*", ms=18)
    fig.suptitle(f"Sweep (target depth >= {target:.2f} mm, no burn-through at {H:.1f} mm; red star = optimum)")
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)
    return best, rows


# ----------------------------------------------------------------------------
# 10. GRID CONVERGENCE
# ----------------------------------------------------------------------------
def convergence(base, dxs=(0.5, 0.35, 0.25, 0.2), fname="fig4_convergence.png"):
    plt = _plt()
    out = []
    for d in dxs:
        r = run(replace(base, dx=d), verbose=False)
        out.append((d, r["depth"], r["width"], r["T_max"]))
        print(f"  dx = {d:.2f} mm -> depth {r['depth']:.2f}  width {r['width']:.2f}  Tmax {r['T_max']:.0f}")
    a = np.array(out)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for i, (ttl, unit) in enumerate((("Pool depth", "mm"), ("Pool width", "mm"), ("Peak temperature", "C"))):
        ax[i].plot(a[:, 0], a[:, i + 1], "o-")
        ax[i].set_xlabel("grid size dx (mm)"); ax[i].set_ylabel(f"{ttl} ({unit})")
        ax[i].grid(alpha=0.3); ax[i].invert_xaxis()
    fig.suptitle("Grid convergence"); fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# 11. MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    RUN_SWEEP = True
    RUN_CONVERGENCE = True

    base = Params()
    print("=== BASE RUN ===")
    res = run(base)
    plot_contours(res)
    plot_cycles(res)

    print("\n=== RESULTS (base case) ===")
    print(f"  Melt pool length : {res['length']:.2f} mm")
    print(f"  Melt pool width  : {res['width']:.2f} mm")
    print(f"  Melt pool depth  : {res['depth']:.2f} mm   "
          f"({'FULL penetration' if res['full_pen'] else 'partial penetration'}, "
          f"sheet interface at {base.t1:.1f} mm)")
    print(f"  Peak temperature : {res['T_max']:.0f} C")
    print(f"  Fusion zone width (top, mid-weld): {res['fz_width']:.2f} mm")
    print(f"  HAZ width each side (>= {base.T_haz:.0f} C): {res['haz_width']/2:.2f} mm")
    print(f"  Cooling time t8/5 (centreline): {res['t85']:.2f} s")

    print("\n=== VALIDATION vs ROSENTHAL (peak temperature, rough check) ===")
    y_chk, z_chk = 4.0, 0.0
    T_fd = res["Tpeak"][int(0.5 * (base.weld_start + base.weld_end) / base.dx),
                        int(y_chk / base.dx), int(z_chk / base.dx)]
    T_ro = rosenthal_peak(base, y_chk, max(z_chk, 0.05))
    print(f"  at y = {y_chk} mm: FD = {T_fd:.0f} C, Rosenthal = {T_ro:.0f} C, "
          f"difference = {100*(T_fd-T_ro)/T_ro:+.0f} %")
    print("  (FD is expected to read higher: the 3 mm block has an insulated-ish bottom and "
          "finite width, whereas Rosenthal assumes a semi-infinite body.)")

    if RUN_CONVERGENCE:
        print("\n=== GRID CONVERGENCE ===")
        convergence(base)

    if RUN_SWEEP:
        print("\n=== PARAMETRIC SWEEP / OPTIMISATION (power x speed) ===")
        sweep_and_optimise(base, powers=[1500.0, 2000.0, 2500.0, 3000.0], speeds=[10.0, 20.0, 30.0])

    print("\nDone. Figures saved: fig1_contours.png, fig2_thermal_cycles.png, "
          "fig3_sweep.png, fig4_convergence.png")
