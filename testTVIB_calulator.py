import numpy as np

# Spectroscopic constants for H2 X1Σg+
omega_e = 4401.21       # cm^-1
omega_exe = 121.33      # cm^-1

# Number of vibrational levels needed (must match qX2d rows)
Nv = qX2d.shape[0]

v = np.arange(Nv)

# Term values in cm^-1
G_cm = omega_e*(v + 0.5) - omega_exe*(v + 0.5)**2

# Convert to eV (1 eV = 8065.544 cm^-1)
Gx_eV = G_cm / 8065.544

print("G(v) in eV:")
print(Gx_eV)

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ============================================================
# USER INPUTS
# ============================================================

SPECTRUM_FILE = "/Users/Elliot/Library/CloudStorage/OneDrive-DublinCityUniversity/4th Year/plasma_data/corrected_data/H116_HRC21401__0__12-12-39-514_cal.txt"
Q_X_TO_D = "q_X_to_d.csv"
Q_D_TO_A = "q_d_to_a.csv"

BASELINE_WINDOW = 301
PEAK_HALF_WIDTH_NM = 1.5   # gives 3 nm total width

band_regions_nm = {
    (0, 0): (601.5, 605.0),
    (1, 1): (611.5, 615.0),
    (2, 2): (622.5, 626.0),
    (3, 3): (632.5, 636.0),
}

FULCHER_XLIM = (598.0, 637.0)
kB_eV = 8.617333262145e-5


# ============================================================
# HELPERS
# ============================================================

def find_file(patterns):
    here = os.getcwd()
    for p in patterns:
        matches = glob.glob(os.path.join(here, p))
        if matches:
            return matches[0]
    return None

def baseline_correct_rolling_min(y, window=301):
    window = int(window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    shape = (len(y), window)
    strides = (ypad.strides[0], ypad.strides[0])
    Y = np.lib.stride_tricks.as_strided(ypad, shape=shape, strides=strides)
    baseline = np.min(Y, axis=1)
    return y - baseline, baseline

def integrate_peak_window(lam, y, peak_lam, half_width_nm, positive_only=True):
    left = peak_lam - half_width_nm
    right = peak_lam + half_width_nm
    m = (lam >= left) & (lam <= right)
    if not np.any(m):
        return np.nan, left, right
    yy = y[m].copy()
    if positive_only:
        yy = np.clip(yy, 0, None)
    area = float(np.trapz(yy, lam[m]))
    return area, left, right

def boltzmann_pop(G_eV, T):
    G = np.asarray(G_eV, dtype=float)
    w = np.exp(-(G - np.min(G)) / (kB_eV * T))
    return w / np.sum(w)

def predict_I(qX2d, qd2a, Gx_eV, T):
    Pvx = boltzmann_pop(Gx_eV, T)
    Nd = qX2d.T @ Pvx
    return Nd[:, None] * qd2a

def best_scale(I_pred, I_meas):
    mask = np.isfinite(I_meas)
    x = I_pred[mask].ravel()
    y = I_meas[mask].ravel()
    denom = x @ x
    return (x @ y) / denom if denom > 0 else 0.0


# ============================================================
# LOAD FC MATRICES
# ============================================================

Q_X_TO_D = find_file([Q_X_TO_D, "*X*to*d*.csv", "*x*to*d*.csv"])
Q_D_TO_A = find_file([Q_D_TO_A, "*d*to*a*.csv", "*D*to*A*.csv"])

qX2d_df = pd.read_csv(Q_X_TO_D)
qd2a_df = pd.read_csv(Q_D_TO_A)

qX2d = qX2d_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
qd2a = qd2a_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)

Nvd = min(qX2d.shape[1], qd2a.shape[0])
qX2d = qX2d[:, :Nvd]
qd2a = qd2a[:Nvd, :]

# X-state G terms
omega_e = 4401.21
omega_exe = 121.33

Nv = qX2d.shape[0]
v = np.arange(Nv)
G_cm = omega_e*(v + 0.5) - omega_exe*(v + 0.5)**2
Gx_eV = G_cm / 8065.544


# ============================================================
# LOAD SPECTRUM
# ============================================================

df = pd.read_csv(SPECTRUM_FILE)
lam = df["wavelength_nm"].to_numpy(float)
I = df["cal_intensity"].to_numpy(float)

I_bc, baseline = baseline_correct_rolling_min(I, window=BASELINE_WINDOW)


# ============================================================
# FIND PEAKS IN RAW BASELINE-CORRECTED SPECTRUM
# ============================================================

band_peaks = {}
band_limits = {}
band_areas = {}

for band, (lo, hi) in band_regions_nm.items():
    m = (lam >= lo) & (lam <= hi)
    if not np.any(m):
        raise ValueError(f"No points in region {lo}-{hi} nm for band {band}")

    lam_r = lam[m]
    I_r = I_bc[m]

    # actual peak from baseline-corrected spectrum
    j = int(np.argmax(I_r))
    peak_lam = float(lam_r[j])
    band_peaks[band] = peak_lam

    area, left, right = integrate_peak_window(
        lam, I_bc, peak_lam, PEAK_HALF_WIDTH_NM, positive_only=True
    )
    band_areas[band] = area
    band_limits[band] = (left, right)

print("\nDetected peak centres + integrated areas:")
for band in sorted(band_peaks.keys()):
    left, right = band_limits[band]
    print(f"{band}: peak = {band_peaks[band]:.3f} nm, limits = ({left:.3f}, {right:.3f}) nm, area = {band_areas[band]:.6g}")


# ============================================================
# BUILD I_meas
# ============================================================

nvd, nva = qd2a.shape
I_meas = np.full((nvd, nva), np.nan, float)

for (vp, va), area in band_areas.items():
    if vp < nvd and va < nva:
        I_meas[vp, va] = area


# ============================================================
# FIT Tvib
# ============================================================

def objective(T):
    I_pred = predict_I(qX2d, qd2a, Gx_eV, T)
    a = best_scale(I_pred, I_meas)
    mask = np.isfinite(I_meas)
    e = a * I_pred[mask] - I_meas[mask]
    return float(np.sum(e * e))

res = minimize_scalar(objective, bounds=(300, 8000), method="bounded")
Tvib = float(res.x)
print(f"\nBest-fit Tvib = {Tvib:.1f} K")


# ============================================================
# PLOT 1: FULCHER REGION WITH SELECTED PEAKS
# ============================================================

plt.figure(figsize=(11, 5.5))
plt.plot(lam, I_bc, color="black", lw=1.2, label="Baseline-corrected spectrum")

colors = {
    (0, 0): "#5ea4ff",
    (1, 1): "#86ffb0",
    (2, 2): "#ffe88a",
    (3, 3): "#ff9090",
}

for band, (lo, hi) in band_regions_nm.items():
    peak_lam = band_peaks[band]
    left, right = band_limits[band]
    c = colors.get(band, "lightgray")

    plt.axvspan(lo, hi, color=c, alpha=0.10)
    plt.axvline(peak_lam, color="crimson", ls="--", lw=1.2)
    plt.axvline(left, color="0.5", ls=":", lw=1.0)
    plt.axvline(right, color="0.5", ls=":", lw=1.0)

    idx = np.argmin(np.abs(lam - peak_lam))
    peak_y = I_bc[idx]
    plt.plot(peak_lam, peak_y, "o", color="crimson", ms=6)

    plt.annotate(
        f"{band[0]}-{band[1]}\n{peak_lam:.3f} nm",
        xy=(peak_lam, peak_y),
        xytext=(peak_lam + 0.2, peak_y),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.9)
    )

plt.xlim(*FULCHER_XLIM)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (Arbitrary Units)")
plt.title("Fulcher Region with Selected Peak Positions and Area Window")
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# PLOT 2: BASELINE SUBTRACTION
# ============================================================

m = (lam >= FULCHER_XLIM[0]) & (lam <= FULCHER_XLIM[1])

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

axes[0].plot(lam[m], I[m], color="black", lw=1.2, label="Raw calibrated spectrum")
axes[0].plot(lam[m], baseline[m], color="tab:purple", ls="--", lw=1.5, label="Estimated baseline")
axes[0].set_ylabel("Intensity / a.u.")
axes[0].set_title("Baseline estimation in the Fulcher region")
axes[0].legend()
axes[0].grid(False)

axes[1].plot(lam[m], I_bc[m], color="tab:green", lw=1.2, label="Baseline-corrected spectrum")
axes[1].axhline(0, color="0.4", ls=":", lw=1.0)
axes[1].set_xlabel("Wavelength / nm")
axes[1].set_ylabel("Intensity / a.u.")
axes[1].set_title("Baseline-corrected Fulcher region")
axes[1].legend()
axes[1].grid(False)

plt.tight_layout()
plt.show()


# ============================================================
# PLOT 3: AREA DIAGRAMS
# ============================================================

fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=False)

for ax, band in zip(axes, sorted(band_peaks.keys())):
    peak_lam = band_peaks[band]
    left, right = band_limits[band]

    pad = 1.0
    m_local = (lam >= left - pad) & (lam <= right + pad)
    m_int = (lam >= left) & (lam <= right)

    ax.plot(lam[m_local], I_bc[m_local], color="black", lw=1.2, label="Baseline-corrected spectrum")
    ax.fill_between(
        lam[m_int],
        np.clip(I_bc[m_int], 0, None),
        0,
        color="tab:orange",
        alpha=0.45,
        label=f"Integrated area = {band_areas[band]:.3g}"
    )

    ax.axvline(peak_lam, color="crimson", ls="--", lw=1.2, label="Selected peak")
    ax.axvline(left, color="0.5", ls=":", lw=1.0)
    ax.axvline(right, color="0.5", ls=":", lw=1.0)

    ax.set_title(f"Band {band[0]}-{band[1]}")
    ax.set_xlabel("Wavelength / nm")
    ax.set_ylabel("Intensity / a.u.")
    ax.grid(False)
    ax.legend(loc="upper right")

plt.tight_layout()
plt.show()


# ============================================================
# PLOT 4: BOLTZMANN-STYLE PLOT (relative to 0-0 band)
# This is only a visualisation.
# It does NOT change the fitted Tvib.
# ============================================================

diag_bands = [(0, 0), (1, 1), (2, 2), (3, 3)]

G_diag = np.array([Gx_eV[vp] for vp, va in diag_bands], dtype=float)
A_diag = np.array([band_areas[(vp, va)] for vp, va in diag_bands], dtype=float)

# require positive finite areas
mask = np.isfinite(A_diag) & (A_diag > 0)

G_plot = G_diag[mask]
A_plot = A_diag[mask]
labels = [f"{vp}-{va}" for (vp, va), keep in zip(diag_bands, mask) if keep]

# use 0-0 band as reference
A00 = band_areas[(0, 0)]
if not np.isfinite(A00) or A00 <= 0:
    raise ValueError("The (0-0) band area must be positive to build the relative Boltzmann plot.")

ln_rel = np.log(A_plot / A00)

# linear fit for display only
p = np.polyfit(G_plot, ln_rel, 1)
xfit = np.linspace(G_plot.min(), G_plot.max(), 200)
yfit = p[0] * xfit + p[1]

plt.figure(figsize=(7, 5))
plt.scatter(G_plot, ln_rel, color="black", s=30, label="Measured diagonal bands")
plt.plot(xfit, yfit, color="black", lw=0.5, label="Linear Fit")

for x, y, lab in zip(G_plot, ln_rel, labels):
    plt.text(x + 0.003, y, lab)

plt.xlabel("Vibrational Energy (eV)")
plt.ylabel(r"$\ln(A/A_{00})$")
plt.title("Boltzmann Trend for Fulcher Diagonal Bands")
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()