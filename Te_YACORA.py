import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# FILE PATHS
# =========================================================
FILE_N4 = "/Users/Elliot/Downloads/beta_n4.dat"         # Yacora state n=4
FILE_N5 = "/Users/Elliot/Downloads/gamma_n5.dat"        # Yacora state n=5
EXCEL_FILE = "/Users/Elliot/Library/CloudStorage/OneDrive-DublinCityUniversity/4th Year/Final Year Project/Plasma Data.xlsx"

# =========================================================
# EXCEL COLUMN NAMES
# =========================================================
ID_COL = "Code"
POWER_COL = "Power (watts)"
PRESSURE_COL = "Pressure (mTorr)"
RATIO_COL = "Hgamma/Hbeta"

# =========================================================
# BALMER DATA FOR Hβ AND Hγ
# =========================================================
# wavelengths in nm
LAMBDA_HB = 486.133   # Hβ : 4 -> 2
LAMBDA_HG = 434.047   # Hγ : 5 -> 2

# Einstein A coefficients in s^-1
A42 = 8.419e6         # Hβ : 4 -> 2
A52 = 2.530e6         # Hγ : 5 -> 2

# =========================================================
# READ YACORA .dat FILES
# =========================================================
def read_yacora_popcoeff(filepath: str, pop_name: str) -> pd.DataFrame:
    """
    Reads Yacora population-coefficient .dat output of the form:
      Calculated Data:
      # ================
      #  T_e   T_n1   n_e   POP_COEFF
      ...
    """
    col_names = ["Te_eV", "Tn1_K", "ne_m3", pop_name]
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        comment="#",
        header=None,
        names=col_names,
        skiprows=4,
        engine="python"
    )
    return df

n4 = read_yacora_popcoeff(FILE_N4, "pc_n4")
n5 = read_yacora_popcoeff(FILE_N5, "pc_n5")

# Merge on common Te, Tn1, ne grid
model = pd.merge(n4, n5, on=["Te_eV", "Tn1_K", "ne_m3"], how="inner")

# =========================================================
# PHYSICALLY MOTIVATED CONVERSION:
# optically thin relative line intensity
#
# I_ul ∝ N_u * A_ul * h*nu ∝ N_u * A_ul / lambda
# where N_u is represented here by the model population coefficient
# up to the model's proportionality assumptions
# =========================================================
model["I_Hbeta_model"] = model["pc_n4"] * A42 / LAMBDA_HB
model["I_Hgamma_model"] = model["pc_n5"] * A52 / LAMBDA_HG

# Final uncalibrated model ratio
model["Hgamma_Hbeta_model"] = model["I_Hgamma_model"] / model["I_Hbeta_model"]

print("Model preview:")
print(
    model[["Te_eV", "Tn1_K", "ne_m3", "pc_n4", "pc_n5", "Hgamma_Hbeta_model"]]
    .head(10)
    .to_string(index=False)
)

# =========================================================
# READ EXCEL SHEET AND KEEP ONLY NUMERIC Hgamma/Hbeta ROWS
# =========================================================
meas = pd.read_excel(EXCEL_FILE)
meas.columns = meas.columns.astype(str).str.strip()

# keep only rows with a real code
meas = meas[meas[ID_COL].notna()].copy()
meas[ID_COL] = meas[ID_COL].astype(str).str.strip()

# convert useful columns
for col in [POWER_COL, PRESSURE_COL, RATIO_COL]:
    meas[col] = pd.to_numeric(meas[col], errors="coerce")

# keep only rows with a numeric Hgamma/Hbeta
meas_gamma = meas[meas[RATIO_COL].notna()].copy()

print("\nMeasured Hgamma/Hbeta rows:")
print(meas_gamma[[ID_COL, POWER_COL, PRESSURE_COL, RATIO_COL]].to_string(index=False))

# =========================================================
# SETTINGS
# =========================================================
selected_ne = [1e16, 5e16, 1e17]

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def find_intercept_te(te_vals, ratio_vals, measured_ratio):
    """
    Find Te where the curve crosses measured_ratio by linear interpolation.
    Returns NaN if there is no crossing.
    """
    te_vals = np.asarray(te_vals, dtype=float)
    ratio_vals = np.asarray(ratio_vals, dtype=float)

    diff = ratio_vals - measured_ratio

    # exact match
    exact = np.where(np.isclose(diff, 0, atol=1e-14))[0]
    if len(exact) > 0:
        return te_vals[exact[0]]

    # sign-change search
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            x1, x2 = te_vals[i], te_vals[i + 1]
            y1, y2 = ratio_vals[i], ratio_vals[i + 1]
            return x1 + (measured_ratio - y1) * (x2 - x1) / (y2 - y1)

    return np.nan


def nearest_curve_value(ratio_vals, measured_ratio):
    """
    Distance from a measured ratio to the nearest point on a model curve.
    """
    ratio_vals = np.asarray(ratio_vals, dtype=float)
    return np.min(np.abs(ratio_vals - measured_ratio))


def objective_scale(scale, model_curve, measured_ratios):
    """
    Objective function for fitting a single multiplicative scale factor
    to a model curve against all measured ratios.
    Each measured ratio is compared to the nearest point on the scaled curve.
    """
    scaled_curve = scale * model_curve
    residuals = [nearest_curve_value(scaled_curve, r) for r in measured_ratios]
    return np.sum(np.square(residuals))


def fit_scale_factor(model_curve, measured_ratios, scale_grid=None):
    """
    Fit a single multiplicative scale factor by brute-force search.
    This is robust and avoids requiring scipy.
    """
    if scale_grid is None:
        scale_grid = np.linspace(0.05, 2.5, 5000)

    obj_vals = np.array([objective_scale(s, model_curve, measured_ratios) for s in scale_grid])
    best_idx = np.argmin(obj_vals)
    return scale_grid[best_idx], obj_vals[best_idx]

# =========================================================
# FIT CALIBRATION FACTOR FOR EACH ne
# =========================================================
measured_ratios = meas_gamma[RATIO_COL].dropna().to_numpy()

calibration_rows = []

for ne_target in selected_ne:
    ne_actual = min(model["ne_m3"].unique(), key=lambda x: abs(x - ne_target))
    sub = model[model["ne_m3"] == ne_actual].sort_values("Te_eV").copy()

    best_scale, best_obj = fit_scale_factor(
        sub["Hgamma_Hbeta_model"].values,
        measured_ratios
    )

    calibration_rows.append({
        "ne_target": ne_target,
        "ne_used": ne_actual,
        "best_scale_factor": best_scale,
        "objective_value": best_obj
    })

calibration_df = pd.DataFrame(calibration_rows)

print("\nCalibration factors:")
print(calibration_df.to_string(index=False))

# attach calibrated curves back into model table
model["Hgamma_Hbeta_model_calibrated"] = np.nan

for _, row in calibration_df.iterrows():
    ne_actual = row["ne_used"]
    scale = row["best_scale_factor"]
    mask = np.isclose(model["ne_m3"], ne_actual)
    model.loc[mask, "Hgamma_Hbeta_model_calibrated"] = (
        model.loc[mask, "Hgamma_Hbeta_model"] * scale
    )

# =========================================================
# PLOT UNCALIBRATED MODEL + MEASURED RATIOS
# =========================================================
plt.figure(figsize=(8, 5))

for ne_target in selected_ne:
    ne_actual = min(model["ne_m3"].unique(), key=lambda x: abs(x - ne_target))
    sub = model[model["ne_m3"] == ne_actual].sort_values("Te_eV")

    plt.plot(
        sub["Te_eV"],
        sub["Hgamma_Hbeta_model"],
        marker="o",
        label=fr"$n_e={ne_actual:.1e}$"
    )

#for _, row in meas_gamma.iterrows():
#    plt.axhline(row[RATIO_COL], color="red", alpha=0.08, linewidth=1)

#avg_ratio = meas_gamma[RATIO_COL].mean()
#plt.axhline(
    #avg_ratio,
    #color="red",
    #linestyle="--",
    #linewidth=2,
    #label=fr"Mean measured $H\gamma/H\beta={avg_ratio:.4f}$"
#)

plt.xlabel(r"$T_e$ (eV)")
plt.ylabel(r"$H\gamma/H\beta$")
plt.title(r"Model $H\gamma/H\beta$")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# PLOT CALIBRATED MODEL + MEASURED RATIOS
# =========================================================
plt.figure(figsize=(8, 5))

for ne_target in selected_ne:
    ne_actual = min(model["ne_m3"].unique(), key=lambda x: abs(x - ne_target))
    sub = model[model["ne_m3"] == ne_actual].sort_values("Te_eV")

    scale = calibration_df.loc[np.isclose(calibration_df["ne_used"], ne_actual), "best_scale_factor"].iloc[0]

    plt.plot(
        sub["Te_eV"],
        sub["Hgamma_Hbeta_model_calibrated"],
        marker="o",
        label=fr"Calibrated $n_e={ne_actual:.1e}$, $s={scale:.3f}$"
    )



plt.xlabel(r"$T_e$ (eV)")
plt.ylabel(r"$H\gamma/H\beta$")
plt.title(r"Calibrated model $H\gamma/H\beta$")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# EXTRACT CALIBRATED INTERCEPTS
# =========================================================
results = []

for _, row in meas_gamma.iterrows():
    measured_ratio = row[RATIO_COL]

    for ne_target in selected_ne:
        ne_actual = min(model["ne_m3"].unique(), key=lambda x: abs(x - ne_target))
        sub = model[model["ne_m3"] == ne_actual].sort_values("Te_eV")

        te_intercept = find_intercept_te(
            sub["Te_eV"].values,
            sub["Hgamma_Hbeta_model_calibrated"].values,
            measured_ratio
        )

        scale = calibration_df.loc[np.isclose(calibration_df["ne_used"], ne_actual), "best_scale_factor"].iloc[0]

        results.append({
            "id": row[ID_COL],
            "power_W": row[POWER_COL],
            "pressure_mTorr": row[PRESSURE_COL],
            "measured_Hgamma_Hbeta": measured_ratio,
            "ne_target": ne_target,
            "ne_used": ne_actual,
            "scale_factor": scale,
            "Te_intercept_eV": te_intercept
        })

results_df = pd.DataFrame(results)

print("\nCalibrated intercept results:")
print(results_df.to_string(index=False))

results_df.to_csv("Hgamma_Hbeta_Yacora_calibrated_intercepts.csv", index=False)
print("\nSaved: Hgamma_Hbeta_Yacora_calibrated_intercepts.csv")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# FILE PATHS
# =========================================================
FILE_N4 = "/Users/Elliot/Downloads/beta_n4.dat"         # Yacora state n=4
FILE_N5 = "/Users/Elliot/Downloads/gamma_n5.dat"        # Yacora state n=5
EXCEL_FILE = "/Users/Elliot/Library/CloudStorage/OneDrive-DublinCityUniversity/4th Year/Final Year Project/Plasma Data.xlsx" # your Excel data sheet

# =========================================================
# EXCEL COLUMN NAMES
# =========================================================
ID_COL = "Code"
POWER_COL = "Power (watts)"
PRESSURE_COL = "Pressure (mTorr)"
RATIO_COL = "Hgamma/Hbeta"

# =========================================================
# BALMER DATA FOR Hβ AND Hγ
# =========================================================
# wavelengths in nm
LAMBDA_HB = 486.133   # Hβ : 4 -> 2
LAMBDA_HG = 434.047   # Hγ : 5 -> 2

# Einstein A coefficients in s^-1
# Hβ: 4 -> 2, Hγ: 5 -> 2
A42 = 8.419e6
A52 = 2.530e6

# =========================================================
# READ YACORA .dat FILES
# =========================================================
def read_yacora_popcoeff(filepath: str, pop_name: str) -> pd.DataFrame:
    """
    Reads Yacora population-coefficient .dat output of the form:
      Calculated Data:
      # ================
      #  T_e   T_n1   n_e   POP_COEFF
      ...
    """
    col_names = ["Te_eV", "Tn1_K", "ne_m3", pop_name]
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        comment="#",
        header=None,
        names=col_names,
        skiprows=4,
        engine="python"
    )
    return df

n4 = read_yacora_popcoeff(FILE_N4, "pc_n4")
n5 = read_yacora_popcoeff(FILE_N5, "pc_n5")

# Merge on common Te, Tn1, ne grid
model = pd.merge(n4, n5, on=["Te_eV", "Tn1_K", "ne_m3"], how="inner")

# =========================================================
# PHYSICALLY MOTIVATED CONVERSION:
# population coefficient -> relative optically thin line intensity
#
# I_ul ∝ N_u * A_ul * h*nu ∝ N_u * A_ul / lambda
# =========================================================
model["I_Hbeta_model"] = model["pc_n4"] * A42 / LAMBDA_HB
model["I_Hgamma_model"] = model["pc_n5"] * A52 / LAMBDA_HG

# Final model ratio
model["Hgamma_Hbeta_model"] = model["I_Hgamma_model"] / model["I_Hbeta_model"]

print("Model preview:")
print(
    model[["Te_eV", "Tn1_K", "ne_m3", "pc_n4", "pc_n5", "Hgamma_Hbeta_model"]]
    .head(10)
    .to_string(index=False)
)

# =========================================================
# READ EXCEL SHEET AND KEEP ONLY NUMERIC Hgamma/Hbeta ROWS
# =========================================================
meas = pd.read_excel(EXCEL_FILE)
meas.columns = meas.columns.astype(str).str.strip()

# keep only rows with a real code
meas = meas[meas[ID_COL].notna()].copy()
meas[ID_COL] = meas[ID_COL].astype(str).str.strip()

# convert useful columns
for col in [POWER_COL, PRESSURE_COL, RATIO_COL]:
    meas[col] = pd.to_numeric(meas[col], errors="coerce")

# keep only rows with a numeric Hgamma/Hbeta
meas_gamma = meas[meas[RATIO_COL].notna()].copy()

print("\nMeasured Hgamma/Hbeta rows:")
print(meas_gamma[[ID_COL, POWER_COL, PRESSURE_COL, RATIO_COL]].to_string(index=False))

# =========================================================
# PLOT MODEL CURVES + ALL MEASURED RATIOS
# =========================================================
selected_ne = [1e16, 3e16, 1e17]

plt.figure(figsize=(8, 5))

for ne_target in selected_ne:
    ne_actual = min(model["ne_m3"].unique(), key=lambda x: abs(x - ne_target))
    sub = model[model["ne_m3"] == ne_actual].sort_values("Te_eV")

    plt.plot(
        sub["Te_eV"],
        sub["Hgamma_Hbeta_model"],
        marker="o",
        label=fr"Model $n_e={ne_actual:.1e}\,\mathrm{{m^{{-3}}}}$"
    )

# plot all measured ratios as horizontal lines
for _, row in meas_gamma.iterrows():
    plt.axhline(
        row[RATIO_COL],
        color="red",
        alpha=0.12,
        linewidth=1
    )

# plot average measured ratio too
avg_ratio = meas_gamma[RATIO_COL].mean()
plt.axhline(
    avg_ratio,
    color="red",
    linestyle="--",
    linewidth=2,
    label=fr"Mean measured $H\gamma/H\beta = {avg_ratio:.4f}$"
)

plt.xlabel(r"$T_e$ (eV)")
plt.ylabel(r"$H\gamma/H\beta$")
plt.title(r"Measured and modelled $H\gamma/H\beta$")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# OPTIONAL: INTERSECTION SEARCH WITHOUT SCALING
# =========================================================
def find_intercept_te(te_vals, ratio_vals, measured_ratio):
    """
    Find Te where model ratio crosses measured ratio by linear interpolation.
    Returns NaN if there is no crossing.
    """
    te_vals = np.asarray(te_vals, dtype=float)
    ratio_vals = np.asarray(ratio_vals, dtype=float)

    diff = ratio_vals - measured_ratio

    exact = np.where(np.isclose(diff, 0, atol=1e-12))[0]
    if len(exact) > 0:
        return te_vals[exact[0]]

    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            x1, x2 = te_vals[i], te_vals[i + 1]
            y1, y2 = ratio_vals[i], ratio_vals[i + 1]
            return x1 + (measured_ratio - y1) * (x2 - x1) / (y2 - y1)

    return np.nan

results = []

for _, row in meas_gamma.iterrows():
    measured_ratio = row[RATIO_COL]

    for ne_target in selected_ne:
        ne_actual = min(model["ne_m3"].unique(), key=lambda x: abs(x - ne_target))
        sub = model[model["ne_m3"] == ne_actual].sort_values("Te_eV")

        te_intercept = find_intercept_te(
            sub["Te_eV"].values,
            sub["Hgamma_Hbeta_model"].values,
            measured_ratio
        )

        results.append({
            "id": row[ID_COL],
            "power_W": row[POWER_COL],
            "pressure_mTorr": row[PRESSURE_COL],
            "measured_Hgamma_Hbeta": measured_ratio,
            "ne_target": ne_target,
            "ne_used": ne_actual,
            "Te_intercept_eV": te_intercept
        })

results_df = pd.DataFrame(results)

print("\nIntercept results:")
print(results_df.to_string(index=False))

results_df.to_csv("Hgamma_Hbeta_Yacora_intercepts.csv", index=False)
print("\nSaved: Hgamma_Hbeta_Yacora_intercepts.csv")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

INTERCEPT_FILE = "Hgamma_Hbeta_Yacora_calibrated_intercepts.csv"
PLOT_NE = 1e16   # change this to 1e16, 3e16, or 1e17

df = pd.read_csv(INTERCEPT_FILE)
df = df[df["Te_intercept_eV"].notna()].copy()

plot_df = df[np.isclose(df["ne_target"], PLOT_NE)].copy()

plt.figure(figsize=(8, 5))

for power, group in plot_df.groupby("power_W"):
    group = group.sort_values("pressure_mTorr")
    plt.plot(
        group["pressure_mTorr"],
        group["Te_intercept_eV"],
        marker="o",
        linewidth=1.8,
        label=f"{int(power)} W"
    )

plt.xlabel("Pressure (mTorr)")
plt.ylabel(r"Effective $T_e$ (eV)")
plt.title(rf"$T_e$ trend vs pressure ($n_e = {PLOT_NE:.1e}$ m$^{{-3}}$)")
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

INTERCEPT_FILE = "Hgamma_Hbeta_Yacora_calibrated_intercepts.csv"
PLOT_NE = 1e16   # change this to 1e16, 3e16, or 1e17

df = pd.read_csv(INTERCEPT_FILE)
df = df[df["Te_intercept_eV"].notna()].copy()

plot_df = df[np.isclose(df["ne_target"], PLOT_NE)].copy()

plt.figure(figsize=(8, 5))

for pressure, group in plot_df.groupby("pressure_mTorr"):
    group = group.sort_values("power_W")

    plt.plot(
        group["power_W"],
        group["Te_intercept_eV"],
        marker="o",
        linewidth=1.5,
        label=f"{int(pressure)} mTorr"
    )

plt.xlabel("Power (W)")
plt.ylabel(r"Calculated effective $T_e$ (eV)")
plt.title(rf"Calculated effective $T_e$ vs power ($n_e \approx {PLOT_NE:.1e}$ m$^{{-3}}$)")
plt.legend()
plt.tight_layout()
plt.show()