import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
DATA_FOLDER = r"/Users/Elliot/Library/CloudStorage/OneDrive-DublinCityUniversity/4th Year/plasma_data/corrected_data"
FILE_PATTERN = "*.txt"
OUTPUT_CSV = "balmer_ratios_by_id.csv"

INTENSITY_COLUMN = "cal_intensity"
WAVELENGTH_COLUMN = "wavelength_nm"

# Balmer line centres (nm)
LINES = {
    "Halpha": 656.279,
    "Hbeta": 486.133,
    "Hgamma": 434.047,
}

# Integration settings
HALF_WINDOW = 2.5
BASELINE_OUTER = 4.0

# Plot settings
PLOT_DIAGNOSTICS = True
PLOT_SAVE_FOLDER = "balmer_diagnostic_plots"
PLOT_FILE_IDS = {"H040_HRC21401__0__16-27-52-756_cal.txt"}
# Example:
# PLOT_FILE_IDS = {"H007", "H013", "H021"}
# Use None to plot every file


# =========================
# FUNCTIONS
# =========================
def get_id_from_filename(filepath: str) -> str:
    fname = os.path.basename(filepath)
    return fname[:4]


def load_spectrum(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    if WAVELENGTH_COLUMN not in df.columns:
        raise ValueError(f"Missing column: {WAVELENGTH_COLUMN}")
    if INTENSITY_COLUMN not in df.columns:
        raise ValueError(f"Missing column: {INTENSITY_COLUMN}")

    out = df[[WAVELENGTH_COLUMN, INTENSITY_COLUMN]].copy()
    out.columns = ["wavelength_nm", "intensity"]
    out = out.dropna().sort_values("wavelength_nm").reset_index(drop=True)
    return out


def integrate_line_area(
    df: pd.DataFrame,
    centre_nm: float,
    half_window: float = HALF_WINDOW,
    baseline_outer: float = BASELINE_OUTER,
    plot: bool = False,
    line_name: str = "",
    spectrum_id: str = "",
    filename: str = ""
):
    """
    Baseline-corrected integrated area for one line.
    Also returns diagnostic arrays for plotting.
    """
    wl = df["wavelength_nm"].to_numpy()
    y = df["intensity"].to_numpy()

    # local display region
    display_region = (wl >= centre_nm - baseline_outer - 1.0) & (wl <= centre_nm + baseline_outer + 1.0)

    # regions used in fitting and integration
    line_region = (wl >= centre_nm - half_window) & (wl <= centre_nm + half_window)
    baseline_region = (
        ((wl >= centre_nm - baseline_outer) & (wl <= centre_nm - half_window)) |
        ((wl >= centre_nm + half_window) & (wl <= centre_nm + baseline_outer))
    )

    if line_region.sum() < 4 or baseline_region.sum() < 4:
        return {
            "area": np.nan,
            "coeffs": None,
            "x_line": None,
            "y_line": None,
            "baseline_line": None,
            "y_corr": None,
            "display_wl": wl[display_region],
            "display_y": y[display_region],
            "baseline_x": wl[baseline_region],
            "baseline_y": y[baseline_region],
        }

    x_fit = wl[baseline_region]
    y_fit = y[baseline_region]

    # linear baseline fit
    coeffs = np.polyfit(x_fit, y_fit, 1)

    x_line = wl[line_region]
    y_line = y[line_region]
    baseline_line = np.polyval(coeffs, x_line)

    y_corr = y_line - baseline_line
    y_corr_pos = np.clip(y_corr, 0, None)

    if len(x_line) < 2:
        area = np.nan
    else:
        area = np.trapz(y_corr_pos, x_line)

    if plot:
        plot_diagnostic(
            wl[display_region],
            y[display_region],
            x_fit,
            y_fit,
            x_line,
            y_line,
            baseline_line,
            y_corr,
            centre_nm,
            half_window,
            baseline_outer,
            line_name,
            spectrum_id,
            filename,
            area
        )

    return {
        "area": area,
        "coeffs": coeffs,
        "x_line": x_line,
        "y_line": y_line,
        "baseline_line": baseline_line,
        "y_corr": y_corr,
        "display_wl": wl[display_region],
        "display_y": y[display_region],
        "baseline_x": x_fit,
        "baseline_y": y_fit,
    }


def plot_diagnostic(
    display_wl,
    display_y,
    baseline_x,
    baseline_y,
    x_line,
    y_line,
    baseline_line,
    y_corr,
    centre_nm,
    half_window,
    baseline_outer,
    line_name,
    spectrum_id,
    filename,
    area
):
    os.makedirs(PLOT_SAVE_FOLDER, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)

    # -------------------------
    # Top plot: raw spectrum + baseline fit
    # -------------------------
    axes[0].plot(display_wl, display_y, color="black", label="Raw spectrum")
    axes[0].scatter(baseline_x, baseline_y, color="tab:orange", s=18, label="Baseline-fit points")

    # plot fitted baseline over line window
    axes[0].plot(x_line, baseline_line, color="tab:red", linestyle="--", label="Linear baseline")

    axes[0].axvspan(centre_nm - half_window, centre_nm + half_window, color="tab:blue", alpha=0.15, label="Integration window")
    axes[0].axvline(centre_nm, color="tab:green", linestyle=":", label=f"{line_name} centre")

    axes[0].set_title(f"{spectrum_id} | {line_name} | Raw spectrum and baseline")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Intensity")
    axes[0].legend(loc="best")

    # -------------------------
    # Bottom plot: baseline-corrected signal
    # -------------------------
    y_corr_pos = np.clip(y_corr, 0, None)
    axes[1].plot(x_line, y_corr, color="tab:purple", label="Baseline-corrected signal")
    axes[1].fill_between(x_line, 0, y_corr_pos, color="tab:blue", alpha=0.35, label=f"Integrated area = {area:.4g}")

    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axvline(centre_nm, color="tab:green", linestyle=":")
    axes[1].set_title(f"{spectrum_id} | {line_name} | Baseline-corrected area")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Corrected intensity")
    axes[1].legend(loc="best")

    plt.tight_layout()

    outname = f"{spectrum_id}_{line_name}_diagnostic.png"
    plt.savefig(os.path.join(PLOT_SAVE_FOLDER, outname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyse_file(filepath: str) -> dict:
    df = load_spectrum(filepath)
    spectrum_id = get_id_from_filename(filepath)
    fname = os.path.basename(filepath)

    result = {
        "id": spectrum_id,
        "filename": fname,
    }

    should_plot = PLOT_DIAGNOSTICS and (
        PLOT_FILE_IDS is None or spectrum_id in PLOT_FILE_IDS
    )

    for line_name, centre in LINES.items():
        line_res = integrate_line_area(
            df,
            centre,
            plot=should_plot,
            line_name=line_name,
            spectrum_id=spectrum_id,
            filename=fname
        )
        result[f"{line_name}_area"] = line_res["area"]

    ha = result["Halpha_area"]
    hb = result["Hbeta_area"]
    hg = result["Hgamma_area"]

    result["Halpha_Hbeta"] = ha / hb if pd.notna(ha) and pd.notna(hb) and hb > 0 else np.nan
    result["Hgamma_Hbeta"] = hg / hb if pd.notna(hg) and pd.notna(hb) and hb > 0 else np.nan

    return result


# =========================
# MAIN
# =========================
def main():
    filepaths = sorted(glob.glob(os.path.join(DATA_FOLDER, FILE_PATTERN)))

    if not filepaths:
        raise FileNotFoundError(f"No files found in {DATA_FOLDER} matching {FILE_PATTERN}")

    results = []
    for fp in filepaths:
        try:
            res = analyse_file(fp)
            results.append(res)
            print(f"Done: {os.path.basename(fp)}")
        except Exception as e:
            print(f"Failed: {os.path.basename(fp)} -> {e}")
            results.append({
                "id": get_id_from_filename(fp),
                "filename": os.path.basename(fp),
                "Halpha_area": np.nan,
                "Hbeta_area": np.nan,
                "Hgamma_area": np.nan,
                "Halpha_Hbeta": np.nan,
                "Hgamma_Hbeta": np.nan,
                "error": str(e),
            })

    results_df = pd.DataFrame(results)

    preferred_cols = [
        "id",
        "filename",
        "Halpha_area",
        "Hbeta_area",
        "Hgamma_area",
        "Halpha_Hbeta",
        "Hgamma_Hbeta",
        "error",
    ]
    existing_cols = [c for c in preferred_cols if c in results_df.columns]
    other_cols = [c for c in results_df.columns if c not in existing_cols]
    results_df = results_df[existing_cols + other_cols]

    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()