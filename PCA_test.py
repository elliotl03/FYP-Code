import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# USER SETTINGS
# =========================================================
DATA_FOLDER = r"/Users/Elliot/Library/CloudStorage/OneDrive-DublinCityUniversity/4th Year/plasma_data/corrected_data"
FILE_PATTERN = "*.txt"

WAVELENGTH_COLUMN = "wavelength_nm"
INTENSITY_COLUMN = "cal_intensity"

# Exclude Hα by stopping before 650–662 nm
WL_MIN = 420.0
WL_MAX = 640.0
WL_STEP = 0.05

# preprocessing options:
# "none"   -> no spectrum-wise normalisation
# "area"   -> divide each spectrum by its total area
# "max"    -> divide each spectrum by its max
# "snv"    -> standard normal variate for each spectrum
# "zscore" -> z-score each wavelength column across spectra
NORMALISATION = "area"

N_COMPONENTS = 5
OUTPUT_FOLDER = "pca_output"


# =========================================================
# HELPERS
# =========================================================
def get_id_from_filename(filepath: str) -> str:
    return os.path.basename(filepath)[:4]


def load_spectrum(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    if WAVELENGTH_COLUMN not in df.columns:
        raise ValueError(f"{filepath} missing column {WAVELENGTH_COLUMN}")
    if INTENSITY_COLUMN not in df.columns:
        raise ValueError(f"{filepath} missing column {INTENSITY_COLUMN}")

    out = df[[WAVELENGTH_COLUMN, INTENSITY_COLUMN]].copy()
    out.columns = ["wavelength_nm", "intensity"]
    out = out.dropna().sort_values("wavelength_nm").reset_index(drop=True)
    return out


def crop_spectrum(df: pd.DataFrame, wl_min: float, wl_max: float) -> pd.DataFrame:
    return df[(df["wavelength_nm"] >= wl_min) & (df["wavelength_nm"] <= wl_max)].copy()


def interpolate_spectrum(df: pd.DataFrame, common_wl: np.ndarray) -> np.ndarray:
    x = df["wavelength_nm"].to_numpy()
    y = df["intensity"].to_numpy()

    if len(x) < 2:
        return np.full(common_wl.shape, np.nan)

    return np.interp(common_wl, x, y)


def preprocess_matrix(X: np.ndarray, method: str = "area") -> np.ndarray:
    Xp = X.copy().astype(float)

    if method == "none":
        return Xp

    elif method == "area":
        areas = np.trapz(Xp, dx=WL_STEP, axis=1)
        areas[areas == 0] = np.nan
        return Xp / areas[:, None]

    elif method == "max":
        max_vals = np.nanmax(Xp, axis=1)
        max_vals[max_vals == 0] = np.nan
        return Xp / max_vals[:, None]

    elif method == "snv":
        means = np.nanmean(Xp, axis=1)
        stds = np.nanstd(Xp, axis=1)
        stds[stds == 0] = np.nan
        return (Xp - means[:, None]) / stds[:, None]

    elif method == "zscore":
        col_means = np.nanmean(Xp, axis=0)
        col_stds = np.nanstd(Xp, axis=0)
        col_stds[col_stds == 0] = 1.0
        return (Xp - col_means) / col_stds

    else:
        raise ValueError(f"Unknown preprocessing method: {method}")


def mean_center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_spec = np.mean(X, axis=0)
    return X - mean_spec, mean_spec


def run_pca_numpy(X: np.ndarray, n_components: int):
    """
    PCA using SVD on mean-centered matrix X.
    Returns:
        scores: (n_samples, n_components)
        loadings: (n_components, n_features)
        explained_variance_ratio: (n_components,)
    """
    U, S, VT = np.linalg.svd(X, full_matrices=False)

    scores = U[:, :n_components] * S[:n_components]
    loadings = VT[:n_components, :]

    eigenvalues = (S ** 2) / (X.shape[0] - 1)
    explained_variance_ratio = eigenvalues[:n_components] / eigenvalues.sum()

    return scores, loadings, explained_variance_ratio


def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


# =========================================================
# MAIN
# =========================================================
def main():
    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepaths = sorted(glob.glob(os.path.join(DATA_FOLDER, FILE_PATTERN)))
    if not filepaths:
        raise FileNotFoundError(f"No files found in {DATA_FOLDER} matching {FILE_PATTERN}")

    common_wl = np.arange(WL_MIN, WL_MAX + WL_STEP, WL_STEP)

    ids = []
    filenames = []
    spectra = []

    for fp in filepaths:
        try:
            spec = load_spectrum(fp)
            spec = crop_spectrum(spec, WL_MIN, WL_MAX)

            if spec.empty:
                print(f"Skipped: {os.path.basename(fp)} (no data in range)")
                continue

            y_interp = interpolate_spectrum(spec, common_wl)

            if np.isnan(y_interp).any():
                print(f"Skipped: {os.path.basename(fp)} (NaNs after interpolation)")
                continue

            ids.append(get_id_from_filename(fp))
            filenames.append(os.path.basename(fp))
            spectra.append(y_interp)

            print(f"Loaded: {os.path.basename(fp)}")

        except Exception as e:
            print(f"Failed: {os.path.basename(fp)} -> {e}")

    if not spectra:
        raise RuntimeError("No valid spectra loaded.")

    X = np.vstack(spectra)

    # Save raw matrix
    raw_df = pd.DataFrame(X, columns=[f"{wl:.3f}" for wl in common_wl])
    raw_df.insert(0, "filename", filenames)
    raw_df.insert(0, "id", ids)
    save_csv(raw_df, output_dir / "spectra_matrix_raw.csv")

    # Preprocess
    X_proc = preprocess_matrix(X, NORMALISATION)

    if np.isnan(X_proc).any():
        raise RuntimeError("NaNs appeared after preprocessing. Check spectra and normalisation.")

    # Mean-center for PCA
    X_centered, mean_spec = mean_center(X_proc)

    # PCA
    n_comp = min(N_COMPONENTS, X_centered.shape[0], X_centered.shape[1])
    scores, loadings, explained = run_pca_numpy(X_centered, n_comp)

    # Save processed matrix
    proc_df = pd.DataFrame(X_proc, columns=[f"{wl:.3f}" for wl in common_wl])
    proc_df.insert(0, "filename", filenames)
    proc_df.insert(0, "id", ids)
    save_csv(proc_df, output_dir / "spectra_matrix_processed.csv")

    # Save scores
    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
    scores_df.insert(0, "filename", filenames)
    scores_df.insert(0, "id", ids)
    save_csv(scores_df, output_dir / "pca_scores.csv")

    # Save loadings
    loadings_df = pd.DataFrame(loadings.T, columns=[f"PC{i+1}" for i in range(loadings.shape[0])])
    loadings_df.insert(0, "wavelength_nm", common_wl)
    save_csv(loadings_df, output_dir / "pca_loadings.csv")

    # Save explained variance
    explained_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(explained))],
        "explained_variance_ratio": explained,
        "cumulative_explained_variance": np.cumsum(explained)
    })
    save_csv(explained_df, output_dir / "pca_explained_variance.csv")

    # Save mean spectrum
    mean_df = pd.DataFrame({
        "wavelength_nm": common_wl,
        "mean_processed_intensity": mean_spec
    })
    save_csv(mean_df, output_dir / "mean_processed_spectrum.csv")

    # =====================================================
    # PLOTS
    # =====================================================

    # Explained variance
    plt.figure(figsize=(7, 4))
    pcs = np.arange(1, len(explained) + 1)
    plt.bar(pcs, explained * 100, label="Individual")
    # plt.plot(pcs, np.cumsum(explained) * 100, marker="o", color="red", label="Cumulative")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance (%)")
    plt.title("PCA Variance")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "explained_variance.png", dpi=300)
    plt.close()

    # Scores plot PC1 vs PC2
    if scores.shape[1] >= 2:
        plt.figure(figsize=(7, 5))
        plt.scatter(scores[:, 0], scores[:, 1], s=50)

        for i, txt in enumerate(ids):
            plt.annotate(txt, (scores[i, 0], scores[i, 1]), fontsize=8, alpha=0.8)

        plt.axhline(0, color="gray", linewidth=0.8)
        plt.axvline(0, color="gray", linewidth=0.8)
        plt.xlabel(f"PC1 ({explained[0]*100:.2f}%)")
        plt.ylabel(f"PC2 ({explained[1]*100:.2f}%)")
        plt.title("PCA Scores Plot")
        plt.tight_layout()
        plt.savefig(output_dir / "scores_PC1_PC2.png", dpi=300)
        plt.close()

    # Loadings plot PC1 and PC2
    n_plot = min(2, loadings.shape[0])
    plt.figure(figsize=(9, 5))
    for i in range(n_plot):
        plt.plot(common_wl, loadings[i], label=f"PC{i+1}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Loading")
    plt.title("PCA Loadings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loadings_PC1_PC2.png", dpi=300)
    plt.close()

    # Mean processed spectrum
    plt.figure(figsize=(9, 5))
    plt.plot(common_wl, mean_spec)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mean processed intensity")
    plt.title("Mean Processed Spectrum")
    plt.tight_layout()
    plt.savefig(output_dir / "mean_processed_spectrum.png", dpi=300)
    plt.close()

    print("\nPCA complete.")
    print(f"Processed {len(ids)} spectra.")
    print(f"Saved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/Elliot/Library/CloudStorage/OneDrive-DublinCityUniversity/vscode/plasma/pca_output/pca_loadings.csv")

plt.figure(figsize=(10,5))
plt.plot(df["wavelength_nm"], df["PC1"], label="PC1")
plt.plot(df["wavelength_nm"], df["PC2"], label="PC2")

for wl, name in [(434.0, "Hγ"), (486.1, "Hβ"), (656.3, "Hα")]:
    plt.axvline(wl, color="k", linestyle="--", alpha=0.4)
    plt.text(wl+1, 0, name, rotation=90, va="bottom")

plt.axvspan(590, 640, color="orange", alpha=0.15, label="Fulcher region")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Loading")
plt.title("PCA loadings with key hydrogen features")
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# FILES
# -----------------------------
scores_file = "/Users/Elliot/Library/CloudStorage/OneDrive-DublinCityUniversity/vscode/plasma/pca_output/pca_scores.csv"
meta_file = "/Users/Elliot/Library/CloudStorage/OneDrive-DublinCityUniversity/4th Year/Final Year Project/Plasma Data.xlsx"

# -----------------------------
# LOAD DATA
# -----------------------------
scores = pd.read_csv(scores_file)
meta = pd.read_excel(meta_file)

print("Scores columns:", scores.columns.tolist())
print("Metadata columns:", meta.columns.tolist())

# -----------------------------
# COLUMN NAMES
# -----------------------------
ID_COL = "Code"
POWER_COL = "Power (watts)"
PRESSURE_COL = "Pressure (mTorr)"

# Keep only needed metadata columns
meta = meta[[ID_COL, POWER_COL, PRESSURE_COL]].copy()

# Rename ID column to match PCA scores file
meta = meta.rename(columns={ID_COL: "id"})

# Clean ID formatting
scores["id"] = scores["id"].astype(str).str.strip()
meta["id"] = meta["id"].astype(str).str.strip()

# Merge PCA scores with metadata
df = pd.merge(scores, meta, on="id", how="inner")

print("\nMerged dataframe preview:")
print(df.head())
print(f"\nMerged rows: {len(df)}")

# -----------------------------
# AVERAGE REPLICATES
# -----------------------------
df_avg = df.groupby([POWER_COL, PRESSURE_COL], as_index=False).agg(
    PC1_mean=("PC1", "mean"),
    PC1_std=("PC1", "std"),
    PC2_mean=("PC2", "mean"),
    PC2_std=("PC2", "std"),
    n_repeats=("id", "count")
)

# Replace NaN std (single measurement only) with 0
df_avg["PC1_std"] = df_avg["PC1_std"].fillna(0)
df_avg["PC2_std"] = df_avg["PC2_std"].fillna(0)

print("\nAveraged dataframe preview:")
print(df_avg.head())

# -----------------------------
# PC1 vs pressure, grouped by power
# -----------------------------
plt.figure(figsize=(8, 5))
for power, group in df_avg.groupby(POWER_COL):
    group = group.sort_values(PRESSURE_COL)
    plt.errorbar(
        group[PRESSURE_COL],
        group["PC1_mean"],
        yerr=group["PC1_std"],
        marker="o",
        capsize=3,
        label=f"{power} W"
    )

plt.xlabel("Pressure (mTorr)")
plt.ylabel("PC1 score")
plt.title("PC1 versus pressure")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# PC2 vs pressure, grouped by power
# -----------------------------
plt.figure(figsize=(8, 5))
for power, group in df_avg.groupby(POWER_COL):
    group = group.sort_values(PRESSURE_COL)
    plt.errorbar(
        group[PRESSURE_COL],
        group["PC2_mean"],
        yerr=group["PC2_std"],
        marker="o",
        capsize=3,
        label=f"{power} W"
    )

plt.xlabel("Pressure (mTorr)")
plt.ylabel("PC2 score")
plt.title("PC2 versus pressure")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# PC1 vs power, grouped by pressure
# -----------------------------
plt.figure(figsize=(8, 5))
for pressure, group in df_avg.groupby(PRESSURE_COL):
    group = group.sort_values(POWER_COL)
    plt.errorbar(
        group[POWER_COL],
        group["PC1_mean"],
        yerr=group["PC1_std"],
        marker="o",
        capsize=3,
        label=f"{pressure} mTorr"
    )

plt.xlabel("Power (W)")
plt.ylabel("PC1 score")
plt.title("PC1 versus power")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# PC2 vs power, grouped by pressure
# -----------------------------
plt.figure(figsize=(8, 5))
for pressure, group in df_avg.groupby(PRESSURE_COL):
    group = group.sort_values(POWER_COL)
    plt.errorbar(
        group[POWER_COL],
        group["PC2_mean"],
        yerr=group["PC2_std"],
        marker="o",
        capsize=3,
        label=f"{pressure} mTorr"
    )

plt.xlabel("Power (W)")
plt.ylabel("PC2 score")
plt.title("PC2 versus power")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# SCORES PLOT COLOURED BY PRESSURE
# Uses averaged points
# -----------------------------
plt.figure(figsize=(8, 6))
sc = plt.scatter(
    df_avg["PC1_mean"],
    df_avg["PC2_mean"],
    c=df_avg[PRESSURE_COL],
    cmap="viridis",
    s=80
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA scores coloured by pressure")
plt.colorbar(sc, label="Pressure (mTorr)")
plt.tight_layout()
plt.show()

# -----------------------------
# SCORES PLOT COLOURED BY POWER
# Uses averaged points
# -----------------------------
plt.figure(figsize=(8, 6))
sc = plt.scatter(
    df_avg["PC1_mean"],
    df_avg["PC2_mean"],
    c=df_avg[POWER_COL],
    cmap="plasma",
    s=80
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA scores coloured by power")
plt.colorbar(sc, label="Power (W)")
plt.tight_layout()
plt.show()