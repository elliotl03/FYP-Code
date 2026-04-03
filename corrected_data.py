#!/usr/bin/env python3
from __future__ import annotations

"""
VS CODE READY — Robust spectral calibration & normalization with messy input handling (Part B).

This script:
- Loads a wavelength-dependent calibration FACTOR C(λ) (multiply) from a messy file (RTF/CSV/TXT with numbers).
- Parses raw spectra from messy files (headers/notes allowed; tab/semicolon/comma/whitespace; decimal comma OK).
- Applies: CALIBRATED = (RAW - DARK) * C(λ)
- Optionally NORMALIZES the calibrated spectrum (max/area/l2/minmax)
- Saves calibrated CSVs and PNG plots (Raw vs Calibrated vs Normalized)
- Optional watch mode to process new files automatically.
"""

import sys, time, re, io, csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# FOLDERS — leave as-is and just drop files in these folders
CALIBRATION_FILE = Path("EXAMPLE")   # <-- PUT YOUR CALIBRATION FILE HERE
INPUT_DIR        = Path("EXAMPLE")           # <-- PUT YOUR RAW SPECTRA FILES HERE
OUTPUT_DIR       = Path("EXAMPLE")     # <-- RESULTS WILL BE WRITTEN HERE

#!/usr/bin/env python3
"""
VS CODE READY — Robust spectral calibration & normalization with messy input handling (Part B).

This script:
- Loads a wavelength-dependent calibration FACTOR C(λ) (multiply) from a messy file (RTF/CSV/TXT with numbers).
- Parses raw spectra from messy files (headers/notes allowed; tab/semicolon/comma/whitespace; decimal comma OK).
- Applies: CALIBRATED = (RAW - DARK) * C(λ)
- Optionally NORMALIZES the calibrated spectrum (max/area/l2/minmax)
- Saves calibrated CSVs and PNG plots (Raw vs Calibrated vs Normalized)
- Optional watch mode to process new files automatically.

from __future__ import annotations
import sys, time, re, io, csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""

# =========================
# 🟩 USER SETTINGS — START
# =========================

# If you KNOW your spectrum headers, set them; otherwise leave blank to use robust parser.
WAVEL_COL   = ""    # e.g., "wavelength_nm" or "Wavelength"
RAW_COL     = ""    # e.g., "intensity_counts" or "Signal"
DARK_COL    = ""    # e.g., "dark_counts" — leave "" if no dark column

# Normalization applied to the CALIBRATED spectrum for plotting/saving
# Options: "none", "max", "area", "l2", "minmax"
NORMALIZATION = "max"

# Show matplotlib windows while also saving PNGs
SHOW_PLOTS = False

# Watch folder continuously for NEW files?
WATCH_FOR_NEW_FILES = False
WATCH_INTERVAL_SEC  = 2.0

# Output filename suffix before extension
OUTPUT_SUFFIX = "_cal"
# =========================
# 🟩 USER SETTINGS — END
# =========================

# ---------- Utilities ----------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def detect_decimal_and_delimiter(lines: list[str]) -> tuple[bool, str | None]:
    """Detect decimal comma and a likely delimiter (tab/semicolon/comma)."""
    sample = "\n".join(lines[:200])
    dec_comma = bool(re.search(r"\d+,\d+", sample)) and not bool(re.search(r"\d+\.\d+", sample))
    if any("\t" in ln for ln in lines[:50]):   delim = "\t"
    elif any(";" in ln for ln in lines[:50]):  delim = ";"
    elif any("," in ln for ln in lines[:50]):  delim = ","
    else:                                      delim = None
    return dec_comma, delim

_NUM = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"

def parse_numeric_block(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the LONGEST contiguous block of numeric rows.
    Returns (wl, raw, dark) with dark=0 if not present.
    - Accepts decimal commas and tab/semicolon/comma/whitespace delimiters.
    - Keeps rows having 2–4 numeric tokens.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty file")

    dec_comma, delim = detect_decimal_and_delimiter(lines)

    # Convert decimal comma to dot (only inside numbers)
    if dec_comma:
        def fix_decimals(s):
            return re.sub(r"(?<=\d),(?=\d)", ".", s)
        lines = [fix_decimals(ln) for ln in lines]

    rows: list[list[float] | None] = []
    for ln in lines:
        parts = re.split(rf"{re.escape(delim)}|\s+" if delim else r"\s+", ln)
        toks = [p for p in parts if re.fullmatch(_NUM, p)]
        if 2 <= len(toks) <= 4:
            try:
                vals = [float(t) for t in toks]
                rows.append(vals)
            except Exception:
                rows.append(None)
        else:
            rows.append(None)

    # Find the longest contiguous numeric region
    best_s, best_e, s = -1, -1, None
    for i, r in enumerate(rows):
        if r is not None and s is None:
            s = i
        end_of_block = (r is None) or (i == len(rows) - 1)
        if end_of_block and s is not None:
            e = i if r is not None else i - 1
            if (e - s) > (best_e - best_s):
                best_s, best_e = s, e
            s = None

    if best_s < 0:
        raise ValueError("No numeric block found (file may not contain data rows).")

    block = [rows[i] for i in range(best_s, best_e + 1) if rows[i] is not None]
    arr = np.array([r[:3] for r in block], dtype=float)  # up to 3 numeric columns
    if arr.shape[1] < 2:
        raise ValueError("Found numbers but fewer than 2 numeric columns.")

    wl  = arr[:, 0]
    raw = arr[:, 1]
    dark = arr[:, 2] if arr.shape[1] >= 3 else np.zeros_like(raw)
    return wl, raw, dark

def normalize(y: np.ndarray, method: str) -> np.ndarray:
    m = (method or "none").lower()
    if m == "none":
        return y
    if m == "max":
        s = np.nanmax(np.abs(y)); return y / s if s > 0 else y
    if m == "area":
        area = np.trapz(np.clip(y, 0, None)); return y / area if area > 0 else y
    if m == "l2":
        l2 = np.sqrt(np.nansum(y**2)); return y / l2 if l2 > 0 else y
    if m == "minmax":
        mn, mx = np.nanmin(y), np.nanmax(y); rng = mx - mn
        return (y - mn) / rng if rng > 0 else y
    raise ValueError(f"Unknown normalization: {method}")

# ---------- Calibration loader (messy-friendly) ----------
def load_calibration_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load calibration FACTOR C(λ) from a messy file.
    Takes the first two numeric columns of the longest numeric block as (λ, factor).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {path}\n"
            f"→ Put your file at {CALIBRATION_FILE} or edit CALIBRATION_FILE in the script."
        )
    text = read_text(path)
    wl, fac, _dark_ignored = parse_numeric_block(text)
    if wl.size < 2:
        raise ValueError("Calibration file did not contain enough numeric rows.")
    # sort by wavelength
    order = np.argsort(wl)
    return wl[order], fac[order]

def interp_factor(target_wl: np.ndarray, calib_wl: np.ndarray, calib_fac: np.ndarray) -> np.ndarray:
    """Interpolate calibration factor onto target wavelengths (clip at edges)."""
    return np.interp(target_wl, calib_wl, calib_fac, left=calib_fac[0], right=calib_fac[-1])

# ---------- Spectrum calibration ----------
def load_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a spectrum from a messy file:
    - If WAVEL_COL/RAW_COL are provided and file is CSV/TSV, try fast path with pandas.
    - Otherwise, robust numeric-block parsing.
    Returns (wl, raw, dark).
    """
    text = read_text(path)
    fast = (WAVEL_COL and RAW_COL and path.suffix.lower() in (".csv", ".tsv"))
    if fast:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
            wl  = pd.to_numeric(df[WAVEL_COL], errors="coerce").to_numpy(float)
            raw = pd.to_numeric(df[RAW_COL],   errors="coerce").to_numpy(float)
            dark = pd.to_numeric(df[DARK_COL], errors="coerce").to_numpy(float) if (DARK_COL and DARK_COL in df.columns) else np.zeros_like(raw)
            good = np.isfinite(wl) & np.isfinite(raw)
            wl, raw, dark = wl[good], raw[good], dark[good]
            if wl.size >= 2:
                return wl, raw, dark
        except Exception:
            # fall back to robust path
            pass
    # robust path
    return parse_numeric_block(text)

def calibrate_one_file(path: Path,
                       calib_wl: np.ndarray, calib_fac: np.ndarray,
                       out_dir: Path) -> Path | None:
    try:
        wl, raw, dark = load_spectrum(path)
    except Exception as e:
        print(f"[skip] {path.name}: cannot parse spectrum ({e})")
        return None

    fac = interp_factor(wl, calib_wl, calib_fac)
    calibrated = (raw - dark) * fac
    normalized = normalize(calibrated, NORMALIZATION)

    out = pd.DataFrame({
        "wavelength_nm": wl,
        "raw_intensity": raw,
        ("dark_counts" if np.any(dark) else "dark_used"): dark,
        "calib_factor": fac,
        "cal_intensity": calibrated,
        f"norm_{NORMALIZATION}": normalized
    })

    out_path = out_dir / f"{path.stem}{OUTPUT_SUFFIX}{path.suffix}"
    out.to_csv(out_path, index=False)

    plots_dir = ensure_dir(out_dir / "plots")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(wl, raw, color="#7f7f7f", lw=1.0, label="Raw")
    if np.any(dark):
        ax.plot(wl, dark, color="#bdbdbd", lw=0.8, label="Dark")
    ax.plot(wl, calibrated, color="#1f77b4", lw=1.2, label="Calibrated")
    ax.plot(wl, normalized, color="#d62728", lw=1.2, label=f"Normalized ({NORMALIZATION})")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Intensity (a.u.)"); ax.set_title(path.name)
    ax.legend(frameon=False, ncol=2); ax.grid(alpha=0.25, ls="--")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{path.stem}{OUTPUT_SUFFIX}.png")
    if SHOW_PLOTS: plt.show()
    plt.close(fig)

    print(f"[ok] {path.name} -> {out_path.name} (CSV + plot)")
    return out_path

# ---------- Main ----------
def main():
    ensure_dir(OUTPUT_DIR); ensure_dir(INPUT_DIR); ensure_dir(CALIBRATION_FILE.parent)

    # Load calibration once
    calib_wl, calib_fac = load_calibration_table(CALIBRATION_FILE)
    print(f"[info] Loaded calibration ({len(calib_wl)} pts) from {CALIBRATION_FILE}")

    processed: set[Path] = set()

    def candidates():
        exts = (".csv", ".tsv", ".txt")
        return sorted(p for p in INPUT_DIR.glob("*") if p.suffix.lower() in exts)

    # Initial batch
    for p in candidates():
        if p not in processed and p.stat().st_size > 0:
            out = calibrate_one_file(p, calib_wl, calib_fac, OUTPUT_DIR)
            if out is not None:
                processed.add(p)

    if not WATCH_FOR_NEW_FILES:
        print("[done] Batch complete.")
        return

    print(f"[watch] Monitoring {INPUT_DIR} every {WATCH_INTERVAL_SEC:.1f}s for new files… Ctrl+C to stop.")
    try:
        while True:
            time.sleep(WATCH_INTERVAL_SEC)
            for p in candidates():
                if p not in processed and p.stat().st_size > 0:
                    out = calibrate_one_file(p, calib_wl, calib_fac, OUTPUT_DIR)
                    if out is not None:
                        processed.add(p)
    except KeyboardInterrupt:
        print("\n[done] Stopped.")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # Avoid noisy SystemExit in IPython/VS Code interactive consoles
        pass
