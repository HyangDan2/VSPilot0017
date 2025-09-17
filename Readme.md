# Slanted Edge MTF (Multi-ROI, SFR-style, Save-on-`C`)

A PySide6 GUI tool to measure **ESF/LSF/MTF** from slanted-edge ROIs.  
Implements a robust ESF accumulator (histogram+oversampling) and **sfrmat3-aligned** conveniences (normal direction, tail guard, pixel box sinc, channel choices, etc.).  
**Press `C`** or click **Save** to export per-ROI CSVs and current ESF/LSF/MTF PNG figures.

https://user-images.example/… (add a short GIF if you want)

## Features
- Load image → add **multiple ROIs** (drag or numeric) → **Compute all**.
- ESF: oversampled histogram, smoothing (Gaussian), optional normalization (1–99 percentile).
- LSF: central difference (default) or first-difference; Hann/Hamming window.
- MTF: normalized by DC; units selectable (**cycles/mm** or **cycles/pixel**).  
  Optional **pixel box correction** via sinc(f\_pix).
- **Auto orientation** using PCA of strong gradient points; aligns bright side → positive normal.
- **Export**: per-ROI `esf_*.csv`, `lsf_*.csv`, `mtf_mm_*.csv`, `mtf_pix_*.csv` + `ESF.png`, `LSF.png`, `MTF.png`.
- **Keyboard**: `C` to save; `Delete/Backspace` to remove selected ROIs in the list.

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python -m src.app
# or
python src/app.py
```

# Usage Guide

## Usage
1. **Load Image** (`PNG/JPG/BMP/TIFF`).
2. **Draw ROI**  
   - Draw a rectangle to mark a slanted-edge region → click **Add ROI (from drag)**  
   - Or manually fill **ROI x/y/w/h** and click **Add ROI (numeric)**.
3. **Set Parameters**  
   - Pixel pitch (mm/pixel)  
   - Smoothing sigma  
   - Derivative, window, etc.
4. **Compute**  
   - ESF / LSF / MTF (all ROIs).
5. **Optional Tuning**
   - MTF X unit: `cycles/mm` vs `cycles/pixel`  
   - MTF X scale: multiplier (×) to rescale the horizontal axis  
   - MTF X min/max: adjust display windowing  
   - Normalize ESF  
   - Inverse Gamma (γ)  
   - Pixel box correction
6. **Export**  
   - Press **C** or **Save** to export results as **CSVs/PNGs**.

---

## Notes on SFR Alignment
- Normal direction is flipped if local intensity correlation is negative → ensures brighter side increases with **+t**.  
- ESF edges are trimmed (percentile-based and low-count bins) and smoothed; optional tail guard stabilizes the far side.  
- Pixel box correction divides by `sinc(f_pixel)` before re-normalization at DC.  

---

## Known Limitations
- Requires reasonably clean slanted edges; very small ROIs (<6×6) are rejected.  
- Gamma handling is simplified (single exponent).  
- No automated **MTF50 extraction** yet (but easy to add from the `mtf` array).  

## Folder Structure
```bash
src/
  app.py
  core/           # processing & math
  ui/             # Qt widgets and dialogs
  utils/          # colors, helpers
```

## License
MIT — see [LICENSE](./License.txt)
