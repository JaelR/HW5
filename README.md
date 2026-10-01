# Kepler Light Curve Explorer — KIC 11395018

Interactive data-visualization app built with **Streamlit** and **Altair** for the
DS-III Data Visualization assignment. It explores the long-term trend and
short-term variability of a Kepler light curve, shows uncertainty honestly, and
lets the user remap variables to visual channels on the fly.

**Live app:** https://bmvubqmkwdpwvuws2y7rkg.streamlit.app/

---

## Dataset

**Target:** KIC 11395018 — a solar-like oscillator observed by NASA's Kepler
mission in long-cadence mode (29.4 minutes per integration).

**Source:** Stitched Kepler light curve, FITS file `hlsp_kepler.fits`
(extension `LIGHTCURVE_STITCHED`), retrieved from the
[Mikulski Archive for Space Telescopes (MAST)](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html).

**Derived CSV:** `kic11395018_lightcurve.csv` (~33,900 rows), produced from the
FITS file with `astropy.io.fits` and committed to the repository so the app is
fully self-contained.

| Column | Meaning |
|---|---|
| `time` | BJD − 2454833 (days since Kepler launch epoch) |
| `cadenceno` | Sequential cadence index |
| `quarter` | Kepler quarter number |
| `flux` | PDCSAP flux (e⁻/s) |
| `flux_err` | 1σ uncertainty on `flux` |
| `sap_flux` | Simple aperture photometry flux |
| `sap_flux_err` | 1σ uncertainty on `sap_flux` |
| `psf_flat_flux` | PSF-fit photometry flux |
| `psf_flat_flux_err` | 1σ uncertainty on `psf_flat_flux` |
| `sap_quality` | Quality bitmask (0 = good) |
| `flatten_mask` | Flattening mask flag |

---

## Features

**Sidebar widgets**
- **Display resolution** — native cadence (29.4 min), 6-hour bins, or daily bins.
- **Quality filter** — optionally drop cadences with non-zero `sap_quality`.
- **Quarter multiselect** — restrict to specific Kepler quarters.

**Six tabs**
| Tab | What it shows |
|---|---|
| 📊 Overview | Light curve and STL trend side by side |
| 📈 Light Curve | Full-width light curve with rolling 95% uncertainty band |
| 🔍 STL | Faceted trend / seasonal / residual decomposition |
| 🎛 Encodings | Grammar-of-graphics playground (X, Y, Size, Color) |
| 🗂 Data | Filtered table + CSV download |
| ℹ️ About | Methodology, honesty notes, provenance |

**Extras**
- `st.download_button` exports the currently filtered dataset as CSV.
- Interactive Altair charts with tooltips and pan/zoom.
- Caching via `@st.cache_data` on CSV loading and STL decomposition.
- Themed layout with page icon, wide layout, and expanders for details.

---

## Pipeline

1. Load the stitched CSV and convert BJD to real datetimes.
2. Normalize flux to relative units centered at 1.0.
3. Optionally bin to 6-hour or daily resolution.
4. Decompose the daily-binned series via **STL** (period = 7 days).
5. Plot the light curve with a rolling ±1.96σ uncertainty band that combines
   pipeline `flux_err` (photon noise) with a rolling standard deviation
   (systematics).

---

## Temporal-honesty choices

- **Cadence disclosure.** Kepler long-cadence integrates 29.4 minutes per
  measurement. The sidebar label is *"Display resolution"*, not *"zoom"*,
  because binning is not zooming — aggregating averages away real short-period
  variability. The active resolution is stated in the figure captions.
- **Quarter gaps.** The x-axis is drawn in real time, so the ~1-day downlinks
  between Kepler quarters appear as horizontal gaps. This is truthful: the
  telescope was not observing during those intervals.
- **Quality flags.** Non-zero `sap_quality` cadences are kept by default so the
  reader sees the raw picture first. The sidebar toggle lets the user exclude
  them — the choice is explicit, never silent.
- **Axis honesty.** The y-axis uses `scale(zero=False)` because relative flux
  spans a narrow range (~0.99–1.01). Forcing zero would compress the signal
  into a flat line.
- **Uncertainty.** The band combines pipeline photon noise with rolling
  systematics. A fixed global band would misrepresent heteroskedasticity across
  quarters.

---

## Perceptual reasoning

The Encodings tab demonstrates the Cleveland–McGill hierarchy of visual
channels:

- **Position on a common scale** is the most accurate quantitative channel
  (Cleveland & McGill, 1984; Munzner, 2014).
- **Size (area)** is far less accurate — the eye underestimates large circles
  and overestimates small ones. Reserve area for ordinal signals or redundant
  emphasis.
- **Color hue** is categorical — excellent for identity, poor for ordering.
  Sequential scales (viridis here) suit ordinal or continuous variables.
- **Tooltips** surface exact values on hover, avoiding the need to decode size
  or hue precisely.

---

## Run locally

```bash
git clone https://github.com/jaelr/hw4_dv.git
cd hw4_dv
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
