"""
Kepler Light Curve Explorer — Trend, Variability & Uncertainty
DS-III Assignment: Time series visualization with temporal framing.

Dataset: KIC 11395018, Kepler long-cadence stitched light curve
         (FITS: hlsp_kepler.fits, extension LIGHTCURVE_STITCHED)
         https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from statsmodels.tsa.seasonal import STL

st.set_page_config(
    page_title="Kepler Light Curve Explorer",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⭐ Kepler Light Curve — KIC 11395018")
st.caption(
    "Kepler long-cadence photometry | stitched multi-quarter baseline | "
    "29.4-minute integration per point"
)


# ---------------- Data Path (works locally AND on Streamlit Cloud) ----------------
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "kic11395018_lightcurve.csv"

if not DATA_FILE.exists():
    st.error(
        f"Data file not found at `{DATA_FILE}`. "
        f"Make sure `kic11395018_lightcurve.csv` is committed to the repo "
        f"at the same level as `app.py`."
    )
    st.stop()


# ---------------- Data Loading (cached) ----------------
@st.cache_data(show_spinner="Loading light curve…")
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Convert BJD (days since 2454833, Kepler launch epoch) to datetime
    epoch = pd.Timestamp("2009-05-02")
    df["datetime"] = epoch + pd.to_timedelta(df["time"], unit="D")

    # Relative flux centered near 1.0
    median_flux = df["flux"].median()
    df["rel_flux"] = df["flux"] / median_flux
    if "flux_err" in df.columns:
        df["rel_flux_err"] = df["flux_err"] / median_flux
    else:
        df["rel_flux_err"] = 0.0

    # Quality flag
    if "sap_quality" in df.columns:
        df["sap_quality"] = df["sap_quality"].fillna(0).astype(int)
    else:
        df["sap_quality"] = 0

    return df.sort_values("datetime").reset_index(drop=True)


df = load_data(DATA_FILE)

total_rows = len(df)
good_rows = int((df["sap_quality"] == 0).sum())
flagged_rows = total_rows - good_rows


# ---------------- Sidebar Controls ----------------
st.sidebar.header("⚙️ Controls")

# Widget 1 — resolution radio
resolution = st.sidebar.radio(
    "Display resolution",
    options=["Native cadence (29.4 min)", "6-hour bins", "Daily bins"],
    index=2,
    help="Kepler long-cadence integrates 29.4 minutes per point. "
         "Binning suppresses short-period variability.",
)

# Widget 2 — quality-flag checkbox
quality_filter = st.sidebar.checkbox(
    "Keep only good-quality cadences (sap_quality == 0)",
    value=False,
    help=f"{flagged_rows:,} cadences carry non-zero quality flags.",
)

# Widget 3 — quarter multiselect
quarters = sorted(df["quarter"].dropna().unique().tolist()) \
    if "quarter" in df.columns else []

if quarters:
    quarter_filter = st.sidebar.multiselect(
        "Quarters to display",
        options=quarters,
        default=quarters,
    )
else:
    quarter_filter = None

st.sidebar.divider()
st.sidebar.metric("Total cadences", f"{total_rows:,}")
st.sidebar.caption(f"Flagged: {flagged_rows:,} / {total_rows:,}")

with st.sidebar.expander("ℹ️ About this app"):
    st.markdown(
        """
        Interactive explorer for **KIC 11395018**, a solar-like oscillator
        observed by Kepler in long-cadence mode.

        - Data: MAST stitched light curve (`hlsp_kepler.fits`).
        - Pipeline: pandas → STL decomposition → Altair.
        - Layout: tabs for chart, decomposition, data, and about.
        """
    )


# ---------------- Apply Filters ----------------
filtered = df.copy()
if quarter_filter is not None:
    filtered = filtered[filtered["quarter"].isin(quarter_filter)]
if quality_filter:
    filtered = filtered[filtered["sap_quality"] == 0]

if filtered.empty:
    st.warning("No data after filtering. Relax the sidebar filters.")
    st.stop()


# ---------------- Resampling ----------------
bin_map = {
    "Native cadence (29.4 min)": None,
    "6-hour bins": "6h",
    "Daily bins": "1D",
}
rule = bin_map[resolution]

if rule is None:
    plot_df = (
        filtered[["datetime", "rel_flux", "rel_flux_err"]]
        .rename(columns={"datetime": "date"})
        .reset_index(drop=True)
    )
else:
    plot_df = (
        filtered.set_index("datetime")
        .resample(rule)
        .agg({"rel_flux": "mean", "rel_flux_err": "mean"})
        .dropna()
        .reset_index()
        .rename(columns={"datetime": "date"})
    )


# ---------------- Rolling Uncertainty Band ----------------
window = {"Native cadence (29.4 min)": 200,
          "6-hour bins": 14,
          "Daily bins": 7}[resolution]

roll_mean = plot_df["rel_flux"].rolling(window, center=True, min_periods=1).mean()
roll_std = plot_df["rel_flux"].rolling(window, center=True, min_periods=1).std()

total_sigma = np.sqrt(roll_std.fillna(0) ** 2 +
                      plot_df["rel_flux_err"].fillna(0) ** 2)

band_df = pd.DataFrame({
    "date": plot_df["date"],
    "mean": roll_mean,
    "lower": roll_mean - 1.96 * total_sigma,
    "upper": roll_mean + 1.96 * total_sigma,
})


# ---------------- STL Decomposition (cached) ----------------
@st.cache_data(show_spinner="Running STL decomposition…")
def decompose_daily(times: np.ndarray, values: np.ndarray):
    s = pd.Series(values, index=pd.to_datetime(times)).sort_index()
    s = s.resample("1D").mean().dropna()
    if len(s) < 14:
        return None
    res = STL(s, period=7, robust=True).fit()
    return {
        "dates": s.index,
        "trend": res.trend.values,
        "seasonal": res.seasonal.values,
        "resid": res.resid.values,
    }


stl_result = decompose_daily(
    filtered["datetime"].values, filtered["rel_flux"].values
)


# =============================================================
# MAIN TABS
# =============================================================
tab_overview, tab_lightcurve, tab_stl, tab_encodings, tab_data, tab_about = st.tabs(
    ["📊 Overview", "📈 Light Curve", "🔍 STL", "🎛 Encodings", "🗂 Data", "ℹ️ About"]
)


# =============================================================
# TAB 1 — Overview (both charts side by side)
# =============================================================
with tab_overview:
    st.subheader("Overview")
    st.markdown(
        f"Showing **{len(plot_df):,}** rows at **{resolution}** resolution "
        f"across {len(quarter_filter) if quarter_filter else len(quarters)} "
        f"quarters. Use the sidebar to change resolution, quality filter, "
        f"or quarter selection."
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("**Light curve with uncertainty**")
        band = alt.Chart(band_df).mark_area(opacity=0.25, color="#4c78a8").encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("lower:Q", title="Relative flux",
                    scale=alt.Scale(zero=False)),
            y2="upper:Q",
        )
        line = alt.Chart(band_df).mark_line(color="#1f4e79", strokeWidth=1.5).encode(
            x="date:T", y="mean:Q"
        )
        points = (
            alt.Chart(plot_df)
            .mark_circle(size=6, opacity=0.25, color="#555")
            .encode(
                x="date:T",
                y="rel_flux:Q",
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("rel_flux:Q", format=".5f", title="Relative flux"),
                ],
            )
        )
        st.altair_chart(
            (band + line + points).properties(height=360),
            use_container_width=True,
        )

    with col2:
        st.markdown("**Trend (STL)**")
        if stl_result is None:
            st.info("Not enough data for STL.")
        else:
            trend_chart = (
                alt.Chart(pd.DataFrame({
                    "date": stl_result["dates"],
                    "value": stl_result["trend"],
                }))
                .mark_line(color="#d62728")
                .encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("value:Q", title="Trend"),
                )
                .properties(height=360, title="Trend component")
            )
            st.altair_chart(trend_chart, use_container_width=True)

    st.caption(
        f"**Figure 1.** KIC 11395018 at **{resolution}**. Shaded band = "
        "rolling ±1.96σ combining pipeline `flux_err` and rolling-window "
        "systematics. Gaps between clusters are the ~1-day downlinks "
        "between Kepler quarters."
    )


# =============================================================
# TAB 2 — Light Curve (full width)
# =============================================================
with tab_lightcurve:
    st.subheader("Light Curve with Uncertainty")

    band = alt.Chart(band_df).mark_area(opacity=0.25, color="#4c78a8").encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("lower:Q", title="Relative flux",
                scale=alt.Scale(zero=False)),
        y2="upper:Q",
    )
    line = alt.Chart(band_df).mark_line(color="#1f4e79", strokeWidth=1.5).encode(
        x="date:T", y="mean:Q"
    )
    points = (
        alt.Chart(plot_df)
        .mark_circle(size=6, opacity=0.25, color="#555")
        .encode(
            x="date:T",
            y="rel_flux:Q",
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("rel_flux:Q", format=".5f", title="Relative flux"),
            ],
        )
    )
    st.altair_chart(
        (band + line + points).properties(height=480),
        use_container_width=True,
    )

    st.caption(
        f"**Figure 2.** Full-width view at **{resolution}**. Points are "
        "individual cadences (or binned averages), the dark line is the "
        "centered rolling mean, and the shaded band is the rolling 95% "
        "uncertainty interval."
    )


# =============================================================
# TAB 3 — STL Decomposition
# =============================================================
with tab_stl:
    st.subheader("STL Decomposition")
    st.markdown(
        "**STL** (Seasonal-Trend decomposition using LOESS) separates the "
        "daily-binned series into three additive components:\n\n"
        "- **Trend** — long-term drift of the light curve.\n"
        "- **Seasonal** — short-period structure (7-day period used here).\n"
        "- **Residual** — unexplained noise and systematics."
    )

    if stl_result is None:
        st.info("Not enough data for STL after filtering.")
    else:
        stl_long = pd.DataFrame({
            "date": list(stl_result["dates"]) * 3,
            "value": np.concatenate([
                stl_result["trend"],
                stl_result["seasonal"],
                stl_result["resid"],
            ]),
            "component": (
                ["Trend"] * len(stl_result["dates"]) +
                ["Seasonal (7-day)"] * len(stl_result["dates"]) +
                ["Residual"] * len(stl_result["dates"])
            ),
        })

        component_colors = alt.Scale(
            domain=["Trend", "Seasonal (7-day)", "Residual"],
            range=["#d62728", "#2ca02c", "#7f7f7f"],
        )

        stl_chart = (
            alt.Chart(stl_long)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("value:Q", title="Component value"),
                color=alt.Color("component:N", title="Component",
                                scale=component_colors),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("component:N", title="Component"),
                    alt.Tooltip("value:Q", format=".5f"),
                ],
            )
            .properties(height=420)
            .facet(
                row=alt.Row("component:N", title=None,
                            header=alt.Header(labelAngle=0, labelAlign="left")),
            )
            .resolve_scale(y="independent")
        )
        st.altair_chart(stl_chart, use_container_width=True)

        st.caption(
            "**Figure 3.** STL components. The trend captures slow drift; "
            "the seasonal panel isolates short-period structure; the "
            "residual shows unexplained scatter and instrumental systematics."
        )


# =============================================================
# TAB 4 — Encodings (grammar-of-graphics playground)
# =============================================================
with tab_encodings:
    st.subheader("🎛 Encoding Playground")
    st.markdown(
        "The same data can say different things depending on which variable "
        "you put on which visual channel. **Position is read most accurately, "
        "area least** — move a variable from Size to an axis and watch how "
        "much easier it becomes to compare."
    )

    # ---------------- Available fields ----------------
    # Numeric fields the user can map to channels
    numeric_fields = [
        "time", "rel_flux", "rel_flux_err",
        "flux", "flux_err", "sap_flux", "sap_quality",
    ]
    # Categorical fields usable for color
    categorical_fields = [c for c in ["quarter"] if c in df.columns]

    # ---------------- Widgets ----------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_field = st.selectbox("X (position)", numeric_fields, index=0)
    with c2:
        y_field = st.selectbox("Y (position)", numeric_fields, index=1)
    with c3:
        size_field = st.selectbox(
            "Size (area)",
            ["(none)"] + numeric_fields,
            index=0,
        )
    with c4:
        color_field = st.selectbox(
            "Color (hue)",
            ["(none)"] + categorical_fields + numeric_fields,
            index=0,
        )

    # ---------------- Downsampling slider ----------------
    max_points = st.slider(
        "Max points to draw (downsampled for speed)",
        min_value=500,
        max_value=max(500, min(len(filtered), 20000)),
        value=min(max(500, len(filtered)), 5000),
        step=500,
        help="Your dataset has ~34k cadences. Drawing all of them makes the "
             "browser sluggish. Downsampling keeps the playground responsive.",
    )

    # ---------------- Filter + downsample ----------------
    playground_df = (
        filtered.dropna(subset=[x_field, y_field])
        .rename(columns={"datetime": "date"})
        .copy()
    )
    if len(playground_df) > max_points:
        playground_df = playground_df.sample(max_points, random_state=0)
    playground_df = playground_df.sort_values("date")

    # ---------------- Build the Altair chart ----------------
    enc = {
        "x": alt.X(f"{x_field}:Q", title=x_field),
        "y": alt.Y(f"{y_field}:Q", title=y_field),
        "tooltip": [
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip(f"{x_field}:Q", title=x_field, format=".4f"),
            alt.Tooltip(f"{y_field}:Q", title=y_field, format=".4f"),
        ],
    }

    if size_field != "(none)":
        enc["size"] = alt.Size(
            f"{size_field}:Q",
            title=size_field,
            scale=alt.Scale(range=[10, 400]),
        )
    else:
        enc["size"] = alt.value(30)

    if color_field != "(none)":
        if color_field in categorical_fields:
            enc["color"] = alt.Color(f"{color_field}:N", title=color_field)
            enc["tooltip"].append(
                alt.Tooltip(f"{color_field}:N", title=color_field)
            )
        else:
            enc["color"] = alt.Color(
                f"{color_field}:Q",
                title=color_field,
                scale=alt.Scale(scheme="viridis"),
            )
            enc["tooltip"].append(
                alt.Tooltip(f"{color_field}:Q", title=color_field, format=".4f")
            )
    else:
        enc["color"] = alt.value("#4c78a8")

    playground_chart = (
        alt.Chart(playground_df)
        .mark_circle(opacity=0.55, stroke=None)
        .encode(**enc)
        .properties(height=520)
        .interactive()
    )

    st.altair_chart(playground_chart, use_container_width=True)

    st.caption(
        f"**Figure 4.** Encoding playground. Showing "
        f"**{len(playground_df):,}** of **{len(plot_df):,}** rows "
        f"(downsampled). X → *{x_field}*, Y → *{y_field}*, "
        f"Size → *{size_field}*, Color → *{color_field}*. "
        "Because position is judged most accurately and area least, "
        "remapping a variable from Size to X or Y makes comparisons "
        "much easier."
    )

    # ---------------- Perceptual justification panel ----------------
    with st.expander("📖 Why this works — perceptual reasoning"):
        st.markdown(
            """
            **Position on a common scale** is the most accurate visual
            channel for quantitative comparison (Cleveland & McGill, 1984;
            Munzner, 2014, Fig. 5.5). When a variable lives on the X or Y
            axis, the reader can compare values by vertical or horizontal
            offset from a shared reference — a very low-error judgment.

            **Size (area)** is far less accurate. The eye underestimates
            large circles and overestimates small ones, so area is best
            reserved for *ordinal* signals or for adding redundant emphasis,
            never for exact magnitude comparison. Move a variable from
            *Size* to *X* or *Y* in this tab and the comparison becomes
            visibly easier.

            **Color (hue)** is a categorical channel: excellent for
            distinguishing *identity* ("which quarter is this?") but poor
            for ordering quantities. Use sequential scales (viridis here)
            only when the underlying variable is ordinal or continuous.

            **Interaction.** Altair's `.interactive()` lets you pan and
            zoom; tooltips surface exact values on hover, which is how the
            playground avoids forcing the reader to decode size or hue
            precisely.
            """
        )


# =============================================================
# TAB 4 — Data
# =============================================================
with tab_data:
    st.subheader("Filtered data")
    st.markdown(
        f"Currently showing **{len(plot_df):,}** rows at "
        f"**{resolution}** resolution. Download the current view as CSV."
    )

    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=plot_df.to_csv(index=False).encode("utf-8"),
        file_name=f"kepler_kic11395018_{resolution.replace(' ', '_').replace('(', '').replace(')', '')}.csv",
        mime="text/csv",
    )

    st.dataframe(plot_df.head(200), use_container_width=True)
    st.caption(f"Showing first 200 of {len(plot_df):,} rows.")

    with st.expander("Column glossary"):
        st.markdown(
            """
            | Column | Meaning |
            |---|---|
            | `time` | BJD − 2454833 (days since Kepler launch epoch) |
            | `cadenceno` | Sequential cadence index |
            | `quarter` | Kepler quarter number |
            | `flux` | PDCSAP flux (e⁻/s) |
            | `flux_err` | 1σ uncertainty on `flux` |
            | `sap_flux` | Simple aperture photometry flux |
            | `sap_quality` | Quality bitmask (0 = good) |
            """
        )


# =============================================================
# TAB 5 — About
# =============================================================
with tab_about:
    st.subheader("About this app")

    st.markdown(
        """
        **Purpose.** Visualize the long-term trend and short-term variability
        of Kepler's stitched light curve for KIC 11395018, with an honest
        representation of uncertainty and temporal resolution.

        **Dataset.** NASA Kepler mission, long-cadence photometry, stitched
        across multiple quarters. Source FITS file `hlsp_kepler.fits`
        (extension `LIGHTCURVE_STITCHED`), retrieved from the Mikulski
        Archive for Space Telescopes (MAST).

        **Pipeline.**
        1. Load the stitched CSV and convert BJD to real datetime.
        2. Normalize flux to relative units centered at 1.0.
        3. Optionally bin to 6-hour or daily resolution.
        4. Decompose the daily-binned series via STL (period = 7 days).
        5. Plot the light curve with a rolling ±1.96σ uncertainty band
           combining pipeline `flux_err` with systematic rolling std.

        **Temporal-honesty choices.**
        - Cadence is disclosed (29.4 min integration per point).
        - Quarter downlink gaps appear as real gaps, not bridged.
        - Quality flags are visible and filterable, never silent.
        - Y-axis uses `zero=False` to preserve the narrow flux range.
        """
    )

    st.divider()
    st.markdown("### 📝 Temporal-Honesty Choice (full note)")
    st.markdown(
        f"""
        **Cadence disclosure.** Kepler long-cadence integrates **29.4 minutes
        per measurement**. The sidebar label is *"Display resolution"*, not
        *"zoom"*, because binning is not zooming: aggregating to 6-hour or
        daily bins averages away real short-period variability. At
        **{resolution}**, each point represents
        {'one 29.4-minute integration' if rule is None else ('a 6-hour average' if rule == '6h' else 'a daily average')}.

        **Quarter gaps.** The x-axis is drawn in real time, so the ~1-day
        downlinks between Kepler quarters appear as horizontal breaks — a
        truthful representation of the observing schedule.

        **Quality flags.** {flagged_rows:,} of {total_rows:,} cadences carry
        non-zero `sap_quality` flags. The sidebar toggle lets you include or
        exclude them.

        **Axis honesty.** The y-axis uses `scale(zero=False)` because
        relative flux spans a narrow range (~0.99–1.01). Uncertainty combines
        pipeline `flux_err` (photon noise) with a rolling standard deviation
        (systematics).
        """
    )