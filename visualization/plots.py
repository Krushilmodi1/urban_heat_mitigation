"""All maps, charts and the summary report. Works fully offline from the files
in data/processed - no Earth Engine call is made here.
"""
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

log = logging.getLogger("plots")
NODATA = -9999
_LEVELS = np.arange(config.LST_MIN, config.LST_MAX + config.LST_STEP / 2, config.LST_STEP)
_RAMP = mcolors.LinearSegmentedColormap.from_list("lst", config.LST_COLORS[1:-1])
LST_CMAP = mcolors.ListedColormap(_RAMP(np.linspace(0, 1, len(_LEVELS) - 1))).with_extremes(
    under=config.LST_COLORS[0], over=config.LST_COLORS[-1], bad="#e6e6e6")
LST_NORM = mcolors.BoundaryNorm(_LEVELS, LST_CMAP.N)
_B = _LEVELS[::2]                               # colour-bar tick labels every 5 deg C
YEAR_COLORS = ["#1b6ca8", "#d95f02", "#2a9d5c", "#7b4ea3", "#b8860b", "#555555"]
VALUE_COLS = ["LST", "NDVI", "NDBI", "LST_city", "NDVI_city", "NDBI_city"]


def is_live_satellite_data():
    return os.path.exists(config.EXTRACTION_MANIFEST)


def source_text():
    if is_live_satellite_data():
        sensor = "Landsat 8 + 9" if config.USE_LANDSAT9 else "Landsat 8"
        return (f"Data: {sensor} Collection 2 Level-2 (USGS), ~10:30 local overpass, clear-sky pixels only, "
                "via Google Earth Engine")
    return ("Environmental Data: Calibrated Offline Synthetic Proxy (Earth Engine offline fallback; "
            "simulated microclimate coupled with real building footprint density).")


def buildings_text():
    meta = {}
    if os.path.exists(config.BUILDINGS_META_JSON):
        with open(config.BUILDINGS_META_JSON) as fh:
            meta = json.load(fh)
    conf = meta.get("confidence_threshold_applied")
    return ("Buildings: Google Research Open Buildings, uploaded CSV (CC BY 4.0 / ODbL); "
            + (f"confidence ≥ {conf}; " if conf is not None else "no confidence filter; ")
            + "single static snapshot.")


def area_labels():
    """Study-area labels from the saved boundary record (flags a district boundary)."""
    from data_pipeline.study_area import study_area_record
    rec = study_area_record()
    district = rec.get("kind") == "district"
    return {"city": "Ahmedabad DISTRICT (GAUL) – not the city" if district else "Ahmedabad municipal boundary",
            "title": "AHMEDABAD DISTRICT (TEST ONLY)" if district else "AHMEDABAD",
            "buffer": f"{config.BUFFER_M // 1000} km buffer", "district": district}


def _out(*parts):
    path = os.path.join(config.PLOTS_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _save(fig, path, dpi=None):
    fig.savefig(path, dpi=dpi or config.PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s", os.path.relpath(path, config.BASE_DIR))


# ------------------------------------------------------------------- boundaries
def load_rings(path, to_crs=None):
    """Outer and inner rings of a saved GeoJSON as arrays of (x, y); optionally reprojected."""
    with open(path) as fh:
        gj = json.load(fh)
    rings = []

    def walk(geom):
        t = geom["type"]
        if t == "Polygon":
            rings.extend(np.asarray(r, dtype=float)[:, :2] for r in geom["coordinates"])
        elif t == "MultiPolygon":
            for poly in geom["coordinates"]:
                rings.extend(np.asarray(r, dtype=float)[:, :2] for r in poly)
        elif t == "GeometryCollection":
            for g in geom["geometries"]:
                walk(g)

    for feat in gj["features"]:
        walk(feat["geometry"])
    if to_crs:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", to_crs, always_xy=True)
        rings = [np.column_stack(tr.transform(r[:, 0], r[:, 1])) for r in rings]
    return rings


def _draw_boundaries(ax, to_crs=None, scale=1.0, legend=True, loc="lower left"):
    """City: solid black line. Buffer: dashed black line."""
    labels = area_labels()
    for path, style, width in ((config.BUFFER_GEOJSON, "--", 1.2), (config.BOUNDARY_GEOJSON, "-", 1.6)):
        if os.path.exists(path):
            for ring in load_rings(path, to_crs):
                ax.plot(ring[:, 0] * scale, ring[:, 1] * scale, color="black", linestyle=style, linewidth=width)
    if legend:
        ax.legend(handles=[Line2D([], [], color="black", lw=1.6, label=labels["city"]),
                           Line2D([], [], color="black", lw=1.2, ls="--", label=labels["buffer"])],
                  loc=loc, fontsize=8, frameon=True, framealpha=0.95)


def _study_extent(margin=0.02):
    xy = np.vstack(load_rings(config.BUFFER_GEOJSON))
    return [xy[:, 0].min() - margin, xy[:, 0].max() + margin, xy[:, 1].min() - margin, xy[:, 1].max() + margin]


def _scale_bar(ax, extent, km=10):
    """Simple scale bar for a lon/lat map."""
    lat = (extent[2] + extent[3]) / 2
    deg = km / (111.32 * np.cos(np.deg2rad(lat)))
    x0 = extent[1] - (extent[1] - extent[0]) * 0.05 - deg
    y0 = extent[2] + (extent[3] - extent[2]) * 0.04
    ax.plot([x0, x0 + deg], [y0, y0], color="black", lw=3, solid_capstyle="butt")
    ax.text(x0 + deg / 2, y0 + (extent[3] - extent[2]) * 0.012, f"{km} km", ha="center", va="bottom", fontsize=8)
    ax.annotate("N", xy=(x0 + deg / 2, y0 + (extent[3] - extent[2]) * 0.12),
                xytext=(x0 + deg / 2, y0 + (extent[3] - extent[2]) * 0.06), ha="center", fontsize=10,
                fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))


# --------------------------------------------------------------------- LST maps
def read_lst(path, band=1):
    """Read one band; pixels outside the saved study-area polygon are masked."""
    import rasterio
    from rasterio.features import geometry_mask
    with rasterio.open(path) as src:
        arr = src.read(band).astype(float)
        b = src.bounds
        if src.crs and src.crs.to_epsg() == 4326 and os.path.exists(config.BUFFER_GEOJSON):
            with open(config.BUFFER_GEOJSON) as fh:
                shapes = [f["geometry"] for f in json.load(fh)["features"]]
            arr[geometry_mask(shapes, out_shape=arr.shape, transform=src.transform)] = np.nan
    arr[(arr == NODATA) | (arr < -100)] = np.nan
    return arr, [b.left, b.right, b.bottom, b.top]


def lst_map(arr, extent, subtitle, out_path, note=None, footer=None):
    """One LST map in the fixed project style (same classes on every map)."""
    full = _study_extent()
    fig, ax = plt.subplots(figsize=(9, 9.8))
    ax.set_facecolor("white")
    im = None
    if arr is not None:
        for ring in load_rings(config.BUFFER_GEOJSON)[:1]:       # grey = inside study area, no valid pixel
            ax.fill(ring[:, 0], ring[:, 1], facecolor="#d9d9d9", edgecolor="none", zorder=0)
        im = ax.imshow(np.ma.masked_invalid(arr), cmap=LST_CMAP.with_extremes(bad=(0, 0, 0, 0)), norm=LST_NORM,
                       extent=extent, origin="upper", interpolation="nearest", zorder=1)
    else:
        for ring in load_rings(config.BUFFER_GEOJSON)[:1]:
            ax.fill(ring[:, 0], ring[:, 1], facecolor="#f0f0f0", edgecolor="none", hatch="///", alpha=0.6)
    _draw_boundaries(ax)
    ax.set_xlim(full[0], full[1])
    ax.set_ylim(full[2], full[3])
    ax.set_aspect(1 / np.cos(np.deg2rad((full[2] + full[3]) / 2)))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.tick_params(labelsize=8)
    ax.grid(color="grey", linewidth=0.3, alpha=0.4)
    _scale_bar(ax, full)

    sm = im or plt.cm.ScalarMappable(norm=LST_NORM, cmap=LST_CMAP)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.02, extend="both", ticks=_B, spacing="uniform")
    cbar.set_label("Land Surface Temperature (°C)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(subtitle, fontsize=11, fontweight="bold", pad=8)
    ax.annotate("LAND SURFACE TEMPERATURE", xy=(0.5, 1), xycoords="axes fraction",
                xytext=(0, 26 + 15 * subtitle.count("\n")), textcoords="offset points",
                ha="center", va="bottom", fontsize=17, fontweight="bold")
    if note:
        ax.text(0.5, 0.5 if arr is None else 0.94, note, transform=ax.transAxes, ha="center", va="center",
                fontsize=11 if arr is None else 9, fontweight="bold", color="#8b0000",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#8b0000"))
    foot = (footer + "\n" if footer else "") + source_text() + (
        f".\nFixed {config.LST_STEP:g} °C classes on every map; below {config.LST_MIN:g} °C / above "
        f"{config.LST_MAX:g} °C use the end colours; "
        "grey = no valid (cloud-free) pixel.")
    ax.annotate(foot, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -42), textcoords="offset points",
                ha="center", va="top", fontsize=7.5, style="italic")
    _save(fig, out_path, config.MAP_DPI)


def _period_text(row):
    def g(name):
        v = getattr(row, name, None)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v
    win_s, win_e = g("window_start"), g("window_end_exclusive")
    parts = []
    if win_s and win_e:
        last_day = (pd.Timestamp(win_e) - pd.Timedelta(days=1)).strftime("%d %b %Y")
        parts.append(f"Observation window: {pd.Timestamp(win_s):%d %b %Y} – {last_day}")
    if g("first_scene"):
        parts.append(f"scenes {g('first_scene')} to {g('last_scene')}")
    n8, n9 = g("n_scenes_L8"), g("n_scenes_L9")
    if n8 is not None:
        parts.append(f"L8: {int(n8)}, L9: {int(n9 or 0)} scene(s)")
    return "; ".join(parts) or None


def plot_monthly_maps(df):
    """One map per month of the period - including months without usable data."""
    from data_pipeline.extract_gee_data import monthly_tif_path
    lab = area_labels()
    for row in df.itertuples():
        name = config.MONTH_NAMES[row.Month - 1].upper()
        out = _out("lst_monthly_maps", str(row.Year), f"ahmedabad_lst_{row.Year}_{row.Month:02d}.png")
        tif = monthly_tif_path(row.Year, row.Month)
        sub = f"{lab['title']} + 10 KM BUFFER — DAYTIME LST, MONTHLY MEAN ({name} {row.Year})"
        footer = _period_text(row)
        if row.status == "no_data":
            lst_map(None, None, sub, out, note="NO VALID LANDSAT OBSERVATIONS\nno cloud-free pixel this month",
                    footer=footer)
            continue
        if row.status == "extraction_failed":
            lst_map(None, None, sub, out, note="EXTRACTION FAILED\nre-run the extract stage", footer=footer)
            continue
        if not os.path.exists(tif):
            lst_map(None, None, sub, out, note="RASTER NOT DOWNLOADED\n(statistics exist; re-run extract)",
                    footer=footer)
            continue
        arr, extent = read_lst(tif)
        note = None
        if row.status != "observed":
            note = (f"INSUFFICIENT VALID OBSERVATIONS\n{row.valid_fraction:.0%} of the area is cloud-free — "
                    "excluded from statistics")
        lst_map(arr, extent, sub + f"\n{int(row.n_scenes)} scene(s), {row.valid_fraction:.0%} cloud-free, "
                f"mean {row.LST:.1f} °C" if row.status == "observed" else
                sub + f"\n{int(row.n_scenes)} scene(s), {row.valid_fraction:.0%} cloud-free", out, note, footer)


def plot_annual_maps(df):
    from data_pipeline.extract_gee_data import annual_tif_path
    lab = area_labels()
    for year, grp in df.groupby("Year"):
        tif = annual_tif_path(year)
        n = int((grp.status == "observed").sum())
        sub = f"{lab['title']} + 10 KM BUFFER — ANNUAL MEAN DAYTIME LST ({year})\nmean of {n} observed month(s) out of {len(grp)}"
        if not os.path.exists(tif):
            log.warning("No annual raster for %d - map shows 'no data'.", year)
            lst_map(None, None, sub, _out(f"ahmedabad_lst_annual_{year}.png"),
                    note="NO ANNUAL RASTER\n(no observed month or download failed)")
            continue
        arr, extent = read_lst(tif)
        months = ", ".join(grp[grp.status == "observed"].Month_Name)
        lst_map(arr, extent, sub, _out(f"ahmedabad_lst_annual_{year}.png"),
                footer=f"Observed months used (equal weight): {months}")


# ----------------------------------------------------------------------- trends
def _observed(df):
    """Monthly values with non-observed months set to NaN (so lines show gaps)."""
    d = df.copy()
    cols = [c for c in VALUE_COLS if c in d.columns]
    d.loc[d.status != "observed", cols] = np.nan
    return d


def plot_yearly_trends(df):
    d = _observed(df)
    for year, grp in d.groupby("Year"):
        s = grp.set_index("Month").reindex(range(1, 13))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(s.index, s.LST, color="#d95f02", marker="o", linewidth=2, label="Study area (city + buffer)")
        ax.plot(s.index, s.LST_city, color="#555555", marker="s", ms=4, linewidth=1.2, linestyle="--",
                label="Inside city boundary")
        for m, v in s.LST.dropna().items():
            ax.annotate(f"{v:.1f}", (m, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        for m in s.index[s.LST.isna()]:
            ax.axvspan(m - 0.5, m + 0.5, color="#eeeeee", zorder=0)
        ax.set_xticks(range(1, 13), config.MONTH_NAMES)
        ax.set_xlim(0.5, 12.5)
        ax.set_ylabel("Monthly mean LST (°C)")
        ax.set_title(f"Ahmedabad monthly land surface temperature — {year}", fontweight="bold")
        handles, labels = ax.get_legend_handles_labels()
        if s.LST.isna().any():
            handles.append(plt.Rectangle((0, 0), 1, 1, color="#eeeeee"))
            labels.append("No / insufficient valid data (not plotted)")
        ax.legend(handles, labels, fontsize=8, loc="lower center")
        ax.grid(alpha=0.3)
        fig.text(0.5, -0.02, source_text(), ha="center", fontsize=7, style="italic")
        _save(fig, _out(f"ahmedabad_lst_trend_{year}.png"))


def plot_cross_year(df):
    d = _observed(df)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (year, grp) in enumerate(d.groupby("Year")):
        s = grp.set_index("Month").reindex(range(1, 13))
        ax.plot(s.index, s.LST, marker="o", linewidth=2, color=YEAR_COLORS[i % len(YEAR_COLORS)], label=str(year))
    ax.set_xticks(range(1, 13), config.MONTH_NAMES)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylabel("Monthly mean LST (°C)")
    ax.set_title("Ahmedabad (+10 km buffer): monthly LST compared across years", fontweight="bold")
    ax.legend(title="Year")
    ax.grid(alpha=0.3)
    fig.text(0.5, -0.02, "Gaps = months without sufficient cloud-free observations. " + source_text(),
             ha="center", fontsize=7, style="italic")
    _save(fig, _out("ahmedabad_lst_cross_year_comparison.png"))


def long_term_trend(df):
    """Linear trend of deseasonalised LST. Returns a dict; 'justified' says whether to draw it."""
    d = _observed(df).dropna(subset=["LST"]).copy()
    out = {"n": len(d), "justified": False, "reason": ""}
    if len(d) < config.TREND_MIN_MONTHS:
        out["reason"] = (f"No trend line: only {len(d)} observed months "
                         f"(at least {config.TREND_MIN_MONTHS} needed to separate a trend from the seasonal cycle)")
        return out
    d["t"] = d.Year + (d.Month - 0.5) / 12
    d["anom"] = d.LST - d.groupby("Month").LST.transform("mean")
    fit = stats.linregress(d.t, d.anom)
    out.update(slope=fit.slope, p=fit.pvalue, intercept=fit.intercept, mean=d.LST.mean())
    if fit.pvalue < config.TREND_ALPHA:
        out["justified"] = True
    else:
        out["reason"] = (f"No trend line: seasonally adjusted slope {fit.slope:+.2f} °C/yr "
                         f"is not significant (p = {fit.pvalue:.2f}, n = {len(d)})")
    return out


def plot_long_term(df):
    d = _observed(df)
    dates = pd.to_datetime(d.Date)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, d.LST, color="#d95f02", marker="o", linewidth=1.8, label="Observed monthly mean LST")
    tr = long_term_trend(df)
    if tr["justified"]:
        t = d.Year + (d.Month - 0.5) / 12
        ax.plot(dates, tr["mean"] + tr["intercept"] + tr["slope"] * t, color="black", linestyle="--",
                label=f"Fitted linear trend (seasonally adjusted): {tr['slope']:+.2f} °C/yr, p = {tr['p']:.3f}")
    else:
        ax.text(0.01, 0.03, tr["reason"], transform=ax.transAxes, fontsize=8, style="italic",
                bbox=dict(facecolor="white", edgecolor="#bbbbbb"))
    ax.set_ylabel("Monthly mean LST (°C)")
    ax.set_title("Ahmedabad (+10 km buffer): monthly LST over the study period", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.text(0.5, -0.03, "Gaps = months without sufficient cloud-free observations. " + source_text(),
             ha="center", fontsize=7, style="italic")
    _save(fig, _out("ahmedabad_lst_time_series.png"))


def annual_comparison_table(df):
    """Annual mean over all observed months, and over months observed in EVERY year (fair comparison)."""
    d = _observed(df)
    obs = d.dropna(subset=["LST"])
    years = sorted(df.Year.unique())
    common = sorted(set.intersection(*[set(obs[obs.Year == y].Month) for y in years])) if len(obs) else []
    rows = []
    for y in years:
        g = obs[obs.Year == y]
        gc = g[g.Month.isin(common)]
        rows.append({"Year": y, "observed_months": len(g), "mean_LST_all_observed": g.LST.mean(),
                     "mean_LST_common_months": gc.LST.mean() if common else np.nan,
                     "n_common_months": len(common),
                     "hottest_month": g.loc[g.LST.idxmax(), "Month_Name"] if len(g) else None,
                     "hottest_LST": g.LST.max(), "coolest_month": g.loc[g.LST.idxmin(), "Month_Name"] if len(g) else None,
                     "coolest_LST": g.LST.min()})
    return pd.DataFrame(rows), common


def plot_annual_comparison(df):
    tab, common = annual_comparison_table(df)
    tab.to_csv(os.path.join(config.PROCESSED_DIR, "annual_lst_comparison.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    x = np.arange(len(tab))
    w = 0.38
    ax.bar(x - w / 2, tab.mean_LST_all_observed, w, color="#fe9929", label="All observed months")
    ax.bar(x + w / 2, tab.mean_LST_common_months, w, color="#cc4c02",
           label=f"Only months observed in every year (n = {len(common)})")
    for i, r in tab.iterrows():
        for off, v in ((-w / 2, r.mean_LST_all_observed), (w / 2, r.mean_LST_common_months)):
            if pd.notna(v):
                ax.text(i + off, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{int(r.Year)}\n({int(r.observed_months)}/12 months observed)" for r in tab.itertuples()])
    ax.set_ylabel("Mean LST (°C)")
    ax.set_title("Annual mean LST", fontweight="bold")
    top = np.nanmax(tab[["mean_LST_all_observed", "mean_LST_common_months"]].values) if len(tab) else 1
    ax.set_ylim(0, (top if np.isfinite(top) else 1) * 1.25)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    # relatively hot / cool periods: monthly anomaly against the period's monthly mean
    d = _observed(df).dropna(subset=["LST"]).copy()
    ax = axes[1]
    if len(d):
        d["anom"] = d.LST - d.groupby("Month").LST.transform("mean")
        dates = pd.to_datetime(d.Date)
        colors = np.where(d.anom >= 0, "#cc4c02", "#2171b5")
        ax.bar(dates, d.anom, width=20, color=colors)
        for _, r in d.loc[[d.anom.idxmax(), d.anom.idxmin()]].iterrows():
            ax.annotate(f"{r.Month_Name} {r.Year}\n{r.anom:+.1f} °C", (pd.Timestamp(r.Date), r.anom),
                        textcoords="offset points", xytext=(0, 8 if r.anom > 0 else -22), ha="center", fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Difference from the same calendar month's mean (°C)")
    ax.set_title("Relatively hot (orange) and cool (blue) months", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30); lbl.set_ha("right")
    fig.text(0.5, -0.08, "Annual means are only comparable when the same months are observed; the common-months bar "
             "removes the effect of different cloud gaps. Anomalies need at least two years of the same month to be "
             "meaningful. " + source_text(), ha="center", fontsize=7, style="italic", wrap=True)
    _save(fig, _out("ahmedabad_lst_annual_comparison.png"))


# ----------------------------------------------------------------- correlations
def corr_table(df, cols, method="pearson"):
    """Pairwise correlation and sample-size matrices (NaN / inf excluded pairwise)."""
    data = df[cols].replace([np.inf, -np.inf], np.nan)
    r = pd.DataFrame(np.nan, index=cols, columns=cols)
    n = pd.DataFrame(0, index=cols, columns=cols)
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    for a in cols:
        for b in cols:
            pair = data[[a, b]].dropna() if a != b else data[[a]].dropna()
            n.loc[a, b] = len(pair)
            if a == b:
                r.loc[a, b] = 1.0 if len(pair) else np.nan
            elif len(pair) >= 3 and pair[a].nunique() > 1 and pair[b].nunique() > 1:
                r.loc[a, b] = fn(pair[a], pair[b])[0]
    return r, n


def _corr_figure(df, cols, labels, title, path, footnote):
    fig, axes = plt.subplots(1, 2, figsize=(7 + 1.6 * len(cols), 3 + 0.75 * len(cols)))
    for ax, method in zip(axes, ("pearson", "spearman")):
        r, n = corr_table(df, cols, method)
        annot = np.array([[("n/a" if np.isnan(r.iloc[i, j]) else f"{r.iloc[i, j]:.2f}") + f"\nn={n.iloc[i, j]:,}"
                           for j in range(len(cols))] for i in range(len(cols))])
        sns.heatmap(r.astype(float), annot=annot, fmt="", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, square=True,
                    xticklabels=labels, yticklabels=labels, annot_kws={"size": 8},
                    cbar_kws={"shrink": 0.7, "label": "Correlation coefficient"}, linewidths=0.5)
        ax.set_title(f"{method.capitalize()} correlation", fontweight="bold")
        ax.tick_params(axis="x", rotation=40)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(0.5, -0.03, footnote, ha="center", fontsize=8, style="italic", wrap=True)
    fig.tight_layout()
    _save(fig, path)


def plot_monthly_correlation(df):
    d = _observed(df)
    _corr_figure(d, ["LST", "NDVI", "NDBI"], ["LST", "NDVI", "NDBI"],
                 "Monthly study-area means: LST, NDVI, NDBI", _out("correlation_monthly_means.png"),
                 "Each point is one month's regional average, so this mostly reflects the shared seasonal cycle "
                 "(summer heat, monsoon greening). Building variables are not included: the building CSV is one "
                 "static snapshot, so it has no month-to-month variation. Small n; correlation is not causation.")


GRID_VARS = [("LST", "LST (°C)"), ("NDVI", "NDVI"), ("NDBI", "NDBI"),
             ("building_count", "Building count"), ("footprint_area_m2", "Footprint area"),
             ("building_density", "Footprint density"), ("buildings_per_km2", "Buildings / km²")]


def _grid_columns(grid):
    pairs = [(c, l) for c, l in GRID_VARS if c in grid.columns and grid[c].notna().sum() >= 3]
    return grid, pairs


def _grid_for_stats(grid, year):
    """Cells for statistics: one year, not edge slivers."""
    g = grid[grid.year == year]
    return g[~g.edge_cell.astype(bool)] if "edge_cell" in g else g


def plot_grid_correlation(grid, year):
    g, pairs = _grid_columns(_grid_for_stats(grid, year))
    cols, labels = [p[0] for p in pairs], [p[1] for p in pairs]
    _corr_figure(g, cols, labels, f"Spatial ({config.GRID_SCALE_M} m grid) correlations — annual mean {year}",
                 _out(f"correlation_grid_{year}.png"),
                 f"Each point is one {config.GRID_SCALE_M} m cell of Ahmedabad + buffer (edge cells < "
                 f"{config.MIN_CELL_STUDY_FRACTION:.0%} inside excluded). Neighbouring cells are not independent "
                 "(spatial autocorrelation), so significance is overstated; buildings are a static snapshot, not "
                 "year-specific. Correlation is not causation.")


def plot_grid_scatter(grid, year):
    g = _grid_for_stats(grid, year)
    specs = [("NDVI", "NDVI (annual mean)", 1), ("NDBI", "NDBI (annual mean)", 1),
             ("building_density", "Building footprint density (% of cell)", 100),
             ("footprint_area_m2", "Building footprint area per cell (ha)", 1e-4)]
    specs = [sp for sp in specs if sp[0] in g and g[sp[0]].notna().sum() >= 3]
    if not specs or g.LST.notna().sum() < 3:
        log.warning("Grid scatter %d skipped: fewer than 3 cells with data.", year)
        return
    fig, axes = plt.subplots(1, len(specs), figsize=(5.2 * len(specs), 5.2), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (col, label, mult) in zip(axes, specs):
        d = g[[col, "LST", "zone"]].replace([np.inf, -np.inf], np.nan).dropna()
        for zone, colour in (("buffer_ring", "#8fb8de"), ("city", "#d95f02")):
            z = d[d.zone == zone]
            ax.scatter(z[col] * mult, z.LST, s=5, alpha=0.35, color=colour, label=f"{zone.replace('_', ' ')} cells",
                       rasterized=True)
        if len(d) >= 3 and d[col].nunique() > 1:
            r, rho = stats.pearsonr(d[col], d.LST)[0], stats.spearmanr(d[col], d.LST)[0]
            fit = np.polyfit(d[col] * mult, d.LST, 1)
            xs = np.linspace((d[col] * mult).min(), (d[col] * mult).max(), 50)
            ax.plot(xs, np.polyval(fit, xs), color="black", linewidth=1.2, linestyle="--", label="Linear fit")
            ax.text(0.03, 0.97, f"Pearson r = {r:.2f}\nSpearman ρ = {rho:.2f}\nn = {len(d):,} cells",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="#bbbbbb"))
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"Annual mean LST {year} (°C)")
    axes[-1].legend(loc="lower right", fontsize=8, markerscale=3)
    fig.suptitle(f"LST against vegetation, built-up index and buildings — {config.GRID_SCALE_M} m cells, {year}",
                 fontweight="bold")
    fig.text(0.5, -0.03, "Associations only - not causal. " + buildings_text(), ha="center", fontsize=7, style="italic")
    fig.tight_layout()
    _save(fig, _out(f"scatter_lst_drivers_{year}.png"))


def plot_density_classes(grid, year):
    """LST by building-density class, separately for the city and the buffer."""
    g = _grid_for_stats(grid, year).dropna(subset=["LST", "building_density"]).copy()
    if len(g) < 3:
        return
    bins = [-0.001, 0.01, 0.05, 0.15, 0.30, 1.0]
    names = ["<1 %", "1–5 %", "5–15 %", "15–30 %", ">30 %"]
    g["density_class"] = pd.cut(g.building_density, bins=bins, labels=names)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(data=g, x="density_class", y="LST", hue="zone", hue_order=["city", "buffer_ring"],
                palette={"city": "#d95f02", "buffer_ring": "#8fb8de"}, fliersize=1.5, linewidth=0.9, ax=ax)
    counts = g.groupby(["density_class", "zone"], observed=False).size().unstack(fill_value=0)
    ax.set_xticks(range(len(names)), [f"{n}\ncity n={counts.loc[n].get('city', 0):,} · "
                                      f"buffer n={counts.loc[n].get('buffer_ring', 0):,}" for n in names], fontsize=8)
    ax.set_xlabel(f"Building footprint density of the {config.GRID_SCALE_M} m cell")
    ax.set_ylabel(f"Annual mean LST {year} (°C)")
    ax.set_title(f"LST by building density class — city vs. 10 km buffer ring ({year})", fontweight="bold")
    ax.legend(title="Zone")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, _out(f"lst_by_building_density_{year}.png"))


# ------------------------------------------------------------------ grid maps
def _grid_raster(g, col):
    s = config.GRID_SCALE_M
    g = g.drop_duplicates("grid_id")
    xs = np.arange(g.x_utm.min(), g.x_utm.max() + s / 2, s)
    ys = np.arange(g.y_utm.max(), g.y_utm.min() - s / 2, -s)
    piv = g.pivot_table(index="y_utm", columns="x_utm", values=col, dropna=False).reindex(index=ys, columns=xs)
    ext = [(xs.min() - s / 2) / 1000, (xs.max() + s / 2) / 1000, (ys.min() - s / 2) / 1000, (ys.max() + s / 2) / 1000]
    return np.ma.masked_invalid(piv.values.astype(float)), ext


def _panel(ax, data, ext, cmap, norm, label, title, extend):
    im = ax.imshow(data, extent=ext, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
    _draw_boundaries(ax, to_crs=config.GRID_CRS, scale=1 / 1000)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    ax.set_xlabel("Easting (km, UTM 43N)"); ax.set_ylabel("Northing (km, UTM 43N)")
    ax.set_title(title, fontweight="bold", fontsize=11)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.75, extend=extend, ticks=_B if norm is LST_NORM else None)
    cb.set_label(label)


def _dens_cmap():
    return plt.get_cmap("magma_r").with_extremes(bad="#e6e6e6")


def _mask_edges(g):
    """Edge slivers (< MIN_CELL_STUDY_FRACTION inside the study area) give unstable densities - hide them."""
    g = g.copy()
    if "edge_cell" in g:
        edge = g.edge_cell.astype(bool)
        g.loc[edge, ["building_density", "buildings_per_km2", "LST"]
              if "LST" in g else ["building_density", "buildings_per_km2"]] = np.nan
    return g


EDGE_NOTE = (f"Edge cells less than {int(config.MIN_CELL_STUDY_FRACTION * 100)} % inside the study area are left "
             "blank (their densities are unstable).")


def plot_building_maps(bgrid):
    """Static building maps from the building grid: footprint density and count density."""
    if bgrid["building_count"].notna().sum() == 0:
        return
    bgrid = _mask_edges(bgrid)
    s = config.GRID_SCALE_M
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    if bgrid.building_density.notna().any():
        dens, ext = _grid_raster(bgrid, "building_density")
        vmax = max(5, np.nanpercentile(dens.filled(np.nan) * 100, 99))
        _panel(axes[0], dens * 100, ext, _dens_cmap(), mcolors.PowerNorm(0.5, vmin=0, vmax=vmax),
               "Footprint density (% of cell area covered by buildings)",
               "Building footprint density\n(total footprint area ÷ cell area)", "max")
    else:
        axes[0].text(0.5, 0.5, "Footprint area not available\n(CSV has no polygons or area column)",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_axis_off()
    cnt, ext = _grid_raster(bgrid, "buildings_per_km2")
    vmax = max(10, np.nanpercentile(cnt.filled(np.nan), 99))
    _panel(axes[1], cnt, ext, plt.get_cmap("viridis").copy(), mcolors.PowerNorm(0.5, vmin=0, vmax=vmax),
           "Buildings per km²", "Building count density\n(number of buildings ÷ km²)", "max")
    fig.suptitle(f"Ahmedabad + 10 km buffer — buildings on the {s} m grid", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.03, buildings_text() + " Each building is assigned to the cell containing its centroid. " + EDGE_NOTE,
             ha="center", fontsize=7, style="italic", wrap=True)
    _save(fig, _out("buildings", "building_density_maps.png"), config.MAP_DPI)


def plot_footprints():
    """Footprint map from the cleaned building CSV (polygons if available, else centroids)."""
    path = config.BUILDINGS_CLEAN_CSV
    if not os.path.exists(path):
        return
    cols = pd.read_csv(path, nrows=0).columns
    has_poly = "geometry_wkt" in cols
    n = sum(len(c) for c in pd.read_csv(path, usecols=["x_utm"], chunksize=500_000))
    fig, ax = plt.subplots(figsize=(11, 11))
    draw_polys = has_poly and n <= config.MAX_FOOTPRINTS_TO_DRAW
    if draw_polys:
        import shapely
        from matplotlib.collections import PolyCollection
        from pyproj import Transformer
        tr = Transformer.from_crs(config.OPEN_BUILDINGS_CRS, config.PROJECTED_CRS, always_xy=True)
        verts = []
        for chunk in pd.read_csv(path, usecols=["geometry_wkt"], chunksize=200_000):
            geoms = shapely.from_wkt(chunk.geometry_wkt.values, on_invalid="ignore")
            for gm in geoms:
                if gm is None:
                    continue
                for p in getattr(gm, "geoms", [gm]):
                    if p.geom_type == "Polygon":
                        xy = np.asarray(p.exterior.coords)
                        x, y = tr.transform(xy[:, 0], xy[:, 1])
                        verts.append(np.column_stack([x, y]) / 1000)
        ax.add_collection(PolyCollection(verts, facecolor="#7a0177", edgecolor="none", rasterized=True))
        what = f"{n:,} building footprints (polygons)"
    else:
        for chunk in pd.read_csv(path, usecols=["x_utm", "y_utm"], chunksize=500_000):
            ax.scatter(chunk.x_utm / 1000, chunk.y_utm / 1000, s=0.05, color="#7a0177", alpha=0.5,
                       linewidths=0, rasterized=True)
        what = (f"{n:,} building centroids" + (" (too many polygons to draw; centroids shown)" if has_poly
                                              else " (CSV has points, not footprints)"))
    _draw_boundaries(ax, to_crs=config.PROJECTED_CRS, scale=1 / 1000)
    rings = np.vstack(load_rings(config.BUFFER_GEOJSON, config.PROJECTED_CRS)) / 1000
    ax.set_xlim(rings[:, 0].min() - 1, rings[:, 0].max() + 1); ax.set_ylim(rings[:, 1].min() - 1, rings[:, 1].max() + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km, UTM 43N)"); ax.set_ylabel("Northing (km, UTM 43N)")
    ax.set_title(f"Ahmedabad + 10 km buffer — {what}", fontweight="bold")
    fig.text(0.5, 0.04, buildings_text(), ha="center", fontsize=7, style="italic")
    _save(fig, _out("buildings", "building_footprints_map.png"), config.MAP_DPI)


def plot_lst_vs_density_maps(grid, year):
    """LST beside building footprint density on the same grid."""
    g = _mask_edges(grid[grid.year == year])
    lst, ext = _grid_raster(g, "LST")
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    _panel(axes[0], lst, ext, LST_CMAP, LST_NORM, "Land Surface Temperature (°C)",
           f"Annual mean daytime LST, {year} ({config.GRID_SCALE_M} m cells)", "both")
    if g.building_density.notna().any():
        dens, _ = _grid_raster(g, "building_density")
        vmax = max(5, np.nanpercentile(dens.filled(np.nan) * 100, 99))
        _panel(axes[1], dens * 100, ext, _dens_cmap(), mcolors.PowerNorm(0.5, vmin=0, vmax=vmax),
               "Building footprint density (% of cell)", "Building footprint density (static snapshot)", "max")
    else:
        axes[1].text(0.5, 0.5, "No building data", ha="center", va="center", transform=axes[1].transAxes)
    fig.suptitle("Land surface temperature and built-up intensity on the same grid", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.03, buildings_text() + " " + EDGE_NOTE + "\n" + source_text(), ha="center", fontsize=7, style="italic", wrap=True)
    _save(fig, _out(f"lst_and_building_density_{year}.png"), config.MAP_DPI)


# ----------------------------------------------------------------------- report
def _r(v):
    return "n/a" if pd.isna(v) else f"{v:.2f}"


def write_report(df, grid=None, summary=None):
    lab = area_labels()
    lines = ["# Ahmedabad urban heat — summary report", ""]
    if lab["district"]:
        lines += ["> **Warning:** the study area is the GAUL *district*, not the municipal city. "
                  "Results are for pipeline testing only.", ""]
    if df is not None:
        d = _observed(df)
        src_label = (f"{'Landsat 8 + 9' if config.USE_LANDSAT9 else 'Landsat 8'} Collection 2 Level-2 (USGS via GEE)"
                     if is_live_satellite_data() else
                     "Calibrated Offline Synthetic Proxy (Earth Engine offline fallback coupled with real building footprint density)")
        lines += [f"Period: {df.Date.min()} to {df.Date.max()} ({len(df)} months). "
                  f"Source: {src_label}.",
                  "", "## Data gaps", ""]
        counts = df.status.value_counts()
        lines.append(f"- Observed months: {counts.get('observed', 0)} of {len(df)}")
        for status, text in (("insufficient_coverage", "Insufficient cloud-free coverage (excluded from statistics)"),
                             ("no_data", "No valid observation"), ("extraction_failed", "Extraction failed")):
            sub = df[df.status == status]
            months = ", ".join(f"{r.Month_Name} {r.Year}" for r in sub.itertuples()) or "none"
            lines.append(f"- {text}: {months}")
        lines += ["", "No missing month was filled or interpolated.", "", "## Monthly LST", ""]
        if d.LST.notna().any():
            hot, cool = d.loc[d.LST.idxmax()], d.loc[d.LST.idxmin()]
            lines.append(f"- Hottest observed month: {hot.Month_Name} {hot.Year} ({hot.LST:.1f} °C study-area mean)")
            lines.append(f"- Coolest observed month: {cool.Month_Name} {cool.Year} ({cool.LST:.1f} °C)")
            tab, common = annual_comparison_table(df)
            for r in tab.itertuples():
                lines.append(f"- {r.Year}: {r.observed_months} observed months, mean {r.mean_LST_all_observed:.1f} °C; "
                             f"common-months mean ({r.n_common_months} months) "
                             + ("n/a" if pd.isna(r.mean_LST_common_months) else f"{r.mean_LST_common_months:.1f} °C"))
            tr = long_term_trend(df)
            lines.append("- Long-term trend: " + (f"{tr['slope']:+.2f} °C/yr (p = {tr['p']:.3f}, seasonally adjusted)"
                                                  if tr["justified"] else tr["reason"]))
            r, n = corr_table(d, ["LST", "NDVI", "NDBI"])
            lines.append(f"- Monthly-mean correlation with LST: NDVI r = {_r(r.loc['LST', 'NDVI'])}, "
                         f"NDBI r = {_r(r.loc['LST', 'NDBI'])} (n = {n.loc['LST', 'NDVI']} months; seasonal, not causal)")
    else:
        lines += ["## Satellite data", "", "- Not available: the Earth Engine extraction stage has not been run.", ""]
    if summary is not None:
        lines += ["", "## Buildings (uploaded Open Buildings CSV, static snapshot)", "",
                  "| Zone | Area km² | Buildings | Buildings/km² | Footprint km² | Footprint density % | Mean m² | Median m² |",
                  "|---|---|---|---|---|---|---|---|"]
        f = lambda v, p=0: "n/a" if pd.isna(v) else f"{v:,.{p}f}"  # noqa: E731
        for r in summary.itertuples():
            lines.append(f"| {r.zone} | {f(r.zone_area_km2, 1)} | {f(r.building_count)} | {f(r.buildings_per_km2)} | "
                         f"{f(r.total_footprint_km2, 2)} | {f(r.footprint_density_pct, 1)} | "
                         f"{f(r.mean_footprint_m2)} | {f(r.median_footprint_m2)} |")
        lines += ["", "Footprint density = total footprint area ÷ land area (%). Buildings/km² is a count density "
                  "and is a different measure."]
    else:
        lines += ["", "## Buildings", "", "- Not available: the Open Buildings CSV has not been processed."]
    if grid is not None and grid.year.notna().any() and grid.LST.notna().any():
        lines += ["", f"## Grid-level ({config.GRID_SCALE_M} m) relationships with LST", ""]
        for year in sorted(grid.year.dropna().unique()):
            g, pairs = _grid_columns(_grid_for_stats(grid, year))
            cols = [p[0] for p in pairs]
            if "LST" not in cols:
                continue
            r, n = corr_table(g, cols)
            rho, _ = corr_table(g, cols, "spearman")
            for c, label in pairs[1:]:
                lines.append(f"- {int(year)} — {label}: Pearson r = {_r(r.loc['LST', c])}, "
                             f"Spearman ρ = {_r(rho.loc['LST', c])}, n = {n.loc['LST', c]:,} cells")
        lines += ["", "Limitations: grid cells are spatially autocorrelated; building data are one static snapshot "
                  "while LST is year-specific; Landsat observes ~10:30 local time on clear days only. "
                  "Correlation does not show causation."]
    with open(config.REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("Saved %s", config.REPORT_MD)


def make_all_plots(df, grid=None, summary=None):
    """Draw everything the available data support, and say what was not drawn."""
    sns.set_theme(style="whitegrid")
    not_drawn = []
    if df is not None:
        plot_monthly_maps(df)
        plot_annual_maps(df)
        plot_yearly_trends(df)
        plot_cross_year(df)
        plot_long_term(df)
        plot_annual_comparison(df)
        plot_monthly_correlation(df)
    else:
        not_drawn.append("LST maps, trends and monthly correlations (no monthly_lst_metrics.csv - run extract)")

    if os.path.exists(config.BUILDINGS_GRID_CSV):
        bgrid = pd.read_csv(config.BUILDINGS_GRID_CSV)
        from data_pipeline.spatial_grid import make_grid
        bgrid = make_grid()[["grid_id", "x_utm", "y_utm", "edge_cell"]].merge(bgrid, on="grid_id", how="left")
        plot_building_maps(bgrid)
        plot_footprints()
    else:
        not_drawn.append("building maps (no processed building CSV - run buildings)")

    has_lst_grid = grid is not None and grid.year.notna().any() and grid.LST.notna().any()
    has_bld_grid = grid is not None and grid.building_count.notna().any()
    if has_lst_grid:
        years = [y for y in sorted(grid.year.dropna().unique()) if grid[grid.year == y].LST.notna().any()]
        for year in years:
            plot_grid_correlation(grid, int(year))
            plot_grid_scatter(grid, int(year))
            if has_bld_grid:
                plot_density_classes(grid, int(year))
        plot_lst_vs_density_maps(grid, int(years[-1]))
        if not has_bld_grid:
            not_drawn.append("LST-building correlations (no building data in the grid)")
    else:
        not_drawn.append("grid correlations and LST-building maps (no Landsat grid - run extract, then grid)")
    write_report(df, grid, summary)
    for item in not_drawn:
        log.warning("Not drawn: %s", item)
    return not_drawn
