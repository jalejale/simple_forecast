"""
app.py — Python Forecasting App (Browser-based via Streamlit)
Template: date, brand, sub_brand, qty

Features:
  - Upload CSV/Excel using the standard template (date, brand, sub_brand, qty)
  - Download blank CSV template
  - Cascading Brand → Sub-Brand filter
  - Report selector in sidebar — sidebar only shows relevant settings
  - 6 forecasting models: Moving Average, SES, Holt, Holt-Winters, SARIMA, Auto ARIMA
  - Seasonal decomposition
  - Interactive Plotly charts with 95% confidence intervals
  - Accuracy metrics (MAE, RMSE, MAPE)
  - Export forecast to Excel
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import forecasting as fc

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="📈 Forecasting App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Modern Dark Theme Base */
    .stApp {
        background-color: #0e1117;
        color: #f7f9fc;
    }
    
    /* Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1e23;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    
    /* Sleek Metric Cards */
    .metric-card {
        background: #1f232b;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover { 
        transform: translateY(-2px); 
        border-color: #6366f1;
    }
    .metric-card h3 { margin:0; font-size:0.8rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; }
    .metric-card h2 { margin:8px 0 0; font-size:1.8rem; font-weight:700; color:#f8fafc; }

    /* App Typography */
    .app-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .app-subtitle { color: #94a3b8; font-size: 1rem; margin-top: 4px; }

    /* Filter Badges */
    .filter-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.85rem;
        color: #c7d2fe;
        margin-right: 8px;
        margin-bottom: 12px;
        font-weight: 500;
    }

    /* Primary Buttons */
    .stButton > button {
        background-color: #6366f1;
        color: white; 
        border: none; 
        border-radius: 8px;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }
    .stButton > button:hover { background-color: #4f46e5; color: white; }

    .stDownloadButton > button {
        background-color: #10b981;
        color: white; border: none; border-radius: 8px;
        font-weight: 600; transition: background-color 0.2s ease;
    }
    .stDownloadButton > button:hover { background-color: #059669; color: white; }

    /* Alerts / Callouts */
    .stInfo { background-color: rgba(99, 102, 241, 0.1) !important; color: #e0e7ff !important; border: 1px solid rgba(99,102,241,0.3) !important; border-radius: 8px !important; }
    .stSuccess { background-color: rgba(16, 185, 129, 0.1) !important; color: #d1fae5 !important; border: 1px solid rgba(16,185,129,0.3) !important; border-radius: 8px !important; }
    .stWarning { background-color: rgba(245, 158, 11, 0.1) !important; color: #fef3c7 !important; border: 1px solid rgba(245,158,11,0.3) !important; border-radius: 8px !important; }

    hr { border-color: rgba(255,255,255,0.08); }
    
    /* Make native tabs beautiful */
    div[data-testid="stTabs"] button {
        font-size: 0.95rem;
        font-weight: 500;
        color: #94a3b8;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        color: #6366f1;
        font-weight: 600;
    }
    
    /* Make tab labels highly visible */
    p {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_COLS    = ["date", "brand", "sub_brand", "qty"]
SAMPLE_DATA_PATH = "sample_data.csv"

REPORTS = [
    "📊 Data Overview",
    "📉 Moving Average",
    "🔵 SES",
    "📐 Holt's Linear",
    "❄️ Holt-Winters",
    "🤖 SARIMA",
    "🔮 Auto ARIMA",
    "🔀 Decomposition",
]

COLORS = {
    "actual":   "#a5b4fc",
    "fitted":   "#f59e0b",
    "forecast": "#38ef7d",
    "ci":       "rgba(56,239,125,0.15)",
    "trend":    "#f64f59",
    "seasonal": "#667eea",
    "resid":    "#fb923c",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def blank_template_bytes() -> bytes:
    buf = io.StringIO()
    buf.write(",".join(TEMPLATE_COLS) + "\n")
    buf.write("2024-01-01,BrandA,SubBrand-1,100\n")
    buf.write("2024-02-01,BrandA,SubBrand-1,110\n")
    return buf.getvalue().encode("utf-8")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=True, sheet_name="Forecast")
    return buf.getvalue()


def metrics_html(metrics: dict, cols: int = 3) -> str:
    items = "".join(
        f'<div class="metric-card"><h3>{k}</h3><h2>{v}</h2></div>'
        for k, v in metrics.items()
    )
    return (f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);'
            f'gap:12px;margin-bottom:16px">{items}</div>')


def make_forecast_figure(series, result, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        name="Actual", mode="lines+markers",
        line=dict(color=COLORS["actual"], width=2), marker=dict(size=5),
    ))
    fitted = result.get("fitted")
    if fitted is not None:
        fig.add_trace(go.Scatter(
            x=fitted.index, y=fitted.values, name="Fitted", mode="lines",
            line=dict(color=COLORS["fitted"], width=1.5, dash="dot"),
        ))
    lower, upper, fc_val = result.get("lower"), result.get("upper"), result.get("forecast")
    if lower is not None and upper is not None and fc_val is not None:
        fig.add_trace(go.Scatter(
            x=list(fc_val.index) + list(fc_val.index[::-1]),
            y=list(upper.values) + list(lower.values[::-1]),
            fill="toself", fillcolor=COLORS["ci"],
            line=dict(color="rgba(0,0,0,0)"), name="95% CI",
        ))
    if fc_val is not None:
        fig.add_trace(go.Scatter(
            x=fc_val.index, y=fc_val.values,
            name="Forecast", mode="lines+markers",
            line=dict(color=COLORS["forecast"], width=2.5),
            marker=dict(size=6, symbol="diamond"),
        ))
    fig.add_vline(x=series.index[-1], line_dash="dash",
                  line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(color="#e8eaf6", size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#a5b4fc"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#a5b4fc"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)",
                    font=dict(color="#e8eaf6")),
        hovermode="x unified", height=430,
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig


def make_decomp_figure(decomp_result):
    fig = make_subplots(rows=4, cols=1,
                        subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
                        shared_xaxes=True, vertical_spacing=0.07)
    for s, color, row in [
        (decomp_result.observed, COLORS["actual"], 1),
        (decomp_result.trend,    COLORS["trend"],   2),
        (decomp_result.seasonal, COLORS["seasonal"], 3),
        (decomp_result.resid,    COLORS["resid"],   4),
    ]:
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                 line=dict(color=color, width=1.8), showlegend=False),
                      row=row, col=1)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(255,255,255,0.03)",
                      height=680, margin=dict(t=30, b=30, l=50, r=20))
    for ax in fig.layout:
        if "xaxis" in ax or "yaxis" in ax:
            fig.layout[ax].update(gridcolor="rgba(255,255,255,0.08)", color="#a5b4fc")
    for ann in fig.layout.annotations:
        ann.font.color = "#a5b4fc"
    return fig


def render_standard_forecast(label, run_fn, series, periods, extra_kwargs=None,
                              needs_button=False, btn_key=None):
    if extra_kwargs is None:
        extra_kwargs = {}

    def _run():
        with st.spinner(f"Fitting {label}..."):
            try:
                result = run_fn(series, periods=periods, **extra_kwargs)
                st.markdown(metrics_html(result["metrics"]), unsafe_allow_html=True)
                fc_df = pd.DataFrame({
                    "Forecast": result["forecast"],
                    "Lower 95%": result["lower"],
                    "Upper 95%": result["upper"],
                })
                st.plotly_chart(
                    make_forecast_figure(series, result, f"{label} — {periods}-Period Forecast"),
                    use_container_width=True,
                )
                with st.expander("📋 Forecast Table"):
                    st.dataframe(fc_df.style.format("{:.2f}"), use_container_width=True)
                st.download_button(
                    "⬇️ Download Forecast (Excel)", to_excel_bytes(fc_df),
                    f"{(btn_key or label).lower().replace(' ','_')}_forecast.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_{btn_key or label}",
                )
                if "summary" in result:
                    with st.expander("📄 Model Summary"):
                        st.code(result["summary"], language="text")
            except Exception as e:
                st.error(f"{label} error: {e}")

    if needs_button:
        if st.button(f"🚀 Run {label}", key=f"btn_{btn_key or label}"):
            _run()
        else:
            st.info(f"📌 Adjust parameters in the sidebar, then click **Run {label}**.")
    else:
        _run()


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_sample():
    df = pd.read_csv(SAMPLE_DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_uploaded(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported format.")
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in TEMPLATE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {TEMPLATE_COLS}")
    df["date"] = pd.to_datetime(df["date"])
    df["qty"]  = pd.to_numeric(df["qty"], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Forecasting App")
    st.markdown("---")

    # ── Template Download ────────────────────────────────────────────────────
    st.markdown("### 📄 Template")
    st.download_button(
        "⬇️ Download CSV Template",
        data=blank_template_bytes(),
        file_name="forecast_template.csv",
        mime="text/csv",
    )
    st.caption("`date` · `brand` · `sub_brand` · `qty`")
    st.markdown("---")

    # ── Data Source ──────────────────────────────────────────────────────────
    st.markdown("### 📂 Data Source")
    data_source = st.radio("", ["Use Sample Data", "Upload File"],
                           label_visibility="collapsed")

    raw_df = None
    if data_source == "Upload File":
        uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if uploaded:
            try:
                raw_df = load_uploaded(uploaded)
                st.success(f"✔ {len(raw_df)} rows loaded")
            except Exception as e:
                st.error(str(e))
    else:
        raw_df = load_sample()
        st.info(f"ℹ Sample: {len(raw_df)} rows | 2019–2023")

    # ── Brand / Sub-Brand Filter ─────────────────────────────────────────────
    series = None
    selected_brand = None
    selected_sub_brand = None
    sub_brands = []

    if raw_df is not None:
        st.markdown("### 🏷️ Filter")
        brands = sorted(raw_df["brand"].dropna().unique().tolist())
        selected_brand = st.selectbox("Brand", brands)

        sub_brands = sorted(
            raw_df.loc[raw_df["brand"] == selected_brand, "sub_brand"]
            .dropna().unique().tolist()
        )
        selected_sub_brand = st.selectbox("Sub-Brand", sub_brands)

        freq_map = {
            "Monthly (MS)":   "MS",
            "Daily (D)":      "D",
            "Weekly (W)":     "W",
            "Quarterly (QS)": "QS",
        }
        freq_label = st.selectbox("Frequency", list(freq_map.keys()))
        freq = freq_map[freq_label]

        mask = (
            (raw_df["brand"] == selected_brand) &
            (raw_df["sub_brand"] == selected_sub_brand)
        )
        filtered = raw_df[mask].copy().sort_values("date").set_index("date")["qty"]
        try:
            series = filtered.asfreq(freq).interpolate()
        except Exception as e:
            st.error(f"Frequency error: {e}")




# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="app-title">📈 Time Series Forecasting</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Select a report tab below to configure and view forecasts</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

if series is None or series.dropna().empty:
    st.warning("⚠️ Please load data using the sidebar.")
    st.stop()

# Active filter badges
st.markdown(
    f'<div style="margin-bottom:12px">'
    f'<span class="filter-badge">🏷️ Brand: <b>{selected_brand}</b></span>'
    f'<span class="filter-badge">📦 Sub-Brand: <b>{selected_sub_brand}</b></span>'
    f'<span class="filter-badge">📅 {len(series.dropna())} obs</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# Render actual Streamlit Tabs
tabs = st.tabs(REPORTS)

# ── 📊 Data Overview ─────────────────────────────────────────────────────────
with tabs[0]:
    s_stat = [
        ("Observations", str(len(series.dropna()))),
        ("Min Qty",  f"{series.min():,.0f}"),
        ("Max Qty",  f"{series.max():,.0f}"),
        ("Mean Qty", f"{series.mean():,.1f}"),
    ]
    for col, (label, value) in zip(st.columns(4), s_stat):
        with col:
            st.markdown(
                f'<div class="metric-card"><h3>{label}</h3><h2>{value}</h2></div>',
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader(f"Qty Over Time — {selected_brand} / {selected_sub_brand}")
    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines+markers",
        line=dict(color=COLORS["actual"], width=2),
        marker=dict(size=6), name="Qty",
    ))
    fig0.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#f8fafc"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#f8fafc", title="Quantity"),
        hovermode="x unified", height=380,
        margin=dict(t=20, b=40, l=50, r=20),
    )
    st.plotly_chart(fig0, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Descriptive Statistics**")
        st.dataframe(series.describe().rename("Qty").to_frame()
                     .style.format("{:.2f}"), use_container_width=True)
    with col_b:
        st.markdown("**Raw Data (last 24)**")
        df_show = series.tail(24).reset_index()
        df_show.columns = ["Date", "Qty"]
        df_show["Date"] = df_show["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(f"All Sub-Brands of '{selected_brand}'")
    palette = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]
    fig_cmp = go.Figure()
    for i, sb in enumerate(sub_brands):
        m = (raw_df["brand"] == selected_brand) & (raw_df["sub_brand"] == sb)
        s_ = raw_df[m].sort_values("date").set_index("date")["qty"]
        fig_cmp.add_trace(go.Scatter(
            x=s_.index, y=s_.values, name=sb, mode="lines",
            line=dict(color=palette[i % len(palette)], width=2),
        ))
    fig_cmp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#f8fafc"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#f8fafc"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="#f8fafc")),
        hovermode="x unified", height=380,
        margin=dict(t=20, b=40, l=50, r=20),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)


# ── 📉 Moving Average ─────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Moving Average")
    col1, col2 = st.columns([1, 2])
    with col1:
        periods = st.number_input("Forecast Periods", 1, 36, 12, key="ma_p")
        ma_window = st.number_input("Window", 2, 24, 3, key="ma_w")
    with col2:
        st.caption("Settings configured here apply only to this model.")
        
    st.markdown("---")
    render_standard_forecast(
        "Moving Average", fc.moving_average, series, periods,
        extra_kwargs={"window": ma_window}, btn_key="ma",
    )


# ── 🔵 SES ───────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Simple Exponential Smoothing")
    col1, col2 = st.columns([1, 2])
    with col1:
        periods = st.number_input("Forecast Periods", 1, 36, 12, key="ses_p")
        alpha = st.slider("Alpha (α)", 0.01, 0.99, 0.3, key="ses_alpha")
        
    st.markdown("---")
    render_standard_forecast(
        "SES", fc.ses_forecast, series, periods,
        extra_kwargs={"alpha": alpha}, btn_key="ses",
    )


# ── 📐 Holt's Linear ─────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Holt's Linear Trend")
    periods = st.number_input("Forecast Periods", 1, 36, 12, key="holt_p")
    st.markdown("---")
    render_standard_forecast("Holt's Linear", fc.holt_forecast, series, periods,
                             btn_key="holt")


# ── ❄️ Holt-Winters ──────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Holt-Winters (Triple Exponential Smoothing)")
    col1, col2, col3 = st.columns(3)
    with col1:
        periods = st.number_input("Forecast Periods", 1, 36, 12, key="hw_p")
    with col2:
        seasonal_periods = st.number_input("Seasonal Periods", 4, 52, 12, key="hw_sp")
    with col3:
        hw_trend    = st.selectbox("Trend",   ["add", "mul", None], index=0, key="hw_t")
        hw_seasonal = st.selectbox("Seasonal", ["add", "mul"],       index=0, key="hw_s")
        
    st.markdown("---")
    render_standard_forecast(
        "Holt-Winters", fc.holtwinters_forecast, series, periods,
        extra_kwargs={"seasonal_periods": seasonal_periods,
                      "trend": hw_trend, "seasonal": hw_seasonal},
        btn_key="hw",
    )


# ── 🤖 SARIMA ────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("SARIMA")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        periods = st.number_input("Forecast", 1, 36, 12, key="sar_p")
        seasonal_periods = st.number_input("Season", 4, 52, 12, key="sar_sp")
    with col2:
        p = st.number_input("p", 0, 3, 1, step=1, key="sar_p_val")
        P = st.number_input("P", 0, 2, 1, step=1, key="sar_P_val")
    with col3:
        d = st.number_input("d", 0, 2, 1, step=1, key="sar_d_val")
        D = st.number_input("D", 0, 2, 1, step=1, key="sar_D_val")
    with col4:
        q = st.number_input("q", 0, 3, 1, step=1, key="sar_q_val")
        Q = st.number_input("Q", 0, 2, 1, step=1, key="sar_Q_val")
        
    sarima_label = f"SARIMA ({p},{d},{q})×({P},{D},{Q},{seasonal_periods})"
    st.markdown("---")
    render_standard_forecast(
        sarima_label, fc.sarima_forecast, series, periods,
        extra_kwargs={"order": (int(p), int(d), int(q)),
                      "seasonal_order": (int(P), int(D), int(Q), seasonal_periods)},
        needs_button=True, btn_key="sarima",
    )


# ── 🔮 Auto ARIMA ────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Auto ARIMA — Automatic Order Selection")
    col1, col2, col3 = st.columns(3)
    with col1:
        periods = st.number_input("Forecast Periods", 1, 36, 12, key="aa_p")
        aa_m = st.number_input("Seasonal Period (m)", 1, 52, 12, step=1, key="aa_m")
    with col2:
        aa_criterion = st.selectbox("Information Criterion", ["aic", "bic", "aicc", "oob"], key="aa_crit")
    with col3:
        st.write("")
        st.write("")
        aa_seasonal  = st.checkbox("Include Seasonal", value=True, key="aa_seas")
        aa_stepwise  = st.checkbox("Stepwise Search", value=True, key="aa_step")
        
    st.markdown("---")
    if st.button("🚀 Run Auto ARIMA", key="btn_auto_arima"):
        with st.spinner("Searching for best ARIMA order..."):
            try:
                result_aa = fc.auto_arima_forecast(
                    series,
                    seasonal=aa_seasonal,
                    m=int(aa_m),
                    periods=periods,
                    stepwise=aa_stepwise,
                    information_criterion=aa_criterion,
                )
                st.success(f"✅ {result_aa['order_str']}")

                o, so = result_aa["order"], result_aa["seasonal_order"]
                order_label = (
                    f"ARIMA({o[0]},{o[1]},{o[2]})×({so[0]},{so[1]},{so[2]},{so[3]})"
                    if aa_seasonal else f"ARIMA({o[0]},{o[1]},{o[2]})"
                )
                met = result_aa["metrics"]
                st.markdown(
                    f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px">'
                    f'<div class="metric-card"><h3>Best Order</h3>'
                    f'<h2 style="font-size:1.05rem;margin-top:8px">{order_label}</h2></div>'
                    f'<div class="metric-card"><h3>MAE</h3><h2>{met["MAE"]}</h2></div>'
                    f'<div class="metric-card"><h3>RMSE</h3><h2>{met["RMSE"]}</h2></div>'
                    f'<div class="metric-card"><h3>MAPE (%)</h3><h2>{met["MAPE (%)"]}</h2></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    make_forecast_figure(series, result_aa,
                                         f"Auto ARIMA {order_label} — {periods}-Period Forecast"),
                    use_container_width=True,
                )
                fc_df = pd.DataFrame({
                    "Forecast": result_aa["forecast"],
                    "Lower 95%": result_aa["lower"],
                    "Upper 95%": result_aa["upper"],
                })
                with st.expander("📋 Forecast Table"):
                    st.dataframe(fc_df.style.format("{:.2f}"), use_container_width=True)
                st.download_button(
                    "⬇️ Download Forecast (Excel)", to_excel_bytes(fc_df),
                    "auto_arima_forecast.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_auto_arima",
                )
                with st.expander("📄 Auto ARIMA Model Summary"):
                    st.code(result_aa["summary"], language="text")
            except Exception as e:
                st.error(f"Auto ARIMA failed: {e}")
    else:
        st.info("📌 Adjust parameters above, then click **Run Auto ARIMA**.")


# ── 🔀 Decomposition ─────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader(f"Seasonal Decomposition")
    col1, col2 = st.columns([1, 2])
    with col1:
        seasonal_periods = st.number_input("Seasonal Periods", 4, 52, 12, key="decomp_sp")
        decomp_model = st.selectbox("Model", ["additive", "multiplicative"], key="decomp_model")
        
    st.markdown("---")
    with st.spinner("Decomposing..."):
        try:
            clean = series.dropna()
            if len(clean) < seasonal_periods * 2:
                st.warning(f"Need at least {seasonal_periods * 2} observations.")
            else:
                decomp = fc.decompose_series(clean, model=decomp_model,
                                             period=seasonal_periods)
                st.plotly_chart(make_decomp_figure(decomp), use_container_width=True)
                strength_t = max(0, 1 - decomp.resid.var() /
                                 (decomp.trend.dropna() + decomp.resid.dropna()).var())
                strength_s = max(0, 1 - decomp.resid.var() /
                                 (decomp.seasonal + decomp.resid.dropna()).var())
                st.markdown(metrics_html({
                    "Trend Strength":    round(float(strength_t), 3),
                    "Seasonal Strength": round(float(strength_s), 3),
                    "Residual Std":      round(float(decomp.resid.std()), 3),
                }), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Decomposition error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#555;font-size:0.8rem;">'
    "Python Forecasting App · Template: date | brand | sub_brand | qty"
    "</p>",
    unsafe_allow_html=True,
)
