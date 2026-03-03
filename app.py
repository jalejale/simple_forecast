"""
app.py — Python Forecasting App (Dash)
Template: date, brand, sub_brand, qty

Run: python app.py
Opens: http://localhost:8050
"""

import io
import base64

import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import forecasting as fc

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_COLS    = ["date", "brand", "sub_brand", "qty"]
SAMPLE_DATA_PATH = "sample_data.csv"

COLORS = {
    "actual":   "#a5b4fc",
    "fitted":   "#f59e0b",
    "forecast": "#38ef7d",
    "ci":       "rgba(56,239,125,0.15)",
    "trend":    "#f64f59",
    "seasonal": "#667eea",
    "resid":    "#fb923c",
}

PALETTE = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

FREQ_MAP = {
    "Monthly (MS)":   "MS",
    "Daily (D)":      "D",
    "Weekly (W)":     "W",
    "Quarterly (QS)": "QS",
}

METRIC_HINTS = {
    "MAE": "Mean Absolute Error: The average absolute difference between forecast and actuals. Lower is better.",
    "RMSE": "Root Mean Squared Error: The square root of average squared errors. Penalizes larger errors more than MAE. Lower is better.",
    "MAPE": "Mean Absolute Percentage Error: The average percentage difference between forecast and actuals. Lower is better.",
    "OBSERVATIONS": "Total number of recorded data points in this historical series.",
    "MIN QTY": "The lowest recorded quantity in the historical data.",
    "MAX QTY": "The highest recorded quantity in the historical data.",
    "MEAN QTY": "The average quantity over the historical data.",
    "TREND STRENGTH": "Measures how much variance is explained by the trend (0 to 1). Higher = stronger trend.",
    "SEASONAL STRENGTH": "Measures how much variance is explained by seasonality (0 to 1). Higher = stronger seasonality.",
    "RESIDUAL STD": "Standard deviation of the residuals (noise/error). Lower means the model captures more signal."
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


def parse_uploaded(contents, filename):
    """Decode a dcc.Upload content string into a DataFrame."""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    name = filename.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(decoded))
    else:
        raise ValueError("Unsupported file format.")
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in TEMPLATE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df["qty"]  = pd.to_numeric(df["qty"], errors="coerce")
    return df


def load_sample() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_series(df: pd.DataFrame, brand: str, sub_brand: str, freq: str) -> pd.Series:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["brand"] == brand) & (df["sub_brand"] == sub_brand)
    filtered = df[mask].copy().sort_values("date").set_index("date")["qty"]
    filtered.index = pd.DatetimeIndex(filtered.index)
    return filtered.asfreq(freq).interpolate()


def dark_layout_kwargs(height=420, title_text=None):
    kwargs = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#a5b4fc"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#a5b4fc"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.1)",
                    font=dict(color="#e8eaf6")),
        hovermode="x unified",
        height=height,
        margin=dict(t=50 if title_text else 20, b=40, l=50, r=20),
        font=dict(family="Inter, sans-serif", color="#e8eaf6"),
    )
    if title_text:
        kwargs["title"] = dict(text=f"<b>{title_text}</b>",
                               font=dict(color="#e8eaf6", size=16))
    return kwargs


def forecast_figure(series, result, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, name="Actual", mode="lines+markers",
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
        fig.add_trace(go.Scatter(
            x=fc_val.index, y=fc_val.values, name="Forecast", mode="lines+markers",
            line=dict(color=COLORS["forecast"], width=2.5),
            marker=dict(size=6, symbol="diamond"),
        ))
    fig.add_vline(x=series.index[-1], line_dash="dash",
                  line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.update_layout(**dark_layout_kwargs(title_text=title))
    return fig


def metrics_div(metrics: dict, cols: int = 3):
    cards = [
        html.Div([
            html.H3([k, " ", html.Span("ℹ️", title=METRIC_HINTS.get(k, ""), style={"cursor": "help", "opacity": 0.6})]),
            html.H2(str(v)),
        ], className="metric-card")
        for k, v in metrics.items()
    ]
    return html.Div(cards, className="metrics-grid")


def forecast_div(series, result, label, periods, dl_id):
    """Return the standard forecast output: metrics + chart + table + download."""
    met = result.get("metrics", {})
    fc_df = pd.DataFrame({
        "Forecast":  result["forecast"],
        "Lower 95%": result["lower"],
        "Upper 95%": result["upper"],
    }).reset_index()
    fc_df.columns = ["Date", "Forecast", "Lower 95%", "Upper 95%"]
    fc_df["Date"] = fc_df["Date"].astype(str)

    return html.Div([
        metrics_div(met),
        dcc.Graph(
            figure=forecast_figure(series, result, f"{label} — {periods}-Period Forecast"),
            config={"displayModeBar": False},
        ),
        html.Details([
            html.Summary("📋 Forecast Table", style={"cursor": "pointer", "color": "#94a3b8",
                                                       "fontSize": "0.9rem", "marginBottom": "8px"}),
            dash_table.DataTable(
                data=fc_df.round(2).to_dict("records"),
                columns=[{"name": c, "id": c} for c in fc_df.columns],
                style_header={"backgroundColor": "#374151", "color": "#94a3b8",
                              "fontWeight": "600", "fontSize": "0.78rem"},
                style_data={"backgroundColor": "#1f2937", "color": "#e2e8f0", "fontSize": "0.85rem"},
                style_cell={"border": "1px solid rgba(255,255,255,0.08)"},
                page_size=12,
            ),
        ], style={"marginTop": "10px"}),
        html.Div([
            html.Button("⬇️ Download Forecast (Excel)", id=dl_id,
                        className="btn-primary",
                        style={"marginTop": "14px"}),
            dcc.Download(id=f"{dl_id}-download"),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="📈 Forecasting App",
    suppress_callback_exceptions=True,
)
server = app.server   # expose Flask server for deployment

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def sidebar():
    return html.Div(id="sidebar", children=[
        html.H2("📈 Forecasting App"),
        html.Hr(className="side-divider"),

        # ── Template ──
        html.P("📄 TEMPLATE", className="side-section-title"),
        html.Button("⬇️ Download CSV Template", id="btn-template",
                    className="btn-download"),
        dcc.Download(id="download-template"),
        html.P("`date` · `brand` · `sub_brand` · `qty`", className="side-caption"),
        html.Hr(className="side-divider"),

        # ── Data Source ──
        html.P("📂 DATA SOURCE", className="side-section-title"),
        dcc.RadioItems(
            id="data-source",
            options=[
                {"label": " Use Sample Data", "value": "sample"},
                {"label": " Upload File",      "value": "upload"},
            ],
            value="sample",
            className="radio-group",
            labelStyle={"display": "block", "marginBottom": "6px"},
        ),
        dcc.Upload(
            id="upload-box",
            children=html.Div("📁 Drag & drop or click to upload CSV / Excel"),
            accept=".csv,.xlsx,.xls",
            style={"display": "none"},
        ),
        html.Div(id="upload-info"),
        html.Hr(className="side-divider"),

        # ── Filters ──
        html.P("🏷️ FILTER", className="side-section-title"),
        html.Label("Brand", className="side-section-title"),
        dcc.Dropdown(id="brand-dd", clearable=False,
                     style={"marginBottom": "10px"}),
        html.Label("Sub-Brand", className="side-section-title"),
        dcc.Dropdown(id="subbrand-dd", clearable=False,
                     style={"marginBottom": "10px"}),
        html.Label("Frequency", className="side-section-title"),
        dcc.Dropdown(
            id="freq-dd",
            options=[{"label": k, "value": v} for k, v in FREQ_MAP.items()],
            value="MS", clearable=False,
        ),

        # Hidden data store
        dcc.Store(id="store-df"),
    ])


def main_content():
    return html.Div(id="main-content", children=[
        html.P("📈 Time Series Forecasting", className="app-title"),
        html.P("Select a report tab below to configure and view forecasts", className="app-subtitle"),

        # Active filter badges (updated by callback)
        html.Div(id="filter-badges", className="filter-badges"),

        # ── Tabs ──
        dcc.Tabs(id="report-tabs", value="tab-overview", className="custom-tabs", children=[
            dcc.Tab(label="📊 Data Overview",   value="tab-overview",     className="tab", selected_className="tab--selected"),
            dcc.Tab(label="📉 Moving Average",  value="tab-ma",           className="tab", selected_className="tab--selected"),
            dcc.Tab(label="🔵 SES",             value="tab-ses",          className="tab", selected_className="tab--selected"),
            dcc.Tab(label="📐 Holt's Linear",   value="tab-holt",         className="tab", selected_className="tab--selected"),
            dcc.Tab(label="❄️ Holt-Winters",    value="tab-hw",           className="tab", selected_className="tab--selected"),
            dcc.Tab(label="🤖 SARIMA",          value="tab-sarima",       className="tab", selected_className="tab--selected"),
            dcc.Tab(label="🔮 Auto ARIMA",      value="tab-auto-arima",   className="tab", selected_className="tab--selected"),
            dcc.Tab(label="🔀 Decomposition",   value="tab-decomp",       className="tab", selected_className="tab--selected"),
        ]),

        # Dynamic content for fast tabs
        html.Div(id="tab-content", className="tab-content"),
        
        # Persistent containers for slow tabs
        html.Div(id="tab-sarima-container", className="tab-content", style={"display": "none"}, children=[sarima_tab()]),
        html.Div(id="tab-aa-container", className="tab-content", style={"display": "none"}, children=[auto_arima_tab()]),

        html.Div([
            html.P("Python Forecasting App · Template: date | brand | sub_brand | qty")
        ], id="footer"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

# ── Toggle upload box visibility ──────────────────────────────────────────────
@app.callback(
    Output("upload-box", "style"),
    Input("data-source", "value"),
)
def toggle_upload(source):
    if source == "upload":
        return {"display": "block"}
    return {"display": "none"}


# ── Load raw data into store ───────────────────────────────────────────────────
@app.callback(
    Output("store-df", "data"),
    Output("upload-info", "children"),
    Input("data-source", "value"),
    Input("upload-box", "contents"),
    State("upload-box", "filename"),
)
def load_data(source, contents, filename):
    if source == "sample":
        df = load_sample()
        df["date"] = df["date"].astype(str)  # serialize date as string for Store
        info = html.Div(f"ℹ Sample: {len(df)} rows | 2019–2023", className="data-info")
        return df.to_json(date_format="iso", orient="split"), info
    if contents:
        try:
            df = parse_uploaded(contents, filename)
            df["date"] = df["date"].astype(str)  # serialize date as string for Store
            info = html.Div(f"✔ {len(df)} rows loaded from {filename}", className="data-info alert-success")
            return df.to_json(date_format="iso", orient="split"), info
        except Exception as e:
            info = html.Div(f"❌ {e}", className="alert-error")
            return None, info
    return None, ""


# ── Brand dropdown ─────────────────────────────────────────────────────────────
@app.callback(
    Output("brand-dd", "options"),
    Output("brand-dd", "value"),
    Input("store-df", "data"),
)
def update_brands(data):
    if not data:
        return [], None
    df = pd.read_json(io.StringIO(data), orient="split")
    brands = sorted(df["brand"].dropna().unique().tolist())
    opts = [{"label": b, "value": b} for b in brands]
    return opts, brands[0] if brands else None


# ── Sub-brand dropdown ─────────────────────────────────────────────────────────
@app.callback(
    Output("subbrand-dd", "options"),
    Output("subbrand-dd", "value"),
    Input("brand-dd", "value"),
    State("store-df", "data"),
)
def update_subbrands(brand, data):
    if not data or not brand:
        return [], None
    df = pd.read_json(io.StringIO(data), orient="split")
    sbs = sorted(df.loc[df["brand"] == brand, "sub_brand"].dropna().unique().tolist())
    opts = [{"label": s, "value": s} for s in sbs]
    return opts, sbs[0] if sbs else None


# ── Filter badges ──────────────────────────────────────────────────────────────
@app.callback(
    Output("filter-badges", "children"),
    Input("brand-dd", "value"),
    Input("subbrand-dd", "value"),
    Input("store-df", "data"),
)
def update_badges(brand, sub_brand, data):
    if not (brand and sub_brand and data):
        return []
    df = pd.read_json(io.StringIO(data), orient="split")
    mask = (df["brand"] == brand) & (df["sub_brand"] == sub_brand)
    n = mask.sum()
    return [
        html.Span(f"🏷️ Brand: {brand}",          className="filter-badge"),
        html.Span(f"📦 Sub-Brand: {sub_brand}",   className="filter-badge"),
        html.Span(f"📅 {n} obs",                  className="filter-badge"),
    ]


# ── Template download ──────────────────────────────────────────────────────────
@app.callback(
    Output("download-template", "data"),
    Input("btn-template", "n_clicks"),
    prevent_initial_call=True,
)
def download_template(n):
    if not n:
        raise PreventUpdate
    return dcc.send_bytes(blank_template_bytes, "forecast_template.csv")


# ── Tab content router ─────────────────────────────────────────────────────────
@app.callback(
    Output("tab-content", "children"),
    Output("tab-content", "style"),
    Output("tab-sarima-container", "style"),
    Output("tab-aa-container", "style"),
    Input("report-tabs", "value"),
    Input("brand-dd", "value"),
    Input("subbrand-dd", "value"),
    Input("freq-dd", "value"),
    State("store-df", "data"),
)
def render_tab(tab, brand, sub_brand, freq, data):
    show_fast = {"display": "block"}
    show_sar  = {"display": "block"} if tab == "tab-sarima" else {"display": "none"}
    show_aa   = {"display": "block"} if tab == "tab-auto-arima" else {"display": "none"}
    hide_fast = {"display": "none"}

    if tab in ["tab-sarima", "tab-auto-arima"]:
        return dash.no_update, hide_fast, show_sar, show_aa

    if not (data and brand and sub_brand and freq):
        err = html.Div("⚠️ Please load data and select Brand / Sub-Brand from the sidebar.", className="alert-warning")
        return err, show_fast, show_sar, show_aa
    try:
        df = pd.read_json(io.StringIO(data), orient="split")
        series = get_series(df, brand, sub_brand, freq)
    except Exception as e:
        err = html.Div(f"❌ Error loading data: {e}", className="alert-error")
        return err, show_fast, show_sar, show_aa

    if tab == "tab-overview":
        content = overview_tab(df, series, brand, sub_brand)
    elif tab == "tab-ma":
        content = ma_tab()
    elif tab == "tab-ses":
        content = ses_tab()
    elif tab == "tab-holt":
        content = holt_tab()
    elif tab == "tab-hw":
        content = hw_tab()
    elif tab == "tab-decomp":
        content = decomp_tab(series)
    else:
        content = html.Div("Unknown tab.")
    
    return content, show_fast, show_sar, show_aa


# ─────────────────────────────────────────────────────────────────────────────
# TAB LAYOUTS (static controls — charts rendered via model callbacks)
# ─────────────────────────────────────────────────────────────────────────────

def overview_tab(df, series, brand, sub_brand):
    n   = len(series.dropna())
    mn  = series.min()
    mx  = series.max()
    avg = series.mean()

    # Time series chart
    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(x=series.index, y=series.values,
                              mode="lines+markers",
                              line=dict(color=COLORS["actual"], width=2),
                              marker=dict(size=6), name="Qty"))
    fig0.update_layout(**dark_layout_kwargs(height=380,
                                            title_text=f"Qty Over Time — {brand} / {sub_brand}"))

    # Sub-brand comparison
    sub_brands = sorted(df.loc[df["brand"] == brand, "sub_brand"].dropna().unique().tolist())
    fig_cmp = go.Figure()
    for i, sb in enumerate(sub_brands):
        m_ = (df["brand"] == brand) & (df["sub_brand"] == sb)
        s_ = df[m_].sort_values("date").set_index("date")["qty"]
        fig_cmp.add_trace(go.Scatter(
            x=s_.index, y=s_.values, name=sb, mode="lines",
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        ))
    fig_cmp.update_layout(**dark_layout_kwargs(height=380,
                                               title_text=f"All Sub-Brands of '{brand}'"))

    # Stats table
    stats = series.describe().reset_index()
    stats.columns = ["Stat", "Qty"]
    stats["Qty"] = stats["Qty"].round(2)

    # Raw last-24 table
    raw24 = series.tail(24).reset_index()
    raw24.columns = ["Date", "Qty"]
    raw24["Date"] = raw24["Date"].astype(str)

    return html.Div([
        html.Div([
            html.Div([html.H3(["OBSERVATIONS ", html.Span("ℹ️", title=METRIC_HINTS.get("OBSERVATIONS",""), style={"cursor": "help", "opacity": 0.6})]), html.H2(str(n))],   className="metric-card"),
            html.Div([html.H3(["MIN QTY ", html.Span("ℹ️", title=METRIC_HINTS.get("MIN QTY",""), style={"cursor": "help", "opacity": 0.6})]),      html.H2(f"{mn:,.0f}")],  className="metric-card"),
            html.Div([html.H3(["MAX QTY ", html.Span("ℹ️", title=METRIC_HINTS.get("MAX QTY",""), style={"cursor": "help", "opacity": 0.6})]),      html.H2(f"{mx:,.0f}")],  className="metric-card"),
            html.Div([html.H3(["MEAN QTY ", html.Span("ℹ️", title=METRIC_HINTS.get("MEAN QTY",""), style={"cursor": "help", "opacity": 0.6})]),     html.H2(f"{avg:,.1f}")], className="metric-card"),
        ], className="metrics-grid"),
        dcc.Graph(figure=fig0, config={"displayModeBar": False}),
        html.Div([
            html.Div([
                html.H4("Descriptive Statistics", className="sub-head"),
                dash_table.DataTable(
                    data=stats.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in stats.columns],
                    style_header={"backgroundColor": "#374151", "color": "#94a3b8",
                                  "fontWeight": "600", "fontSize": "0.78rem"},
                    style_data={"backgroundColor": "#1f2937", "color": "#e2e8f0",
                                "fontSize": "0.85rem"},
                    style_cell={"border": "1px solid rgba(255,255,255,0.08)"},
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.H4("Raw Data (last 24)", className="sub-head"),
                dash_table.DataTable(
                    data=raw24.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in raw24.columns],
                    style_header={"backgroundColor": "#374151", "color": "#94a3b8",
                                  "fontWeight": "600", "fontSize": "0.78rem"},
                    style_data={"backgroundColor": "#1f2937", "color": "#e2e8f0",
                                "fontSize": "0.85rem"},
                    style_cell={"border": "1px solid rgba(255,255,255,0.08)"},
                    page_size=12,
                ),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "24px", "marginTop": "16px"}),
        dcc.Graph(figure=fig_cmp, config={"displayModeBar": False}, style={"marginTop": "24px"}),
    ])


def settings_row(*items):
    """Wrap settings items in a flex row."""
    return html.Div(items, className="settings-row")


def setting(label, component):
    return html.Div([html.Label(label), component], className="settings-item")


# ── Moving Average tab layout ──────────────────────────────────────────────────
def ma_tab():
    return html.Div([
        html.H3("Moving Average", className="section-head"),
        settings_row(
            setting("Forecast Periods", dcc.Input(id="ma-periods", type="number", value=12, min=1, max=36, step=1)),
            setting("Window Size",      dcc.Input(id="ma-window",  type="number", value=3,  min=2, max=24, step=1)),
        ),
        html.Hr(),
        html.Div(id="ma-output"),
        dcc.Download(id="ma-download"),
    ])


# ── SES tab layout ─────────────────────────────────────────────────────────────
def ses_tab():
    return html.Div([
        html.H3("Simple Exponential Smoothing (SES)", className="section-head"),
        settings_row(
            setting("Forecast Periods", dcc.Input(id="ses-periods", type="number", value=12, min=1, max=36, step=1)),
            setting("Alpha (α)",        html.Div([
                dcc.Slider(id="ses-alpha", min=0.01, max=0.99, step=0.01, value=0.3,
                           marks={0.1: "0.1", 0.3: "0.3", 0.5: "0.5", 0.7: "0.7", 0.99: "0.99"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"width": "220px", "paddingTop": "4px"})),
        ),
        html.Hr(),
        html.Div(id="ses-output"),
        dcc.Download(id="ses-download"),
    ])


# ── Holt tab layout ────────────────────────────────────────────────────────────
def holt_tab():
    return html.Div([
        html.H3("Holt's Linear Trend", className="section-head"),
        settings_row(
            setting("Forecast Periods", dcc.Input(id="holt-periods", type="number", value=12, min=1, max=36, step=1)),
        ),
        html.Hr(),
        html.Div(id="holt-output"),
        dcc.Download(id="holt-download"),
    ])


# ── Holt-Winters tab layout ────────────────────────────────────────────────────
def hw_tab():
    return html.Div([
        html.H3("Holt-Winters (Triple Exponential Smoothing)", className="section-head"),
        settings_row(
            setting("Forecast Periods",  dcc.Input(id="hw-periods", type="number", value=12,  min=1,  max=36, step=1)),
            setting("Seasonal Periods",  dcc.Input(id="hw-sp",      type="number", value=12,  min=4,  max=52, step=1)),
            setting("Trend",             dcc.Dropdown(id="hw-trend",    options=["add","mul","None"], value="add", clearable=False, style={"width": "100px"})),
            setting("Seasonal",          dcc.Dropdown(id="hw-seasonal", options=["add","mul"],        value="add", clearable=False, style={"width": "100px"})),
        ),
        html.Hr(),
        html.Div(id="hw-output"),
        dcc.Download(id="hw-download"),
    ])


# ── SARIMA tab layout ──────────────────────────────────────────────────────────
def sarima_tab():
    def num(id_, val):
        return dcc.Input(id=id_, type="number", value=val, min=0, max=5, step=1)
    return html.Div([
        html.H3("SARIMA", className="section-head"),
        settings_row(
            setting("Forecast",  dcc.Input(id="sar-periods", type="number", value=12, min=1, max=36, step=1)),
            setting("Season (s)", dcc.Input(id="sar-sp",      type="number", value=12, min=4, max=52, step=1)),
            setting("p",  num("sar-p", 1)), setting("d",  num("sar-d", 1)), setting("q",  num("sar-q", 1)),
            setting("P",  num("sar-P", 1)), setting("D",  num("sar-D", 1)), setting("Q",  num("sar-Q", 1)),
        ),
        html.Div(id="sarima-order-label", style={"color": "#94a3b8", "fontSize": "0.85rem", "marginBottom": "10px"}),
        html.Hr(),
        html.Button("🚀 Run SARIMA", id="btn-sarima", className="btn-primary"),
        dcc.Loading(
            html.Div(id="sarima-output", style={"marginTop": "18px"}),
            type="dot", color="#6366f1"
        ),
        dcc.Download(id="sarima-download"),
    ])


# ── Auto ARIMA tab layout ──────────────────────────────────────────────────────
def auto_arima_tab():
    return html.Div([
        html.H3("Auto ARIMA — Automatic Order Selection", className="section-head"),
        settings_row(
            setting("Forecast Periods",      dcc.Input(id="aa-periods", type="number", value=12, min=1, max=36, step=1)),
            setting("Seasonal Period (m)",   dcc.Input(id="aa-m",       type="number", value=12, min=1, max=52, step=1)),
            setting("Information Criterion", dcc.Dropdown(id="aa-criterion",
                                                          options=["aic","bic","aicc","oob"],
                                                          value="aic", clearable=False,
                                                          style={"width": "110px"})),
            setting("Options", html.Div([
                dcc.Checklist(id="aa-options",
                              options=[{"label": " Include Seasonal", "value": "seasonal"},
                                       {"label": " Stepwise Search",  "value": "stepwise"}],
                              value=["seasonal","stepwise"],
                              className="checkbox-group",
                              labelStyle={"display": "block", "marginBottom": "4px"}),
            ])),
        ),
        html.Hr(),
        html.Button("🚀 Run Auto ARIMA", id="btn-auto-arima", className="btn-primary"),
        dcc.Loading(
            html.Div(id="auto-arima-output", style={"marginTop": "18px"}),
            type="dot", color="#6366f1"
        ),
        dcc.Download(id="auto-arima-download"),
    ])


# ── Decomposition tab layout ───────────────────────────────────────────────────
def decomp_tab(series):
    return html.Div([
        html.H3("Seasonal Decomposition", className="section-head"),
        settings_row(
            setting("Seasonal Periods", dcc.Input(id="decomp-sp",    type="number", value=12, min=4, max=52, step=1)),
            setting("Model",            dcc.Dropdown(id="decomp-model",
                                                     options=["additive","multiplicative"],
                                                     value="additive", clearable=False,
                                                     style={"width": "150px"})),
        ),
        html.Hr(),
        html.Div(id="decomp-output"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def _get_series_from_state(brand, sub_brand, freq, data):
    df = pd.read_json(io.StringIO(data), orient="split")
    return get_series(df, brand, sub_brand, freq)


# ── Moving Average ─────────────────────────────────────────────────────────────
@app.callback(
    Output("ma-output", "children"),
    Input("ma-periods", "value"),
    Input("ma-window",  "value"),
    Input("brand-dd",    "value"),
    Input("subbrand-dd", "value"),
    Input("freq-dd",     "value"),
    State("store-df",    "data"),
)
def run_ma(periods, window, brand, sub_brand, freq, data):
    if not all([data, brand, sub_brand, freq, periods, window]):
        return html.Div("Configure settings above.", className="alert-info")
    try:
        series = _get_series_from_state(brand, sub_brand, freq, data)
        result = fc.moving_average(series, periods=int(periods), window=int(window))
        return forecast_div(series, result, "Moving Average", periods, "btn-ma-dl")
    except Exception as e:
        return html.Div(f"❌ Error: {e}", className="alert-error")


@app.callback(
    Output("ma-download", "data"),
    Input("btn-ma-dl", "n_clicks"),
    State("ma-periods",  "value"),
    State("ma-window",   "value"),
    State("brand-dd",    "value"),
    State("subbrand-dd", "value"),
    State("freq-dd",     "value"),
    State("store-df",    "data"),
    prevent_initial_call=True,
)
def dl_ma(n, periods, window, brand, sub_brand, freq, data):
    if not n:
        raise PreventUpdate
    series = _get_series_from_state(brand, sub_brand, freq, data)
    result = fc.moving_average(series, periods=int(periods), window=int(window))
    fc_df = pd.DataFrame({"Forecast": result["forecast"], "Lower 95%": result["lower"], "Upper 95%": result["upper"]})
    return dcc.send_bytes(lambda _=None: to_excel_bytes(fc_df), "ma_forecast.xlsx")


# ── SES ────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("ses-output", "children"),
    Input("ses-periods", "value"),
    Input("ses-alpha",   "value"),
    Input("brand-dd",    "value"),
    Input("subbrand-dd", "value"),
    Input("freq-dd",     "value"),
    State("store-df",    "data"),
)
def run_ses(periods, alpha, brand, sub_brand, freq, data):
    if not all([data, brand, sub_brand, freq, periods]):
        return html.Div("Configure settings above.", className="alert-info")
    try:
        series = _get_series_from_state(brand, sub_brand, freq, data)
        result = fc.ses_forecast(series, periods=int(periods), alpha=float(alpha or 0.3))
        return forecast_div(series, result, "SES", periods, "btn-ses-dl")
    except Exception as e:
        return html.Div(f"❌ Error: {e}", className="alert-error")


@app.callback(
    Output("ses-download", "data"),
    Input("btn-ses-dl",  "n_clicks"),
    State("ses-periods", "value"),
    State("ses-alpha",   "value"),
    State("brand-dd",    "value"),
    State("subbrand-dd", "value"),
    State("freq-dd",     "value"),
    State("store-df",    "data"),
    prevent_initial_call=True,
)
def dl_ses(n, periods, alpha, brand, sub_brand, freq, data):
    if not n:
        raise PreventUpdate
    series = _get_series_from_state(brand, sub_brand, freq, data)
    result = fc.ses_forecast(series, periods=int(periods), alpha=float(alpha or 0.3))
    fc_df = pd.DataFrame({"Forecast": result["forecast"], "Lower 95%": result["lower"], "Upper 95%": result["upper"]})
    return dcc.send_bytes(lambda _=None: to_excel_bytes(fc_df), "ses_forecast.xlsx")


# ── Holt ───────────────────────────────────────────────────────────────────────
@app.callback(
    Output("holt-output", "children"),
    Input("holt-periods", "value"),
    Input("brand-dd",     "value"),
    Input("subbrand-dd",  "value"),
    Input("freq-dd",      "value"),
    State("store-df",     "data"),
)
def run_holt(periods, brand, sub_brand, freq, data):
    if not all([data, brand, sub_brand, freq, periods]):
        return html.Div("Configure settings above.", className="alert-info")
    try:
        series = _get_series_from_state(brand, sub_brand, freq, data)
        result = fc.holt_forecast(series, periods=int(periods))
        return forecast_div(series, result, "Holt's Linear", periods, "btn-holt-dl")
    except Exception as e:
        return html.Div(f"❌ Error: {e}", className="alert-error")


@app.callback(
    Output("holt-download", "data"),
    Input("btn-holt-dl",  "n_clicks"),
    State("holt-periods", "value"),
    State("brand-dd",     "value"),
    State("subbrand-dd",  "value"),
    State("freq-dd",      "value"),
    State("store-df",     "data"),
    prevent_initial_call=True,
)
def dl_holt(n, periods, brand, sub_brand, freq, data):
    if not n:
        raise PreventUpdate
    series = _get_series_from_state(brand, sub_brand, freq, data)
    result = fc.holt_forecast(series, periods=int(periods))
    fc_df = pd.DataFrame({"Forecast": result["forecast"], "Lower 95%": result["lower"], "Upper 95%": result["upper"]})
    return dcc.send_bytes(lambda _=None: to_excel_bytes(fc_df), "holt_forecast.xlsx")


# ── Holt-Winters ───────────────────────────────────────────────────────────────
@app.callback(
    Output("hw-output", "children"),
    Input("hw-periods",  "value"),
    Input("hw-sp",       "value"),
    Input("hw-trend",    "value"),
    Input("hw-seasonal", "value"),
    Input("brand-dd",    "value"),
    Input("subbrand-dd", "value"),
    Input("freq-dd",     "value"),
    State("store-df",    "data"),
)
def run_hw(periods, sp, trend, seasonal, brand, sub_brand, freq, data):
    if not all([data, brand, sub_brand, freq, periods, sp]):
        return html.Div("Configure settings above.", className="alert-info")
    try:
        series = _get_series_from_state(brand, sub_brand, freq, data)
        t = None if trend == "None" else trend
        result = fc.holtwinters_forecast(series, periods=int(periods),
                                         seasonal_periods=int(sp),
                                         trend=t, seasonal=seasonal)
        return forecast_div(series, result, "Holt-Winters", periods, "btn-hw-dl")
    except Exception as e:
        return html.Div(f"❌ Error: {e}", className="alert-error")


@app.callback(
    Output("hw-download", "data"),
    Input("btn-hw-dl",   "n_clicks"),
    State("hw-periods",  "value"),
    State("hw-sp",       "value"),
    State("hw-trend",    "value"),
    State("hw-seasonal", "value"),
    State("brand-dd",    "value"),
    State("subbrand-dd", "value"),
    State("freq-dd",     "value"),
    State("store-df",    "data"),
    prevent_initial_call=True,
)
def dl_hw(n, periods, sp, trend, seasonal, brand, sub_brand, freq, data):
    if not n:
        raise PreventUpdate
    series = _get_series_from_state(brand, sub_brand, freq, data)
    t = None if trend == "None" else trend
    result = fc.holtwinters_forecast(series, periods=int(periods), seasonal_periods=int(sp),
                                     trend=t, seasonal=seasonal)
    fc_df = pd.DataFrame({"Forecast": result["forecast"], "Lower 95%": result["lower"], "Upper 95%": result["upper"]})
    return dcc.send_bytes(lambda _=None: to_excel_bytes(fc_df), "hw_forecast.xlsx")


# ── SARIMA ─────────────────────────────────────────────────────────────────────
@app.callback(
    Output("sarima-order-label", "children"),
    Input("sar-p", "value"), Input("sar-d", "value"), Input("sar-q", "value"),
    Input("sar-P", "value"), Input("sar-D", "value"), Input("sar-Q", "value"),
    Input("sar-sp", "value"),
)
def update_sarima_label(p, d, q, P, D, Q, sp):
    return f"SARIMA ({p},{d},{q})×({P},{D},{Q},{sp})"


@app.callback(
    Output("sarima-output", "children"),
    Output("btn-sarima", "children", allow_duplicate=True),
    Output("btn-sarima", "disabled", allow_duplicate=True),
    Input("btn-sarima",  "n_clicks"),
    State("sar-periods", "value"),
    State("sar-sp",      "value"),
    State("sar-p",       "value"), State("sar-d", "value"), State("sar-q", "value"),
    State("sar-P",       "value"), State("sar-D", "value"), State("sar-Q", "value"),
    State("brand-dd",    "value"),
    State("subbrand-dd", "value"),
    State("freq-dd",     "value"),
    State("store-df",    "data"),
    prevent_initial_call=True,
)
def run_sarima(n, periods, sp, p, d, q, P, D, Q, brand, sub_brand, freq, data):
    reset_args = ("🚀 Run SARIMA", False)
    if not all([data, brand, sub_brand, freq]):
        err = html.Div("Load data first.", className="alert-info")
        return err, *reset_args
    try:
        series = _get_series_from_state(brand, sub_brand, freq, data)
        order = (int(p), int(d), int(q))
        seasonal_order = (int(P), int(D), int(Q), int(sp))
        label = f"SARIMA {order}×{seasonal_order}"
        result = fc.sarima_forecast(series, periods=int(periods),
                                    order=order, seasonal_order=seasonal_order)
        children = [forecast_div(series, result, label, periods, "btn-sarima-dl")]
        if "summary" in result:
            children.append(html.Details([
                html.Summary("📄 Model Summary", style={"cursor": "pointer", "color": "#94a3b8"}),
                html.Pre(result["summary"], style={"backgroundColor": "#111827", "padding": "12px",
                                                   "borderRadius": "8px", "color": "#94a3b8",
                                                   "fontSize": "0.78rem", "overflow": "auto",
                                                   "marginTop": "10px"}),
            ]))
        return html.Div(children), *reset_args
    except Exception as e:
        err = html.Div(f"❌ SARIMA Error: {e}", className="alert-error")
        return err, *reset_args


@app.callback(
    Output("sarima-download", "data"),
    Input("btn-sarima-dl", "n_clicks"),
    State("sar-periods", "value"),
    State("sar-sp",      "value"),
    State("sar-p", "value"), State("sar-d", "value"), State("sar-q", "value"),
    State("sar-P", "value"), State("sar-D", "value"), State("sar-Q", "value"),
    State("brand-dd",    "value"),
    State("subbrand-dd", "value"),
    State("freq-dd",     "value"),
    State("store-df",    "data"),
    prevent_initial_call=True,
)
def dl_sarima(n, periods, sp, p, d, q, P, D, Q, brand, sub_brand, freq, data):
    if not n:
        raise PreventUpdate
    series = _get_series_from_state(brand, sub_brand, freq, data)
    result = fc.sarima_forecast(series, periods=int(periods),
                                order=(int(p), int(d), int(q)),
                                seasonal_order=(int(P), int(D), int(Q), int(sp)))
    fc_df = pd.DataFrame({"Forecast": result["forecast"], "Lower 95%": result["lower"], "Upper 95%": result["upper"]})
    return dcc.send_bytes(lambda _=None: to_excel_bytes(fc_df), "sarima_forecast.xlsx")


# ── Auto ARIMA ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("auto-arima-output", "children"),
    Output("btn-auto-arima", "children", allow_duplicate=True),
    Output("btn-auto-arima", "disabled", allow_duplicate=True),
    Input("btn-auto-arima", "n_clicks"),
    State("aa-periods",    "value"),
    State("aa-m",          "value"),
    State("aa-criterion",  "value"),
    State("aa-options",    "value"),
    State("brand-dd",      "value"),
    State("subbrand-dd",   "value"),
    State("freq-dd",       "value"),
    State("store-df",      "data"),
    prevent_initial_call=True,
)
def run_auto_arima(n, periods, m, criterion, options, brand, sub_brand, freq, data):
    reset_args = ("🚀 Run Auto ARIMA", False)
    if not all([data, brand, sub_brand, freq]):
        err = html.Div("Load data first.", className="alert-info")
        return err, *reset_args
    try:
        series = _get_series_from_state(brand, sub_brand, freq, data)
        seasonal  = "seasonal"  in (options or [])
        stepwise  = "stepwise"  in (options or [])
        result    = fc.auto_arima_forecast(
            series, periods=int(periods), seasonal=seasonal,
            m=int(m or 12), stepwise=stepwise, information_criterion=criterion or "aic",
        )
        o, so     = result["order"], result["seasonal_order"]
        order_str = (f"ARIMA({o[0]},{o[1]},{o[2]})×({so[0]},{so[1]},{so[2]},{so[3]})"
                     if seasonal else f"ARIMA({o[0]},{o[1]},{o[2]})")
        label     = f"Auto ARIMA {order_str}"
        children  = [
            html.Div(f"✅ Best order: {order_str}", className="alert-success"),
            forecast_div(series, result, label, periods, "btn-aa-dl"),
        ]
        if "summary" in result:
            children.append(html.Details([
                html.Summary("📄 Auto ARIMA Summary",
                             style={"cursor": "pointer", "color": "#94a3b8"}),
                html.Pre(result["summary"], style={"backgroundColor": "#111827", "padding": "12px",
                                                   "borderRadius": "8px", "color": "#94a3b8",
                                                   "fontSize": "0.78rem", "overflow": "auto",
                                                   "marginTop": "10px"}),
            ]))
        return html.Div(children), *reset_args
    except Exception as e:
        err = html.Div(f"❌ Auto ARIMA Error: {e}", className="alert-error")
        return err, *reset_args


@app.callback(
    Output("auto-arima-download", "data"),
    Input("btn-aa-dl",     "n_clicks"),
    State("aa-periods",    "value"),
    State("aa-m",          "value"),
    State("aa-criterion",  "value"),
    State("aa-options",    "value"),
    State("brand-dd",      "value"),
    State("subbrand-dd",   "value"),
    State("freq-dd",       "value"),
    State("store-df",      "data"),
    prevent_initial_call=True,
)
def dl_auto_arima(n, periods, m, criterion, options, brand, sub_brand, freq, data):
    if not n:
        raise PreventUpdate
    series   = _get_series_from_state(brand, sub_brand, freq, data)
    seasonal = "seasonal" in (options or [])
    stepwise = "stepwise" in (options or [])
    result   = fc.auto_arima_forecast(series, periods=int(periods), seasonal=seasonal,
                                       m=int(m or 12), stepwise=stepwise,
                                       information_criterion=criterion or "aic")
    fc_df = pd.DataFrame({"Forecast": result["forecast"], "Lower 95%": result["lower"], "Upper 95%": result["upper"]})
    return dcc.send_bytes(lambda _=None: to_excel_bytes(fc_df), "auto_arima_forecast.xlsx")


# ── Decomposition ──────────────────────────────────────────────────────────────
@app.callback(
    Output("decomp-output", "children"),
    Input("decomp-sp",    "value"),
    Input("decomp-model", "value"),
    Input("brand-dd",     "value"),
    Input("subbrand-dd",  "value"),
    Input("freq-dd",      "value"),
    State("store-df",     "data"),
)
def run_decomp(sp, model, brand, sub_brand, freq, data):
    if not all([data, brand, sub_brand, freq, sp]):
        return html.Div("Configure settings above.", className="alert-info")
    try:
        series = _get_series_from_state(brand, sub_brand, freq, data)
        clean  = series.dropna()
        if len(clean) < int(sp) * 2:
            return html.Div(f"⚠️ Need at least {int(sp)*2} observations.", className="alert-warning")
        decomp = fc.decompose_series(clean, model=model, period=int(sp))
        fig = make_subplots(rows=4, cols=1,
                            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
                            shared_xaxes=True, vertical_spacing=0.07)
        for s_, color, row in [
            (decomp.observed, COLORS["actual"],   1),
            (decomp.trend,    COLORS["trend"],    2),
            (decomp.seasonal, COLORS["seasonal"], 3),
            (decomp.resid,    COLORS["resid"],    4),
        ]:
            fig.add_trace(go.Scatter(x=s_.index, y=s_.values, mode="lines",
                                     line=dict(color=color, width=1.8), showlegend=False),
                          row=row, col=1)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.03)",
            height=680, margin=dict(t=30, b=30, l=50, r=20),
            font=dict(family="Inter, sans-serif", color="#e8eaf6"),
        )
        for ax in fig.layout:
            if "xaxis" in ax or "yaxis" in ax:
                fig.layout[ax].update(gridcolor="rgba(255,255,255,0.08)", color="#a5b4fc")
        for ann in fig.layout.annotations:
            ann.font.color = "#a5b4fc"

        strength_t = max(0.0, 1 - decomp.resid.var() /
                         (decomp.trend.dropna() + decomp.resid.dropna()).var())
        strength_s = max(0.0, 1 - decomp.resid.var() /
                         (decomp.seasonal + decomp.resid.dropna()).var())

        return html.Div([
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.Div([
                html.Div([html.H3(["TREND STRENGTH ", html.Span("ℹ️", title=METRIC_HINTS["TREND STRENGTH"], style={"cursor": "help", "opacity": 0.6})]),    html.H2(f"{strength_t:.3f}")], className="metric-card"),
                html.Div([html.H3(["SEASONAL STRENGTH ", html.Span("ℹ️", title=METRIC_HINTS["SEASONAL STRENGTH"], style={"cursor": "help", "opacity": 0.6})]), html.H2(f"{strength_s:.3f}")], className="metric-card"),
                html.Div([html.H3(["RESIDUAL STD ", html.Span("ℹ️", title=METRIC_HINTS["RESIDUAL STD"], style={"cursor": "help", "opacity": 0.6})]),      html.H2(f"{decomp.resid.std():.3f}")], className="metric-card"),
            ], className="metrics-grid", style={"marginTop": "12px"}),
        ])
    except Exception as e:
        return html.Div(f"❌ Decomposition error: {e}", className="alert-error")


# ── Loading UI (Client-Side) ───────────────────────────────────────────────────

app.clientside_callback(
    """function(n) { if(n) return ["⏳ Running SARIMA...", true]; return window.dash_clientside.no_update; }""",
    Output("btn-sarima", "children"),Output("btn-sarima", "disabled"),
    Input("btn-sarima", "n_clicks"), prevent_initial_call=True
)

app.clientside_callback(
    """function(n) { if(n) return ["⏳ Running Auto ARIMA...", true]; return window.dash_clientside.no_update; }""",
    Output("btn-auto-arima", "children"),Output("btn-auto-arima", "disabled"),
    Input("btn-auto-arima", "n_clicks"), prevent_initial_call=True
)


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

app.layout = html.Div(id="app-shell", children=[sidebar(), main_content()])

if __name__ == "__main__":
    print("\n   🚀 Starting Forecasting App (Dash)...")
    print("   🌐 Open your browser: http://localhost:8050\n")
    app.run(debug=False, port=8050, host="0.0.0.0")
