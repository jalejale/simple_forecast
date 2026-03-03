# 📈 Simple Forecast

A browser-based time series forecasting application built with Python and Plotly Dash. This app provides an intuitive, professional dark-mode interface to analyze temporal data and generate forecasts using six different statistical and machine learning models.

![App Screenshot](https://raw.githubusercontent.com/jalejale/simple_forecast/main/.github/preview.png) *(Preview image placeholder)*

## ✨ Features

- **Upload Data**: Easily upload your own datasets via CSV or Excel.
- **Data Template**: Download a standardized CSV template directly from the app.
- **Hierarchical Filtering**: Filter your data by Brand, and then seamlessly by Sub-Brand.
- **Native Tabs Navigation**: Clean, accessible tab-based UI for switching between reports.
- **6 Forecasting Models**:
  - 📉 Moving Average
  - 🔵 Simple Exponential Smoothing (SES)
  - 📐 Holt's Linear Trend
  - ❄️ Holt-Winters (Triple Exponential Smoothing)
  - 🤖 SARIMA
  - 🔮 Auto ARIMA (Automatic Order Selection)
- **Seasonal Decomposition**: Visualize Observed, Trend, Seasonal, and Residual components.
- **Interactive Visualizations**: High-quality, interactive Plotly charts with 95% confidence intervals.
- **Accuracy Metrics**: Automatically calculates MAE, RMSE, and MAPE for each model.
- **Export to Excel**: Download the exact forecast data (including confidence intervals) directly to an Excel file.

## 🚀 Quick Start

### Prerequisites

Ensure you have Python 3.9+ installed on your system.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jalejale/simple_forecast.git
   cd simple_forecast
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

Start the Plotly Dash local server:

```bash
python app.py
```

The application will automatically initialize and you can view it in your default web browser at `http://localhost:8050`.

## 📁 Project Structure

- `app.py`: The main Plotly Dash frontend application, defining the UI layout, styling, and callback reactivity.
- `forecasting.py`: The backend engine containing all the time series modeling functions (statsmodels, pmdarima).
- `requirements.txt`: The list of Python package dependencies.
- `sample_data.csv`: A sample dataset to test the application immediately without needing your own data.

## 📊 Data Format Requirement

If you upload your own data, ensure it contains the following columns exactly as named:
- `date`: The time period (e.g., YYYY-MM-DD).
- `brand`: The primary category.
- `sub_brand`: The secondary sub-category.
- `qty`: The numerical quantity/value to forecast.

## 🛠️ Built With

- [Plotly Dash](https://dash.plotly.com/)
- [Pandas](https://pandas.pydata.org/)
- [Statsmodels](https://www.statsmodels.org/)
- [Pmdarima (Auto ARIMA)](https://alkaline-ml.com/pmdarima/)
- [Plotly](https://plotly.com/python/)
