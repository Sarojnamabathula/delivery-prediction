# DelivAI — Intelligent Delivery Analytics Platform 🚚

DelivAI is an end-to-end, real-time AI-powered logistics tracking and route optimization platform. Designed to provide actionable insights for delivery management, it combines advanced machine learning models with an interactive web dashboard to predict delivery times, highlight performance metrics, and visualize complex delivery networks.

## ✨ Key Features
- **🔮 Prediction Engine**: Predict delivery durations using various inputs (distance, weather, traffic, vehicle type, and package weight). Leverages pre-trained XGBoost and Ensemble Machine Learning models to provide confidence intervals and categorize delivery types.
- **📊 Analytics & Insights**: Explore historical data via interactive multidimensional charts (Plotly). Features robust filtering by date, distance constraints, and vehicle types, allowing deep dives into metrics like *Duration by Vehicle*, *Time vs Workday Congestion*, and *Impact of Weather*.
- **🤖 Model Performance**: Understand the underlying AI metrics. See live model comparisons (R², MAE, RMSE) and explore feature importance using native plots and SHAP analysis.
- **🗺️ Route Map**: An interactive Folium-based geospatial map that renders logistical networks, mapping the fastest and slowest corridors between Origins and Destinations with color-coded pathing based on speed.

## 🛠️ Technology Stack
- **Dashboard & UI**: Streamlit
- **Machine Learning**: Scikit-Learn, XGBoost
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Plotly, Folium, Seaborn, Matplotlib
- **Model Explainability**: SHAP
- **API Services**: FastAPI, Uvicorn (For automated pipeline integration and external testing)

## 📁 Repository Structure
```
delivery_prediction/
├── app/
│   ├── app.py             # Main Streamlit Dashboard Application
│   └── utils.py           # Helper classes and constant mappings (like CITY_COORDS)
├── data/                  # Contains raw and processed logistical CSV datasets
├── models/                # Serialized model (.pkl) artifacts (XGB, RF, Ensemble)
├── notebooks/             # Jupyter notebooks for initial EDA and experimental modeling
├── outputs/               # Saved charts and static analytical reports
├── tests/                 # Pytest test cases for validating API and prediction logic
├── requirements.txt       # Project dependencies
└── README.md
```

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sarojnamabathula/delivery-prediction.git
   cd delivery-prediction
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Dashboard:**
   ```bash
   cd app
   streamlit run app.py
   ```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to check the issues page and open a PR if you want to improve model accuracy or add new dashboards.

## 📝 License
This project is open-source and available under the MIT License.
