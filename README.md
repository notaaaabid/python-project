# 🫁 Sleep Apnea & Sleep Quality Analyser

A modular GUI-based Python application for predicting sleep apnea and
analysing sleep quality using machine learning.

---

## 📁 Project Structure

```
sleep_apnea_app/
│
├── run_app.py          ← Entry point — run this!
│
├── data_loader.py      ← Module 1: Dataset generation / loading
├── preprocessor.py     ← Module 2: Cleaning & feature engineering
├── model_trainer.py    ← Module 3: ML training & evaluation
├── analytics.py        ← Module 4: All matplotlib/seaborn charts
└── gui_app.py          ← Module 5: Tkinter GUI (ties it all together)
```

---

## 🚀 How to Run

```bash
# 1. Navigate to the project folder
cd sleep_apnea_app

# 2. Install dependencies (one-time)
pip install scikit-learn pandas numpy matplotlib seaborn joblib

# 3. Launch the app
python run_app.py
```

> **Using a real Kaggle dataset?**  
> Download `Sleep_health_and_lifestyle_dataset.csv` from Kaggle and click
> **"Load Custom CSV"** inside the app.

---

## 📦 Module Breakdown

| Module | What it does | Key concepts |
|---|---|---|
| `data_loader.py` | Generates/loads the 400-record dataset | NumPy distributions, Pandas DataFrames, CSV I/O |
| `preprocessor.py` | Cleans & encodes features for ML | LabelEncoder, StandardScaler, train/test split |
| `model_trainer.py` | Trains 4 classifiers, picks the best | Random Forest, GBM, SVM, LogReg, cross-validation |
| `analytics.py` | Produces all embedded charts | matplotlib Figure API, seaborn, FigureCanvasTkAgg |
| `gui_app.py` | 4-tab Tkinter application | ttk Notebook, threading, dark theme |

---

## 🖥️ GUI Tabs

1. **🔍 Predict** — Fill in your health parameters and get an instant
   sleep disorder prediction with probability gauge and personalised recommendations.

2. **📊 Analytics** — Explore the dataset: disorder distribution, sleep
   duration histogram, correlation heatmap, stress vs quality scatter plot.

3. **🤖 Model** — Confusion matrix, feature importances, model comparison
   bar chart, and full classification report.

4. **ℹ️ About** — Description of every module so you can learn what each
   part of the codebase does.

---

## 🩺 Health Disclaimer

This tool is for **educational purposes only** and is not a substitute
for professional medical advice. Always consult a licensed physician or
sleep specialist for diagnosis and treatment.
