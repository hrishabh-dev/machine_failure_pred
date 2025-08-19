# 🏭 Machine Failure Prediction

Predict machine failure using a robust machine learning pipeline, delivered via a modern FastAPI web app. This project leverages advanced classification models and user-friendly UI to empower industrial operators, engineers, and data scientists to assess machine health in real time.

---

## 🚀 Features

- **Machine Learning Model:** CatBoost classifier with SMOTE oversampling, trained on real industrial sensor data.
- **Web App:** Fast, interactive FastAPI frontend with a beautiful, responsive interface.
- **Live Prediction:** Input machine parameters and instantly receive machine failure risk assessment.
- **Tested API:** Includes pytest-based tests for both homepage loading and prediction results.
- **Modern UX:** Vibrant, modern CSS and clear result messaging for intuitive usability.
- **Extensible:** Easily adaptable to new sensor types or deployment environments.

---

## 🖥️ Demo

![App Screenshot](https://github.com/hrishabh-dev/machine_failure_pred/raw/main/assets/demo.gif)

> _Example: Input sensor values, click predict, and instantly know if your machine is at risk!_

---

## 🧑‍💻 Usage

### 1. Clone the Repo

```bash
git clone https://github.com/hrishabh-dev/machine_failure_pred.git
cd machine_failure_pred
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
cd app
uvicorn main:app --reload
```
Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 📊 Model Details

- **Model:** CatBoostClassifier (with SMOTE for class imbalance)
- **Features Used:**
  - Power
  - OSF (Overstrain Failure)
  - PWF (Power Failure)
  - HDF (Heat Dissipation Failure)
  - TWF (Torque/Weight Failure)
  - Torque [Nm]
  - Rotational speed [rpm]
  - Temp_Difference
- **Performance:**  
  - _Example accuracy:_ **99.75%** on test set  
  - _Classification report:_ (see [evaluation notebook](notebooks/evaluation_of_model.ipynb) for details)

---

## 📝 Example Input

| Feature               | Example Value |
|-----------------------|--------------|
| Power                 | 66382.8      |
| OSF                   | 0            |
| PWF                   | 1            |
| HDF                   | 0            |
| TWF                   | 0            |
| Torque [Nm]           | 45.2         |
| Rotational speed [rpm]| 1433         |
| Temp_Difference       | 10.5         |

---

## 🧪 Testing

Run API tests with:

```bash
pytest tests/
```

---

## 📂 Project Structure

```
machine_failure_pred/
│
├── app/
│   ├── main.py               # FastAPI application
│   ├── models/
│   │   └── catboost_smote_pipeline.joblib
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── notebooks/
│   └── evaluation_of_model.ipynb
│
├── tests/
│   └── test_api.py
│
├── requirements.txt
└── README.md
```

---

## 🎨 UI Preview

![UI Preview](https://github.com/hrishabh-dev/machine_failure_pred/raw/main/assets/ui_preview.png)

---

## 📖 How it works

1. **Data Ingestion:** Reads industrial sensor data (see evaluation notebook for format).
2. **Model Training:** CatBoost pipeline trained with SMOTE to handle class imbalance.
3. **Web Prediction:**
    - User enters machine parameters.
    - Backend forms feature vector and runs through the model.
    - Result ("Machine Failure" or "No Failure") is displayed with colored, animated UI feedback.

---

## 🤝 Contributing

PRs welcome! Please open an issue or discussion for major changes.

---

## 📜 License

[MIT](LICENSE) © 2025 Hrishabh Kumar

---

## 🙏 Acknowledgements

- [CatBoost](https://catboost.ai/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [imblearn (SMOTE)](https://imbalanced-learn.org/)
- [Scikit-learn](https://scikit-learn.org/)
- UI inspired by modern industrial dashboards.

---
