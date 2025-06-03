# 🧠 Disease Prediction Web App

A Flask-based web application that predicts diseases based on user-input symptoms using a trained Random Forest model. It includes fuzzy matching to handle symptom spelling variations.

---

## 🚀 Features

- Predicts possible diseases based on symptoms
- Uses a trained Random Forest Classifier
- FuzzyWuzzy integration for smart symptom matching
- RESTful API and web interface
- CORS enabled for frontend integration

---

## 🧰 Tech Stack

- Python
- Flask
- scikit-learn
- pandas
- FuzzyWuzzy
- HTML/CSS (via Jinja templates)

---

## 📁 Project Structure

```
Diseasepred/
├── app.py                      # Main Flask app
├── 3.py                        # Additional script (possibly training/testing)
├── random_forest_model.pkl     # Pre-trained Random Forest model
├── .venv/                      # Python virtual environment
└── templates/                  # (Optional) HTML templates for frontend
```

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Diseasepred.git
cd Diseasepred
```

### 2. Set up the virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing:
```bash
pip install flask pandas scikit-learn fuzzywuzzy python-Levenshtein
```

### 4. Run the app
```bash
python app.py
```

---

## 🧪 API Endpoint

### `POST /predict`

**Request:**
```json
{
  "symptoms": ["headache", "fever", "cough"]
}
```

**Response:**
```json
{
  "prediction": "Flu"
}
```

---

## 📌 Notes

- Ensure the dataset and model path match your environment setup.
- For full functionality, include the `Training.csv` file used during model training.

---

## 📄 License

This project is licensed under the MIT License.
