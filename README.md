# 🌧️ Rainfall Prediction System (Machine Learning + Streamlit)

This project is a simple **Rainfall Prediction Web App** built using:

- Python
- Pandas
- Scikit-Learn (Linear Regression)
- Streamlit

The model predicts expected rainfall (in mm) based on inputs such as:

✔ State  
✔ Month  
✔ Temperature  
✔ Humidity  
✔ Wind Speed  

---

## 📂 Project Structure

Rainfall-Prediction/
│
├── app.py
├── requirements.txt (recommended)
└── README.md

Copy code

> Note: A virtual environment (e.g., `rain/`) is **NOT uploaded** to GitHub.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

git clone https://github.com/Harshini479890/Rainfall-Prediction.git

cd Rainfall-Prediction

2️⃣ (Optional but Recommended) Create Virtual Environment

Windows

python -m venv venv

venv\Scripts\activate

3️⃣ Install Dependencies

If you have requirements.txt:


pip install -r requirements.txt
Or manually:

pip install streamlit pandas scikit-learn

4️⃣ Run the App

streamlit run app.py

You will see something like:

Local URL: http://localhost:8501

Open it in your browser 🎉

🧠 Model Details

Algorithm: Linear Regression

Synthetic climate data generated for:

Maharashtra

Kerala

Tamil Nadu

Rajasthan

Punjab

Monsoon months weighted higher 🌧️

✨ Features

✔ Interactive UI

✔ Real-time rainfall prediction

✔ Synthetic dataset

✔ Simple & beginner-friendly

📌 Requirements

Python 3.8+

pip
