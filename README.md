#  Teenager Mental Health Predictor App

An interactive Machine Learning web application that analyzes teenager lifestyle metrics (sleep, screen time, physical activity, etc.) to predict the risk of depression. Built entirely in Python using Scikit-Learn and Streamlit.

## Features

* **Real-Time Predictions:** Users input their daily habits via a sidebar, and the model instantly calculates their mental health risk.
* **Probability Confidence:** Doesn't just give a binary "Yes/No" answer. The app outputs the exact probability percentage (e.g., "The model is 82% confident").
* **Explainable AI (XAI):** Features a dynamic bar chart that extracts the model's mathematical coefficients in real-time, showing users exactly *which* habits are protecting them and which are dragging their score down.
* 
* **Class Imbalance Handling:** The underlying Logistic Regression model was trained using balanced class weights to ensure accurate recall for minority positive cases.

##  Tech Stack

* **Frontend UI:** Streamlit
* **Machine Learning:** Scikit-Learn (Logistic Regression, Label Encoding)
* **Data Manipulation:** Pandas, NumPy
* **Model Serialization:** Joblib

## 💻 How to Run Locally

Want to test the app on your own machine? Follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
   ```

2. **Install the required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

* `app.py`: The main Streamlit web application code.
* `teenager_mental_health_model.joblib`: The trained Logistic Regression model.
* `gender.joblib`, `platform.joblib`, `social.joblib`: The trained LabelEncoders for categorical data.
* `requirements.txt`: List of Python dependencies.

## ⚠️ Medical Disclaimer

**This application is for educational and demonstrational purposes only.** The predictions made by this Machine Learning model are based on a limited dataset and should *never* be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or qualified mental health provider with any questions you may have regarding a medical condition.
```


