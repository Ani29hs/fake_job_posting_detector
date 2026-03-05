# Fake Job Posting Detector

## Project Overview

This project uses Machine Learning and Natural Language Processing (NLP) to detect fraudulent job postings. Many online job platforms contain scam listings that try to exploit job seekers. This system analyzes the text of a job posting and predicts whether the job is legitimate or fraudulent.

The model is trained on a dataset of job listings and learns patterns that commonly appear in fake job advertisements.

---

## Objective

To build a machine learning model that can classify job postings as:

* **Legitimate Job (0)**
* **Fraudulent Job (1)**

This helps identify scam job listings and protect job seekers.

---

## Machine Learning Approach

### 1️⃣ Data Preprocessing

* Handled missing values
* Selected relevant text features (title, description, requirements, benefits, etc.)
* Combined multiple text fields into one input feature

### 2️⃣ Text Vectorization

Used **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert job descriptions into numerical vectors while removing common stopwords.

### 3️⃣ Model Training

A **Logistic Regression classifier** was trained to learn patterns between words and fraudulent job postings.

### 4️⃣ Handling Class Imbalance

Since the dataset contains significantly more real jobs than fake ones, **class_weight='balanced'** was used to improve fraud detection.

---

## Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

These metrics ensure the model correctly identifies fraudulent job postings.

---

## Web Application

A **Streamlit web application** was developed where users can:

1. Enter a job description
2. Click **Predict**
3. Receive a prediction:

   * Legitimate Job Posting
   * Fake Job Posting

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* Streamlit
* Matplotlib & Seaborn

---

## Project Structure

```
Fake_Job_Predictor
│
├── data
│   └── fake_job_postings.csv
│
├── model
│   ├── fake_job_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebook
│   └── fake_job_detector.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

## How to Run the Project

1. Clone the repository

```
git clone https://github.com/yourusername/fake-job-posting-detector.git
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the Streamlit app

```
streamlit run app.py
```

---

## Future Improvements

* Use advanced NLP models (BERT or transformers)
* Add word cloud visualization for fake vs real jobs
* Deploy the application online
* Improve performance using ensemble models

---

## Author

Aniket Sharma
B.Tech CSE (AI & ML)
Chitkara University
