<p align="center"> <img src="banner_sentiment.png" alt="Sentiment Analysis Banner" width="600"/> </p>
💬 Indonesian Sentiment Text Classification with Machine Learning
This project is an end-to-end machine learning implementation to classify sentiment from Indonesian-language text reviews. It aims to build and evaluate models that can accurately predict whether a review contains positive, negative, or neutral sentiment.

🧰 Features
✅ Text Preprocessing (noise removal, case folding, stopwords removal)

🔠 Feature Extraction using TF-IDF

🧠 Model Training with:

Logistic Regression

LinearSVC (Support Vector Machine)

📊 Model Evaluation with:

Classification Report (precision, recall, F1-score)

Confusion Matrix

📚 Educational ML Pipeline including:

Data Cleaning & Normalization

Vectorization Techniques

Supervised Learning for NLP

Model Evaluation & Visualization

🗂️ Project Structure
File / Folder	Description
sentiment_analysis.ipynb	Main Jupyter Notebook containing the entire ML workflow
dataset.csv	Raw dataset of Indonesian text reviews
cleaning.py	Script for text preprocessing
model_utils.py	Utility functions for training & evaluating models
requirements.txt	List of required Python packages

🛠️ Tech Stack
Python

Scikit-learn – ML models, TF-IDF, evaluation

Pandas – Data loading & manipulation

NLTK – Natural Language Processing (stopwords, etc.)

Matplotlib & Seaborn – Data visualization

Jupyter Notebook – Interactive development environment

🚀 How to Run
Clone the repository

bash
Copy
Edit
git clone https://github.com/akmalfalahdarmawan/nama-repo-proyek-ini.git
Navigate into the project folder

bash
Copy
Edit
cd nama-repo-proyek-ini
Install all dependencies

bash
Copy
Edit
pip install -r requirements.txt
Launch Jupyter Notebook

bash
Copy
Edit
jupyter notebook
Open and run the sentiment_analysis.ipynb notebook.

