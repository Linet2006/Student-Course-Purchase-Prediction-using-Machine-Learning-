# Student-Course-Purchase-Prediction-using-Machine-Learning-
This project predicts whether a student will purchase an online course based on their activity, such as study hours, completed courses, and platform visits. Using machine learning models (Logistic Regression, Decision Tree, Random Forest), the model is deployed via Streamlit for real-time predictions.
# EdTech Course Purchase Prediction

## Project Description
Predicts whether a student will purchase a course based on their activity and learning behavior.

## Dataset Features
- age
- study_hours_per_week
- previous_courses_completed
- platform_visits_per_month
- assignment_completion_rate

## Model Used
Random Forest Classifier

## How to Run

### Step 1 - Clone the repository
git clone <your_repo_link>

### Step 2 - Go to the project folder
cd C:\Users\YourName\Downloads\your_repo_name

### Step 3 - Install required libraries
pip install streamlit scikit-learn pandas numpy

### Step 4 - Train and save the model
python -c "import pandas as pd; from sklearn.model_selection import train_test_split; from sklearn.ensemble import RandomForestClassifier; import pickle; df = pd.read_csv('edtech_student_course_purchase_dataset.csv'); X = df[['age','study_hours_per_week','previous_courses_completed','platform_visits_per_month','assignment_completion_rate']]; y = df['purchased_course']; X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42); rf_model = RandomForestClassifier(n_estimators=100, random_state=42); rf_model.fit(X_train, y_train); pickle.dump(rf_model, open('model.pkl', 'wb')); print('Done!')"

### Step 5 - Run the Streamlit app
python -m streamlit run app.py

## Internship
Learn Depth Internship Task
