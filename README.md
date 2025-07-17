# Career Path Predictor AI

This project is a complete end-to-end machine learning application designed to predict a suitable career path based on a user's academic performance, skills, and personal interests. The final output is an interactive web application built with Streamlit that provides a personalized career recommendation and a link to a free, relevant online course to help the user get started.

## Project Overview

The core of this project is a classification model that has been trained on a diverse dataset of student and professional profiles. The process involved several key stages:

1.  **Data Unification:** Merging multiple, disparate datasets into a single, cohesive analytical dataset.
2.  **Robust Preprocessing:** Creating a sophisticated pipeline to handle missing data, encode categorical features, scale numerical values, and address significant class imbalance.
3.  **Advanced Model Training:** Utilizing the powerful `XGBoost` library to train a gradient boosting model, configured to leverage GPU acceleration for speed.
4.  **Hyperparameter Tuning:** Employing `RandomizedSearchCV` to automatically find the optimal settings for the model, significantly improving its predictive accuracy.
5.  **Interactive Application:** Building a user-friendly web interface with Streamlit that allows anyone to get a career prediction by simply answering a series of questions.

---

## ⚙️ The Technical Workflow

### 1. Data Collection and Preprocessing

The foundation of the model was built by merging three distinct datasets: `dataset9000.csv`, `career_pred.csv`, and `cs_students.csv`.

- **Standardization:** A crucial first step was to standardize the feature names (skills) and target values (job roles) across all files. For example, roles like "Database Developer" and "Database Administrator" were unified under a single `Database_Professional` category.
- **Feature Engineering:** Categorical data (e.g., skill levels like 'Average', 'Excellent') and binary values ('yes'/'no') were converted into numerical formats.
- **Alignment & Merging:** The processed datasets were aligned to a common schema, ensuring all had the same columns. Missing skill values were imputed with `0`, representing a lack of that skill. Finally, the dataframes were concatenated into a single `merged_analytical_data.csv` file.

### 2. Model Training and Tuning

A `scikit-learn` pipeline was constructed to ensure that data was processed consistently and to prevent data leakage.

-   **Preprocessing Pipeline (`preprocessor.pkl`):**
    -   **Numerical Features:** Missing values were imputed using the `median`, and then all features were scaled using `StandardScaler`.
    -   **Categorical Features:** Missing values were imputed using the `most_frequent` value, and then features were converted into a numerical format using `OneHotEncoder`.
-   **Handling Class Imbalance:** The dataset contained a highly imbalanced distribution of job roles. To prevent the model from becoming biased towards the most common careers, the **SMOTE** (Synthetic Minority Over-sampling Technique) was applied *only to the training data*. This created synthetic examples of the under-represented job roles, resulting in a balanced training set.
-   **XGBoost Model (`xgboost_career_model.pkl`):**
    -   An `XGBClassifier` was chosen for its high performance on tabular data.
    -   The model was configured to train on **NVIDIA GPUs (`device='cuda'`)**, dramatically speeding up the training and tuning process.
    -   **Hyperparameter Tuning:** `RandomizedSearchCV` was used to automatically test 25 different combinations of hyperparameters (like `learning_rate`, `max_depth`, `n_estimators`, etc.) to find the configuration that yielded the highest accuracy. The best model from this search was saved.
-   **Label Encoding (`label_encoder.pkl`):** The text-based job roles were converted to numerical IDs for the model using `LabelEncoder`. This encoder was also saved to translate the model's numerical output back into a human-readable career path.

---

## 🖥️ The Streamlit Application

The final step was to create an interactive web application (`app.py`) where users can interact with the trained model.

**Features:**
- **User-Friendly Form:** The app presents a clean, multi-column form that asks the user to input their skills, academic performance, and work style preferences using intuitive sliders and radio buttons.
- **Real-Time Prediction:** Upon submission, the user's inputs are collected into a pandas DataFrame. This data is then passed through the saved `preprocessor` pipeline to transform it into the exact format the model expects.
- **Instant Results:** The trained XGBoost model predicts the most likely career path. The `label_encoder` translates the numerical prediction back into a job title, which is displayed to the user.
- **Actionable Next Steps:** Alongside the predicted career, the app provides a direct link to a free, high-quality online course to help the user immediately start learning the necessary skills for their recommended path.

### How to Run the Project Locally

To run this application on your own machine, follow these steps:

1.  **Prerequisites:**
    -   Python 3.8+
    -   pip (Python package installer)

2.  **Download Project Files:**
    -   `app.py` (The Streamlit application script)
    -   `xgboost_career_model.pkl` (The trained model)
    -   `preprocessor.pkl` (The preprocessing pipeline)
    -   `label_encoder.pkl` (The label encoder)
    -   `requirements.txt` (A file listing dependencies)

3.  **Create `requirements.txt`:**
    Create a file named `requirements.txt` and add the following lines:
    ```
    streamlit
    pandas
    numpy
    joblib
    xgboost
    scikit-learn
    imbalanced-learn
    ```

4.  **Set Up a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

5.  **Install Dependencies:**
    Make sure all the project files and `requirements.txt` are in the same directory, then run:
    ```bash
    pip install -r requirements.txt
    ```

6.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

Your web browser should automatically open a new tab with the Career Path Predictor application running.

