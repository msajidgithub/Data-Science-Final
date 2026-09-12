# Customer Churn Prediction

End-to-end Machine Learning project that predicts whether a customer will churn (Yes or No), with a Streamlit web app for live predictions.

## Project contents

- `customer_churn_data.csv` — dataset
- `model_training.ipynb` — data cleaning, EDA, preprocessing, model training and evaluation
- `churn_pipeline.pkl` — saved preprocessing + final model pipeline
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies

## How the model was built

1. Cleaned data (removed duplicates and `customer_id`, filled missing values)
2. Explored churn patterns with visualizations
3. Built a Scikit-Learn Pipeline with `ColumnTransformer` (impute, scale, one-hot encode)
4. Trained Logistic Regression and Random Forest
5. Selected **Logistic Regression** based on test-set Accuracy, Precision, Recall, and F1
6. Saved the full pipeline as `churn_pipeline.pkl`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live app

Add your Streamlit Community Cloud link here after deployment.

## Author

Final Assignment — Customer Churn Prediction
