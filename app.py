import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import re
from wordcloud import WordCloud
import io
import base64
from scipy import stats

# Configure Google Gemini API
genai.configure(api_key="enter your gemini api here")

def main():
    st.set_page_config(page_title="Advanced AI-Driven Virtual Data Scientist", layout="wide")
    st.title("Advanced AI-Driven Virtual Data Scientist")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Upload & Exploration", "Feature Analysis", "Model Building", "Advanced Analysis", "AI Insights", "Report Generation"])

    if page == "Data Upload & Exploration":
        data_upload_and_exploration()
    elif page == "Feature Analysis":
        feature_analysis_page()
    elif page == "Model Building":
        model_building_page()
    elif page == "Advanced Analysis":
        advanced_analysis_page()
    elif page == "AI Insights":
        ai_insights_page()
    elif page == "Report Generation":
        report_generation_page()

def data_upload_and_exploration():
    st.header("Data Upload & Exploration")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state['data'] = data
        
        st.subheader("Dataset Preview")
        st.write(data.head())

        st.subheader("Dataset Info")
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())

        st.subheader("Summary Statistics")
        st.write(data.describe())

        st.subheader("Missing Values")
        missing_values = data.isnull().sum()
        st.write(missing_values[missing_values > 0])

        st.subheader("Data Types")
        st.write(data.dtypes)

        st.subheader("Correlation Heatmap")
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            corr_matrix = numeric_data.corr()
            fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig)

        st.subheader("Distribution Plots")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if not numeric_columns.empty:
            selected_column = st.selectbox("Select a column for distribution plot:", numeric_columns)
            fig = px.histogram(data, x=selected_column, marginal="box")
            st.plotly_chart(fig)

        st.session_state['target_variable'] = st.selectbox("Select the target variable:", data.columns)

def feature_analysis_page():
    if 'data' not in st.session_state or 'target_variable' not in st.session_state:
        st.warning("Please upload data first.")
        return

    st.header("Feature Analysis")
    data = st.session_state['data']
    target_variable = st.session_state['target_variable']

    st.subheader("Feature Importance")
    X, y = preprocess_data(data, target_variable)
    
    if data[target_variable].dtype == 'object' or data[target_variable].nunique() < 10:
        importance = mutual_info_classif(X, y)
    else:
        importance = mutual_info_regression(X, y)
    
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    fig = px.bar(feature_importance, x='feature', y='importance', title='Feature Importance')
    st.plotly_chart(fig)

    st.subheader("Feature Correlations")
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    st.plotly_chart(fig)

    st.subheader("Pairplot")
    selected_features = st.multiselect("Select features for pairplot:", data.columns, default=feature_importance['feature'].head(3).tolist() + [target_variable])
    if len(selected_features) > 1:
        fig = px.scatter_matrix(data[selected_features])
        st.plotly_chart(fig)

def model_building_page():
    if 'data' not in st.session_state or 'target_variable' not in st.session_state:
        st.warning("Please upload data first.")
        return

    st.header("Model Building")
    data = st.session_state['data']
    target_variable = st.session_state['target_variable']

    problem_type = "classification" if data[target_variable].dtype == 'object' or data[target_variable].nunique() < 10 else "regression"
    
    st.subheader(f"Problem Type: {problem_type.capitalize()}")

    models = {
        "classification": {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(),
            "SVM": SVC()
        },
        "regression": {
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "SVR": SVR()
        }
    }

    selected_model = st.selectbox("Select a model:", list(models[problem_type].keys()))

    if st.button("Train Model"):
        X, y = preprocess_data(data, target_variable)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = models[problem_type][selected_model]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if problem_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.text(classification_report(y_test, y_pred))
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write(f"R-squared Score: {r2:.2f}")

        st.subheader("Feature Importance")
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            fig = px.bar(feature_importance, x='feature', y='importance', title='Feature Importance')
            st.plotly_chart(fig)

def advanced_analysis_page():
    if 'data' not in st.session_state or 'target_variable' not in st.session_state:
        st.warning("Please upload data first.")
        return

    st.header("Advanced Analysis")
    data = st.session_state['data']
    target_variable = st.session_state['target_variable']

    st.subheader("Dimensionality Reduction")
    method = st.radio("Select method:", ["PCA", "t-SNE"])
    n_components = st.slider("Number of components:", 2, 5, 2)

    X, y = preprocess_data(data, target_variable)

    if method == "PCA":
        reducer = PCA(n_components=n_components)
    else:
        reducer = TSNE(n_components=n_components, random_state=42)

    reduced_data = reducer.fit_transform(X)

    if n_components == 2:
        fig = px.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], color=y, title=f"{method} Visualization")
    elif n_components == 3:
        fig = px.scatter_3d(x=reduced_data[:, 0], y=reduced_data[:, 1], z=reduced_data[:, 2], color=y, title=f"{method} Visualization")
    else:
        st.warning("Cannot visualize more than 3 dimensions.")
        return

    st.plotly_chart(fig)

    st.subheader("Anomaly Detection")
    z_threshold = st.slider("Z-score threshold:", 2.0, 5.0, 3.0, 0.1)
    numeric_data = data.select_dtypes(include=[np.number])
    z_scores = np.abs(stats.zscore(numeric_data))
    anomalies = (z_scores > z_threshold).any(axis=1)
    st.write(f"Number of anomalies detected: {anomalies.sum()}")
    st.write("Anomalous data points:")
    st.write(data[anomalies])

def ai_insights_page():
    if 'data' not in st.session_state or 'target_variable' not in st.session_state:
        st.warning("Please upload data first.")
        return

    st.header("AI Insights")
    data = st.session_state['data']
    target_variable = st.session_state['target_variable']

    st.subheader("Data Insights")
    insights = get_ai_insights(data, target_variable)
    st.write(insights)

    st.subheader("Feature Engineering Suggestions")
    suggestions = get_feature_engineering_suggestions(data, target_variable)
    st.write(suggestions)

    st.subheader("Model Selection Recommendations")
    recommendations = get_model_recommendations(data, target_variable)
    st.write(recommendations)

def report_generation_page():
    if 'data' not in st.session_state or 'target_variable' not in st.session_state:
        st.warning("Please upload data first.")
        return

    st.header("Report Generation")
    data = st.session_state['data']
    target_variable = st.session_state['target_variable']

    if st.button("Generate Report"):
        report = generate_report(data, target_variable)
        st.markdown(report, unsafe_allow_html=True)

        report_html = f"<html><body>{report}</body></html>"
        b64 = base64.b64encode(report_html.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="data_science_report.html">Download HTML Report</a>'
        st.markdown(href, unsafe_allow_html=True)

def preprocess_data(data, target_variable):
    X = data.drop(target_variable, axis=1)
    y = data[target_variable]
    
    # Handle categorical variables
    cat_columns = X.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_columns:
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle datetime columns
    date_columns = X.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        X[f"{col}_year"] = X[col].dt.year
        X[f"{col}_month"] = X[col].dt.month
        X[f"{col}_day"] = X[col].dt.day
        X = X.drop(col, axis=1)
    
    # Handle missing values
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    X[numeric_columns] = imputer.fit_transform(X[numeric_columns])
    
    # Scale features
    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    
    return X, y

def get_ai_insights(data, target_variable):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""Analyze the following dataset:
    
    Columns: {', '.join(data.columns)}
    Target variable: {target_variable}
    
    Provide insights about the data, including:
    1. Potential relationships between features
    2. Important features for predicting the target variable
    3. Any patterns or trends in the data
    4. Suggestions for further analysis
    
    Be specific and provide actionable insights based on the dataset structure."""
    
    response = model.generate_content(prompt)
    return response.text

def get_feature_engineering_suggestions(data, target_variable):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""Given the following dataset information:
    
    Columns: {', '.join(data.columns)}
    Target variable: {target_variable}
    
    Suggest advanced feature engineering techniques that could improve model performance. 
    Consider the following aspects:
    1. Creating interaction terms
    2. Polynomial features
    3. Date/time feature extraction (if applicable)
    4. Text feature extraction (if applicable)
    5. Domain-specific feature creation
    
    Provide specific suggestions based on the dataset structure."""
    
    response = model.generate_content(prompt)
    return response.text

def get_model_recommendations(data, target_variable):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""Given the following dataset information:
    
    Columns: {', '.join(data.columns)}
    Target variable: {target_variable}
    Number of samples: {len(data)}
    
    Recommend suitable machine learning models for this problem. 
    Consider the following aspects:
    1. Problem type (classification or regression)
    2. Dataset size
    3. Feature types
    4. Potential for overfitting or underfitting
    5. Model interpretability requirements
    
    Provide specific model recommendations and explain why they would be suitable for this dataset."""
    
    response = model.generate_content(prompt)
    return response.text

def generate_report(data, target_variable):
    report = f"""
    <h1>Data Science Report</h1>

    <h2>Dataset Overview</h2>
    <p>Number of samples: {data.shape[0]}</p>
    <p>Number of features: {data.shape[1] - 1}</p>
    <p>Target variable: {target_variable}</p>

    <h2>Data Types</h2>
    {data.dtypes.to_frame().to_html()}

    <h2>Summary Statistics</h2>
    {data.describe().to_html()}

    <h2>Missing Values</h2>
    {data.isnull().sum().to_frame().to_html()}

    <h2>Correlation Analysis</h2>
    <p>Top 5 correlations with the target variable:</p>
    {data.corr()[target_variable].sort_values(ascending=False).head().to_frame().to_html()}

    <h2>Feature Importance</h2>
    {feature_importance_analysis(data, target_variable)}

    <h2>Model Performance</h2>
    {model_performance_summary(data, target_variable)}

    <h2>AI Insights</h2>
    {get_ai_insights(data, target_variable)}

    <h2>Feature Engineering Suggestions</h2>
    {get_feature_engineering_suggestions(data, target_variable)}

    <h2>Model Recommendations</h2>
    {get_model_recommendations(data, target_variable)}
    """
    return report

def feature_importance_analysis(data, target_variable):
    X, y = preprocess_data(data, target_variable)
    
    if data[target_variable].dtype == 'object' or data[target_variable].nunique() < 10:
        importance = mutual_info_classif(X, y)
    else:
        importance = mutual_info_regression(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance.head(10).to_html()

def model_performance_summary(data, target_variable):
    X, y = preprocess_data(data, target_variable)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if data[target_variable].dtype == 'object' or data[target_variable].nunique() < 10:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return f"<p>Random Forest Classifier Accuracy: {accuracy:.2f}</p>"
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return f"<p>Random Forest Regressor - MSE: {mse:.2f}, R-squared: {r2:.2f}</p>"

def get_ai_insights(data, target_variable):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""Analyze the following dataset:
    
    Columns: {', '.join(data.columns)}
    Target variable: {target_variable}
    
    Provide insights about the data, including:
    1. Potential relationships between features
    2. Important features for predicting the target variable
    3. Any patterns or trends in the data
    4. Suggestions for further analysis
    
    Be specific and provide actionable insights based on the dataset structure."""
    
    response = model.generate_content(prompt)
    return response.text

def get_feature_engineering_suggestions(data, target_variable):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""Given the following dataset information:
    
    Columns: {', '.join(data.columns)}
    Target variable: {target_variable}
    
    Suggest advanced feature engineering techniques that could improve model performance. 
    Consider the following aspects:
    1. Creating interaction terms
    2. Polynomial features
    3. Date/time feature extraction (if applicable)
    4. Text feature extraction (if applicable)
    5. Domain-specific feature creation
    
    Provide specific suggestions based on the dataset structure."""
    
    response = model.generate_content(prompt)
    return response.text

def get_model_recommendations(data, target_variable):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""Given the following dataset information:
    
    Columns: {', '.join(data.columns)}
    Target variable: {target_variable}
    Number of samples: {len(data)}
    
    Recommend suitable machine learning models for this problem. 
    Consider the following aspects:
    1. Problem type (classification or regression)
    2. Dataset size
    3. Feature types
    4. Potential for overfitting or underfitting
    5. Model interpretability requirements
    
    Provide specific model recommendations and explain why they would be suitable for this dataset."""
    
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    main()
