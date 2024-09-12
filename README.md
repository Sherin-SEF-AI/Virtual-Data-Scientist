# Virtual-Data-Scientist


# Advanced AI-Driven Virtual Data Scientist

This project is an interactive Streamlit-based web application designed to assist data scientists and analysts in performing various data science tasks with ease. It integrates machine learning models, feature engineering, and Google Gemini AI insights to help users gain deep insights into their data, build predictive models, and generate reports.

## Features

1. **Data Upload & Exploration**: 
   - Upload CSV files for analysis.
   - View dataset preview, summary statistics, data types, missing values, and correlations.
   - Visualize data distributions with histograms and heatmaps.

2. **Feature Analysis**: 
   - Conduct feature importance analysis using mutual information techniques.
   - Display feature correlations and visualize pairwise feature relationships with scatter matrices.

3. **Model Building**:
   - Select and train classification or regression models, including Random Forest, Gradient Boosting, Logistic Regression, and Support Vector Machines (SVMs).
   - View model performance metrics such as accuracy, mean squared error, and R-squared score.
   - Analyze feature importance for trained models.

4. **Advanced Analysis**:
   - Perform dimensionality reduction using PCA and t-SNE methods.
   - Detect anomalies using Z-scores for numeric features.
   
5. **AI Insights**: 
   - Obtain actionable insights about the dataset, including feature relationships, important predictors, and suggested analyses, using Google Gemini AI.
   - Receive suggestions for advanced feature engineering and model recommendations based on the dataset structure.

6. **Report Generation**: 
   - Automatically generate a comprehensive HTML report that includes data summaries, feature importance analysis, model performance, AI insights, and more.

## Installation

To run the application locally, follow the steps below:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sherin-SEF-AI/Virtual-Data-Scientist.git
   cd Virtual-Data-Scientist

Dependencies

The project uses the following libraries:


streamlit: Web interface framework

pandas: Data manipulation and analysis

numpy: Numerical operations

matplotlib, seaborn, plotly: Data visualization

scikit-learn: Machine learning models and preprocessing

google-generativeai: Google Gemini AI for insights and recommendations

textblob: Text processing for natural language analysis

wordcloud: Visualize word frequencies in text data
