import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, accuracy_score
import plotly.express as px
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
# Add any other necessary imports

# Set page configuration
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

st.title("Loan Default Prediction Project")
st.write("This application demonstrates an end-to-end data science project focusing on predicting customer loan defaults.")

# Load data function
@st.cache(allow_output_mutation=True)
def load_data():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    return train_data, test_data

# Initialize session state for train and test data
if 'train_df' not in st.session_state or 'test_df' not in st.session_state:
    st.session_state['train_df'], st.session_state['test_df'] = load_data()

# Access data from session state
train_df = st.session_state['train_df']
test_df = st.session_state['test_df']

# @st.cache_data
# def load_data():
#     train_data = pd.read_csv('train.csv')
#     test_data = pd.read_csv('test.csv')
#     return train_data, test_data

# train_df, test_df = load_data()

tab1, tab2, tab3, tab4, tab5,tab6, tab7 = st.tabs(["Data Exploration", "Data Preprocessing", "Data Visualization", "Feature Engineering","Model Training","Hyperparameter Tuning","Model Testing & Submission Generation"])


with tab1:
    st.header("Data Exploration")

    # Displaying data
    if st.checkbox('Show training data'):
        st.write(train_df.head())

    # Displaying data
    if st.checkbox('Show testing data'):
        st.write(test_df.head())
    
    # Check for duplicates
    if st.checkbox('Check for duplicates in training data'):
        st.write('Number of duplicate entries in training data:', train_df.duplicated().sum())

    # Display data types
    if st.checkbox('Show data types of training data'):
        st.write(train_df.dtypes)

    # Add more code for data exploration (e.g., visualizations)

with tab2:
    st.header("Data Preprocessing")

    # Descriptive statistics for numerical features
    if st.checkbox('Show descriptive statistics for numerical features'):
        st.write(train_df.describe().T)

    # Descriptive statistics for categorical features
    if st.checkbox('Show descriptive statistics for categorical features'):
        st.write(train_df.describe(include='object').T)

    # Example: Display shape of the data
    if st.checkbox('Show shape of the datasets'):
        st.write("Training Data Shape:", train_df.shape)
        st.write("Testing Data Shape:", test_df.shape)

    # Missing Values
    st.subheader("Missing Values Analysis")
    if st.checkbox('Show missing values in training data'):
        missing_values = train_df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        st.write(missing_values)

    # Outlier Detection
    st.subheader("Outlier Detection")
    numerical_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
    selected_column = st.selectbox("Select a column for outlier detection", numerical_columns)
    if st.button('Detect Outliers in Selected Column'):
        Q1 = train_df[selected_column].quantile(0.25)
        Q3 = train_df[selected_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = train_df[(train_df[selected_column] < lower_bound) | (train_df[selected_column] > upper_bound)]
        st.write("Number of Outliers:", outliers.shape[0])
        st.write(outliers)

with tab3:
    st.header("Data Visualization")

    # Target Distribution Visualization with Plotly
    st.subheader("Target Distribution")
    if st.button("Show Target Distribution", key="target_dist"):
        # Calculate the percentage distribution
        target_counts = train_df['target'].value_counts(normalize=True) * 100
        target_counts = target_counts.reset_index()
        target_counts.columns = ['target', 'percentage']

        # Create a bar plot with Plotly
        fig = px.bar(target_counts, x='target', y='percentage', text='percentage')
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                          title='Percentage Distribution of Target Variable',
                          xaxis_title='Target',
                          yaxis_title='Percentage',
                          yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)

    # Histogram with Plotly
    st.subheader("Histogram")
    hist_column = st.selectbox("Select Column for Histogram", train_df.columns, key='histogram_column')
    if st.button("Generate Histogram", key="hist"):
        fig = px.histogram(train_df, x=hist_column, marginal="box")  # boxplot to understand skewness
        st.plotly_chart(fig, use_container_width=True)

    # Scatter Plot with Plotly
    st.subheader("Scatter Plot")
    scatter_x = st.selectbox("Select X-axis for Scatter Plot", train_df.columns, key="scatter_x")
    scatter_y = st.selectbox("Select Y-axis for Scatter Plot", train_df.columns, key="scatter_y")
    if st.button("Generate Scatter Plot", key="scatter"):
        fig = px.scatter(train_df, x=scatter_x, y=scatter_y)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    if st.button("Generate Correlation Heatmap", key="corr"):
        plt.figure(figsize=(10, 8))
        sns.heatmap(train_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(plt)

with tab4:
    st.header("Feature Engineering")

    # Create Age from date_of_birth
    st.subheader("Create Age from Date of Birth")
    dob_column = st.selectbox("Select Date of Birth Column", train_df.columns, key="dob_column")
    if st.button("Create Age Feature", key="create_age"):
        current_year = pd.Timestamp.now().year
        train_df['age'] = current_year - pd.to_datetime(train_df[dob_column]).dt.year
        test_df['age'] = current_year - pd.to_datetime(test_df[dob_column]).dt.year
        st.write("Age feature created based on", dob_column)

        # Update session state
        st.session_state['train_df'] = train_df
        st.session_state['test_df'] = test_df

    st.subheader("Create New Features")
    # Placeholder for feature creation code
    # Example: Creating a new feature by combining existing features
    if st.checkbox('Create a new feature'):
        new_feature_name = st.text_input("New Feature Name")
        existing_feature_1 = st.selectbox("Select First Feature", train_df.columns, index=0, key="feat1")
        existing_feature_2 = st.selectbox("Select Second Feature", train_df.columns, index=1, key="feat2")
        operation = st.selectbox("Select Operation", ["Add", "Subtract", "Multiply", "Divide"], key="operation")
        
        if st.button("Create Feature", key="create_feature"):
            def apply_operation(df, feature1, feature2, op):
                if op == "Add":
                    return df[feature1] + df[feature2]
                elif op == "Subtract":
                    return df[feature1] - df[feature2]
                elif op == "Multiply":
                    return df[feature1] * df[feature2]
                elif op == "Divide":
                    return df[feature1] / df[feature2]
                else:
                    return None

            # Apply the operation to both training and testing datasets
            train_df[new_feature_name] = apply_operation(train_df, existing_feature_1, existing_feature_2, operation)
            test_df[new_feature_name] = apply_operation(test_df, existing_feature_1, existing_feature_2, operation)

            # Update session state
            st.session_state['train_df'] = train_df
            st.session_state['test_df'] = test_df

            st.write(f"New feature '{new_feature_name}' created in both training and testing datasets!")

    # Drop Variables
    st.subheader("Drop Variables")
    col_to_drop = st.multiselect("Select Columns to Drop", train_df.columns, key="drop_cols")
    if st.button("Drop Selected Columns", key="drop_cols_button"):
        train_df.drop(columns=col_to_drop, inplace=True)
        test_df.drop(columns=col_to_drop, inplace=True)
        st.write("Dropped columns:", col_to_drop)

        # Update session state
        st.session_state['train_df'] = train_df
        st.session_state['test_df'] = test_df

with tab5:
    st.header("Model Training")
    

    # Data Scaling
    st.subheader("Data Scaling")
    scale_option = st.selectbox("Choose a scaling method", 
                                ["None", "Standard Scaler", "MinMax Scaler", "MaxAbs Scaler", "Robust Scaler", "Normalizer"])

    # Scale the data based on selection
    if scale_option != "None":
        scaler = {"Standard Scaler": StandardScaler(), 
                  "MinMax Scaler": MinMaxScaler(), 
                  "MaxAbs Scaler": MaxAbsScaler(), 
                  "Robust Scaler": RobustScaler(), 
                  "Normalizer": Normalizer()}[scale_option]
        X_train_scaled = scaler.fit_transform(train_df.drop('target', axis=1))
        y_train = train_df['target']
    else:
        X_train_scaled = train_df.drop('target', axis=1)
        y_train = train_df['target']


    # Model Selection
    st.subheader("Model Selection")
    model_option = st.selectbox("Choose a model", ["Logistic Regression", "SVM", "Random Forest", "XGBoost"])

    trained_model = None
    y_pred = None
    is_statsmodels = False

    if model_option == "Logistic Regression":
        X_train_scaled_sm = sm.add_constant(X_train_scaled)
        model = sm.Logit(y_train, X_train_scaled_sm)
        result = model.fit()
        st.write("Logistic Regression model trained successfully!")
        st.write(result.summary2())

        y_pred = result.predict(X_train_scaled_sm)
        y_pred_label = (y_pred > 0.5).astype(int)

        # Metrics
        accuracy = accuracy_score(y_train, y_pred_label)
        st.write(f"Accuracy: {accuracy:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_train, y_pred_label)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_train, y_pred)
        fig = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC={roc_auc_score(y_train, y_pred):.4f})',
                      labels=dict(x='False Positive Rate', y='True Positive Rate'), width=700, height=500)
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_train, y_pred)
        fig = px.area(x=recall, y=precision, title='Precision-Recall Curve', labels=dict(x='Recall', y='Precision'), width=700, height=500)
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=1, y1=0)
        st.plotly_chart(fig)
        is_statsmodels = True
    else:
        if model_option == "SVM":
            model = SVC(probability=True)
        elif model_option == "Random Forest":
            model = RandomForestClassifier()
        elif model_option == "XGBoost":
            model = XGBClassifier()

        model.fit(X_train_scaled, y_train)
        trained_model = model
        y_pred = model.predict_proba(X_train_scaled)[:, 1]
        st.write(f"Model {model_option} trained successfully!")

    # Display metrics
    if not is_statsmodels:
        y_pred_label = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_train, y_pred_label)
        st.write(f"Accuracy: {accuracy:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_train, y_pred_label)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_train, y_pred)
        fig = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC={roc_auc_score(y_train, y_pred):.4f})',
                        labels=dict(x='False Positive Rate', y='True Positive Rate'), width=700, height=500)
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_train, y_pred)
        fig = px.area(x=recall, y=precision, title='Precision-Recall Curve', labels=dict(x='Recall', y='Precision'), width=700, height=500)
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=1, y1=0)
        st.plotly_chart(fig)

    if model_option == "Random Forest" or model_option == "XGBoost":
        model.fit(X_train_scaled, y_train)
        trained_model = model
        y_pred = model.predict_proba(X_train_scaled)[:, 1]
        st.write(f"Model {model_option} trained successfully!")

        # Display Feature Importance (only for Random Forest and XGBoost)
        feature_names = train_df.drop('target', axis=1).columns  # Adjust as needed
        importances = trained_model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]

        # Convert to Pandas DataFrame for easier plotting
        df_importances = pd.DataFrame({'feature': feature_names[sorted_indices], 'importance': importances[sorted_indices]})
        fig = px.bar(df_importances, x='importance', y='feature', orientation='h')
        fig.update_layout(title='Feature Importances', xaxis_title='Importance', yaxis_title='Features')
        st.plotly_chart(fig, use_container_width=True)

    # Button to save the trained model
    if st.button("Save Trained Model"):
        filename = st.text_input("Enter filename to save the model", value="trained_model.pkl")
        if filename:
            joblib.dump(trained_model, filename)
            st.write(f"Model saved as {filename}")

with tab6:
    st.header("Hyperparameter Tuning")

    # Model Selection for Tuning
    st.subheader("Select Model for Tuning")
    tuning_model_option = st.selectbox("Choose a model to tune", ["Logistic Regression", "SVM", "Random Forest", "XGBoost"])

    # Define hyperparameters for each model
    if tuning_model_option == "Logistic Regression":
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2']
        }
        model = LogisticRegression(solver='liblinear')
    elif tuning_model_option == "SVM":
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        model = SVC()
    elif tuning_model_option == "Random Forest":
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        model = RandomForestClassifier()
    elif tuning_model_option == "XGBoost":
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'min_child_weight': [1, 2, 3, 4],
            'subsample': [0.5, 0.7, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.9]
        }
        model = XGBClassifier()

    # Hyperparameter tuning button
    if st.button("Tune Hyperparameters"):
        # GridSearchCV for tuning
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train_scaled, y_train)  # Ensure X_train_scaled and y_train are defined

        # Show tuning results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        st.write("Best parameters found: ", best_params)
        st.write("Best score: ", best_score)

        # Button to save the trained model
        if st.button("Save Tuned Model"):
            filename = st.text_input("Enter filename to save the model", value="trained_model.pkl")
            if filename:
                joblib.dump(grid_search.best_estimator_, filename)
                st.write(f"Model saved as {filename}")

    # # Button to save the tuned model
    # if st.button("Save Tuned Model"):
    #     filename = st.text_input("Enter filename to save the tuned model", value="tuned_model.pkl", key="tuned_filename")
    #     if filename:
    #         joblib.dump(grid_search.best_estimator_, filename)
    #         st.write(f"Model saved as {filename}")
with tab7:
    st.header("Model Testing & Submission Generation")

    # # Load a trained model
    # st.subheader("Load a Trained Model")
    # model_file = st.file_uploader("Upload a trained model file", type=["pkl"])
    # if model_file:
    #     loaded_model = joblib.load(model_file)
    #     st.write("Model loaded successfully.")

    model_file = "trained_model.pkl"
    
    loaded_model = joblib.load(model_file)
    st.write("Model loaded successfully.")

    if st.button("Generate Predictions"):

        # Test the model and create submission
        if loaded_model:
            
            # Assuming test_df is already preprocessed similarly to train_df
            # Extract ID if present
            if 'id' in test_df.columns:
                ids = test_df['id']
                X_test = test_df.drop('id', axis=1)
            else:
                ids = test_df.index
                X_test = test_df

            # Generate predictions
            predictions = loaded_model.predict_proba(X_test)[:, 1]
            
            # Create a submission dataframe
            submission_df = pd.DataFrame({'id': ids, 'prediction': predictions})
            submission_df['outcome'] = (submission_df['prediction'] > 0.5).astype(int)

            # Displaying the first few rows of the submission file
            st.write("Preview of the submission file:")
            st.write(submission_df.head())

            # Download button for the submission file
            @st.cache
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(submission_df)
            st.download_button(
                label="Download Submission as CSV",
                data=csv,
                file_name='submission.csv',
                mime='text/csv',
            )
    
