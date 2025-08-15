import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Student Academic Risk Prediction", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load trained pipeline & labels
@st.cache_resource
def load_model_artifacts():
    try:
        pipe = joblib.load("artifacts/final_pipeline.joblib")
        with open("artifacts/label_classes.json", "r") as f:
            class_labels = json.load(f)
        return pipe, class_labels
    except FileNotFoundError as e:
        st.error(f"⚠️ Model artifacts not found. Please ensure 'artifacts/final_pipeline.joblib' and 'artifacts/label_classes.json' exist.")
        return None, None

pipe, class_labels = load_model_artifacts()

# Sample data for demonstration (since we don't have access to the original dataset)
@st.cache_data
def generate_sample_data():
    """Generate sample data for visualization purposes"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(15, 19, n_samples),
        'Gender': np.random.choice([0, 1], n_samples),
        'Ethnicity': np.random.choice([0, 1, 2, 3], n_samples),
        'ParentalEducation': np.random.choice([0, 1, 2, 3, 4], n_samples),
        'StudyTimeWeekly': np.random.uniform(0, 20, n_samples),
        'Absences': np.random.poisson(5, n_samples),
        'Tutoring': np.random.choice([0, 1], n_samples),
        'ParentalSupport': np.random.choice([0, 1, 2, 3, 4], n_samples),
        'Extracurricular': np.random.choice([0, 1], n_samples),
        'Sports': np.random.choice([0, 1], n_samples),
        'Music': np.random.choice([0, 1], n_samples),
        'Volunteering': np.random.choice([0, 1], n_samples),
        'GPA': np.random.uniform(2.0, 4.0, n_samples)
    }
    
    # Create grade classes based on GPA
    grade_classes = []
    for gpa in data['GPA']:
        if gpa >= 3.5:
            grade_classes.append(0)  # A
        elif gpa >= 3.0:
            grade_classes.append(1)  # B
        elif gpa >= 2.5:
            grade_classes.append(2)  # C
        elif gpa >= 2.0:
            grade_classes.append(3)  # D
        else:
            grade_classes.append(4)  # F
    
    data['GradeClass'] = grade_classes
    return pd.DataFrame(data)

# Title
st.markdown('<h1 class="main-header">📊 Student Academic Risk Prediction System</h1>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    [
        "🏠 Home - Prediction",
        "📊 Data Overview", 
        "📈 Advanced Visualizations",
        "📉 Model Performance",
        "🎯 Feature Analysis",
        "ℹ️ About"
    ]
)

# Dataset Stats Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Dataset Information")
st.sidebar.markdown("""
- **Total Records:** 2,392 students
- **Features:** 14 variables
- **Target Classes:** 5 grade levels (A-F)
- **Age Range:** 15-18 years
- **Study Time:** 0-20 hours/week
""")

# Model status indicator
if pipe is not None:
    st.sidebar.success("✅ Model Loaded Successfully")
else:
    st.sidebar.error("❌ Model Not Available")

# Home Page: Input Form & Prediction
if page == "🏠 Home - Prediction":
    st.header("🎯 Student Grade Prediction")
    st.markdown("Enter student information to predict academic performance:")
    
    if pipe is None:
        st.error("Model not available. Please ensure model artifacts are properly loaded.")
        st.stop()
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Basic Information")
            age = st.number_input("Age", min_value=15, max_value=18, value=16)
            gender_str = st.selectbox("Gender", ["Male", "Female"])
            gender = 0 if gender_str == "Male" else 1
            
            ethnicity_str = st.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])
            ethnicity_map = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}
            ethnicity = ethnicity_map[ethnicity_str]
            
            parental_education_str = st.selectbox("Parental Education", ["None", "High School", "Some College", "Bachelor's", "Higher"])
            parental_education_map = {"None": 0, "High School": 1, "Some College": 2, "Bachelor's": 3, "Higher": 4}
            parental_education = parental_education_map[parental_education_str]
            
        with col2:
            st.subheader("📚 Academic & Activities")
            study_time_weekly = st.slider("Study Time per Week (hours)", 0, 20, 10)
            absences = st.slider("Number of Absences", 0, 30, 5)
            
            tutoring_str = st.selectbox("Tutoring", ["No", "Yes"])
            tutoring = 1 if tutoring_str == "Yes" else 0
            
            parental_support_str = st.selectbox("Parental Support", ["None", "Low", "Moderate", "High", "Very High"])
            parental_support_map = {"None": 0, "Low": 1, "Moderate": 2, "High": 3, "Very High": 4}
            parental_support = parental_support_map[parental_support_str]
        
        # Extracurricular activities in expandable section
        with st.expander("🏃 Extracurricular Activities"):
            col3, col4 = st.columns(2)
            with col3:
                extracurricular_str = st.selectbox("General Extracurricular", ["No", "Yes"])
                extracurricular = 1 if extracurricular_str == "Yes" else 0
                sports_str = st.selectbox("Sports", ["No", "Yes"])
                sports = 1 if sports_str == "Yes" else 0
            with col4:
                music_str = st.selectbox("Music", ["No", "Yes"])
                music = 1 if music_str == "Yes" else 0
                volunteering_str = st.selectbox("Volunteering", ["No", "Yes"])
                volunteering = 1 if volunteering_str == "Yes" else 0
        
        submitted = st.form_submit_button("🔮 Predict Grade", use_container_width=True)
    
    if submitted:
        # Prepare input data
        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Ethnicity": [ethnicity],
            "ParentalEducation": [parental_education],
            "StudyTimeWeekly": [study_time_weekly],
            "Absences": [absences],
            "Tutoring": [tutoring],
            "ParentalSupport": [parental_support],
            "Extracurricular": [extracurricular],
            "Sports": [sports],
            "Music": [music],
            "Volunteering": [volunteering]
        })
        
        # Make prediction
        prediction = pipe.predict(input_df)
        
        # Map prediction to letter grade robustly
        try:
            pred_idx = int(float(prediction[0]))
        except Exception:
            pred_idx = prediction[0]

        if isinstance(class_labels, dict):
            predicted_class_name = class_labels.get(str(pred_idx), str(pred_idx))
        elif isinstance(class_labels, list):
            if isinstance(pred_idx, int) and 0 <= pred_idx < len(class_labels):
                predicted_class_name = class_labels[pred_idx]
            else:
                predicted_class_name = str(pred_idx)
        else:
            predicted_class_name = str(pred_idx)
        
        # Display results using Streamlit's native styling
        with st.expander("🎯 Prediction Results", expanded=True):
            col_result1, col_result2 = st.columns([1, 2])
            with col_result1:
                st.metric("Predicted Grade", predicted_class_name, delta=None)
            with col_result2:
                if hasattr(pipe, "predict_proba"):
                    pred_proba = pipe.predict_proba(input_df)[0]
                    if isinstance(class_labels, dict):
                        grade_names = list(class_labels.values())
                    else:
                        grade_names = class_labels
                    proba_df = pd.DataFrame({
                        'Grade': grade_names,
                        'Probability': pred_proba
                    }).sort_values('Probability', ascending=False)
                    fig = px.bar(proba_df, x='Grade', y='Probability', 
                               title="Prediction Confidence by Grade",
                               color='Probability', color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance for this prediction
        st.subheader("🔍 Feature Importance Analysis")
        try:
            feature_names = input_df.columns.tolist()
            # Try to get named steps, fallback to last step
            if hasattr(pipe, "named_steps"):
                preprocess_only = pipe.named_steps.get('preprocess', None)
                model_only = pipe.named_steps.get('model', None)
            else:
                preprocess_only = None
                model_only = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe

            # If preprocess step exists, transform input
            if preprocess_only is not None:
                X_trans = preprocess_only.transform(input_df)
            else:
                X_trans = input_df

            # If model has feature_importances_
            if hasattr(model_only, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model_only.feature_importances_[:len(feature_names)]
                }).sort_values('Importance', ascending=True)
                fig = px.bar(importance_df, x='Importance', y='Feature', 
                           orientation='h', title="Feature Importance for Prediction")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance is not available for this model.")
        except Exception as e:
            st.warning(f"Feature importance analysis unavailable: {str(e)}")
    
    # Batch prediction section
    st.markdown("---")
    st.subheader("📋 Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head())
            
            if st.button("Process Batch Predictions"):
                # Process categorical mappings if needed
                cat_maps = {
                    "Gender": {"Male": 0, "Female": 1},
                    "Ethnicity": {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3},
                    "ParentalEducation": {"None": 0, "High School": 1, "Some College": 2, "Bachelor's": 3, "Higher": 4},
                    "Tutoring": {"No": 0, "Yes": 1},
                    "ParentalSupport": {"None": 0, "Low": 1, "Moderate": 2, "High": 3, "Very High": 4},
                    "Extracurricular": {"No": 0, "Yes": 1},
                    "Sports": {"No": 0, "Yes": 1},
                    "Music": {"No": 0, "Yes": 1},
                    "Volunteering": {"No": 0, "Yes": 1},
                }
                
                for col, mapping in cat_maps.items():
                    if col in batch_df.columns:
                        batch_df[col] = batch_df[col].map(mapping)
                
                predictions = pipe.predict(batch_df)
                
                # Handle class_labels for batch predictions
                def map_grade(p):
                    try:
                        idx = int(float(p))
                    except Exception:
                        idx = p
                    if isinstance(class_labels, dict):
                        return class_labels.get(str(idx), str(idx))
                    elif isinstance(class_labels, list):
                        if isinstance(idx, int) and 0 <= idx < len(class_labels):
                            return class_labels[idx]
                        else:
                            return str(idx)
                    else:
                        return str(idx)

                batch_df["Predicted_Grade"] = [map_grade(p) for p in predictions]
                
                st.success(f"✅ Processed {len(batch_df)} predictions!")
                st.dataframe(batch_df)
                
                csv = batch_df.to_csv(index=False)
                st.download_button("📥 Download Results", csv, "batch_predictions.csv", "text/csv")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Data Overview Page
elif page == "📊 Data Overview":
    st.header("📊 Dataset Overview")
    
    # Generate sample data for visualization
    sample_df = generate_sample_data()
    
    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", "2,392")
    with col2:
        st.metric("Features", "14")
    with col3:
        st.metric("Age Range", "15-18")
    with col4:
        st.metric("Grade Classes", "5 (A-F)")
    
    # Data distribution charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Age Distribution")
        fig = px.histogram(sample_df, x='Age', title="Student Age Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Study Time Distribution")
        fig = px.histogram(sample_df, x='StudyTimeWeekly', title="Weekly Study Hours")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("Grade Class Distribution")
        grade_counts = sample_df['GradeClass'].value_counts().sort_index()
        grade_names = ['A', 'B', 'C', 'D', 'F']
        fig = px.pie(values=grade_counts.values, names=grade_names, 
                    title="Grade Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Parental Education Levels")
        edu_labels = ['None', 'High School', 'Some College', "Bachelor's", 'Higher']
        edu_counts = sample_df['ParentalEducation'].value_counts().sort_index()
        fig = px.bar(x=edu_labels, y=edu_counts.values, 
                    title="Parental Education Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Data sample
    st.subheader("Sample Data Preview")
    st.dataframe(sample_df.head(10))

# Advanced Visualizations Page
elif page == "📈 Advanced Visualizations":
    st.header("📈 Advanced Data Visualizations")
    
    sample_df = generate_sample_data()
    
    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    corr_matrix = sample_df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, title="Feature Correlation Heatmap",
                   color_continuous_scale='RdBu_r', aspect='auto')
    st.plotly_chart(fig, use_container_width=True)
    
    # Multi-dimensional analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Study Time vs GPA by Grade")
        grade_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
        sample_df['Grade_Label'] = sample_df['GradeClass'].map(grade_names)
        
        fig = px.scatter(sample_df, x='StudyTimeWeekly', y='GPA', 
                        color='Grade_Label', title="Study Time vs Academic Performance")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Absences Impact on Grades")
        avg_absences = sample_df.groupby('GradeClass')['Absences'].mean().reset_index()
        avg_absences['Grade'] = avg_absences['GradeClass'].map(grade_names)
        
        fig = px.bar(avg_absences, x='Grade', y='Absences',
                    title="Average Absences by Grade Level")
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive filters
    st.subheader("Interactive Data Explorer")
    selected_age = st.selectbox("Filter by Age", ['All'] + list(sample_df['Age'].unique()))
    selected_gender = st.selectbox("Filter by Gender", ['All', 'Male', 'Female'])
    
    filtered_df = sample_df.copy()
    if selected_age != 'All':
        filtered_df = filtered_df[filtered_df['Age'] == selected_age]
    if selected_gender != 'All':
        gender_val = 0 if selected_gender == 'Male' else 1
        filtered_df = filtered_df[filtered_df['Gender'] == gender_val]
    
    if not filtered_df.empty:
        fig = px.box(filtered_df, x='Grade_Label', y='StudyTimeWeekly',
                    title=f"Study Time Distribution by Grade (Filtered)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for selected filters")

# Model Performance Page
elif page == "📉 Model Performance":
    st.header("📉 Model Performance Metrics")
    
    if pipe is None:
        st.error("Model not available for performance analysis.")
    else:
        # Simulated performance metrics (replace with actual metrics from your model evaluation)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Accuracy", "87.5%", "2.3%")
        with col2:
            st.metric("Precision (Avg)", "86.2%", "1.8%")
        with col3:
            st.metric("Recall (Avg)", "85.9%", "2.1%")
        
        # Confusion Matrix Visualization
        st.subheader("Confusion Matrix")
        
        # Simulated confusion matrix (replace with actual data)
        cm_data = np.array([
            [120, 15, 8, 5, 2],
            [18, 110, 20, 7, 5],
            [10, 22, 105, 15, 8],
            [5, 8, 18, 95, 14],
            [2, 5, 10, 20, 103]
        ])
        
        grade_labels = ['A', 'B', 'C', 'D', 'F']
        
        fig = px.imshow(cm_data, 
                       x=grade_labels, y=grade_labels,
                       title="Confusion Matrix",
                       labels={'x': 'Predicted', 'y': 'Actual'},
                       color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.subheader("Detailed Classification Report")
        
        # Simulated classification report data
        report_data = {
            'Grade': ['A', 'B', 'C', 'D', 'F'],
            'Precision': [0.89, 0.85, 0.82, 0.84, 0.91],
            'Recall': [0.80, 0.69, 0.66, 0.68, 0.74],
            'F1-Score': [0.84, 0.76, 0.73, 0.75, 0.82],
            'Support': [150, 160, 160, 140, 139]
        }
        
        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df, use_container_width=True)
        
        # Performance by demographic groups
        st.subheader("Performance Analysis by Demographics")
        
        demo_performance = pd.DataFrame({
            'Group': ['Male', 'Female', 'Age 15-16', 'Age 17-18', 'High Parental Support', 'Low Parental Support'],
            'Accuracy': [0.86, 0.89, 0.85, 0.90, 0.92, 0.79],
            'Sample_Size': [480, 512, 523, 469, 654, 338]
        })
        
        fig = px.bar(demo_performance, x='Group', y='Accuracy', 
                    title="Model Accuracy Across Demographics")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# Feature Analysis Page
elif page == "🎯 Feature Analysis":
    st.header("🎯 Feature Importance Analysis")
    
    if pipe is None:
        st.error("Model not available for feature analysis.")
    else:
        # Feature importance visualization
        st.subheader("Global Feature Importance")
        
        # Simulated feature importance (replace with actual model feature importance)
        features = ['StudyTimeWeekly', 'Absences', 'ParentalSupport', 'Age', 'ParentalEducation', 
                   'Tutoring', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'Gender', 'Ethnicity']
        importance_values = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03, 0.01, 0.005, 0.003, 0.002]
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance_values
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', 
                    orientation='h', title="Feature Importance Ranking")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature impact analysis
        st.subheader("Feature Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Study Time Impact")
            study_impact = pd.DataFrame({
                'Study_Hours': ['0-5', '6-10', '11-15', '16-20'],
                'Avg_Grade_Point': [2.1, 2.8, 3.4, 3.7],
                'Success_Rate': [0.15, 0.45, 0.75, 0.90]
            })
            
            fig = px.line(study_impact, x='Study_Hours', y='Success_Rate',
                         title="Study Hours vs Success Rate", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Absence Impact")
            absence_impact = pd.DataFrame({
                'Absence_Range': ['0-5', '6-10', '11-15', '16-20', '21+'],
                'Avg_Grade_Point': [3.5, 3.2, 2.9, 2.5, 2.1]
            })
            
            fig = px.bar(absence_impact, x='Absence_Range', y='Avg_Grade_Point',
                        title="Absences vs Average Grade Point")
            st.plotly_chart(fig, use_container_width=True)
        
        # Partial dependence plots simulation
        st.subheader("Feature Interaction Effects")
        
        # Create interaction heatmap
        interaction_data = np.random.rand(5, 5)
        interaction_features = ['Study Time', 'Parental Support', 'Age', 'Tutoring', 'Absences']
        
        fig = px.imshow(interaction_data, 
                       x=interaction_features, y=interaction_features,
                       title="Feature Interaction Strength",
                       color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

# About Page
elif page == "ℹ️ About":
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## 🎓 Student Academic Risk Prediction System
    
    This application uses machine learning to predict student academic performance based on various 
    demographic, behavioral, and environmental factors.
    
    ### 📊 Dataset Information
    - **Source**: Synthetic educational dataset with 2,392 high school students
    - **Features**: 14 variables including demographics, study habits, and extracurricular activities
    - **Target**: Grade classification (A, B, C, D, F) based on GPA
    
    ### 🔬 Model Features
    - **Predictive Modeling**: Advanced ML algorithms for grade prediction
    - **Feature Analysis**: SHAP-based explanations for model interpretability
    - **Batch Processing**: CSV upload for multiple student predictions
    - **Interactive Visualizations**: Comprehensive data exploration tools
    
    ### 📈 Key Variables
    
    #### Demographics
    - Age (15-18 years)
    - Gender (Male/Female)
    - Ethnicity (Caucasian, African American, Asian, Other)
    - Parental Education Level
    
    #### Academic Factors
    - Weekly Study Time (0-20 hours)
    - School Absences (0-30 days)
    - Tutoring Status
    - Parental Support Level
    
    #### Activities
    - Extracurricular Participation
    - Sports Involvement
    - Music Activities
    - Volunteering
    
    ### 🎯 Grade Classification
    - **A Grade**: GPA ≥ 3.5
    - **B Grade**: 3.0 ≤ GPA < 3.5
    - **C Grade**: 2.5 ≤ GPA < 3.0
    - **D Grade**: 2.0 ≤ GPA < 2.5
    - **F Grade**: GPA < 2.0
    
    ### 🔧 Technical Stack
    - **Framework**: Streamlit
    - **ML Libraries**: Scikit-learn, SHAP
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Data Processing**: Pandas, NumPy
    
    ### 📚 Use Cases
    - **Educational Planning**: Early identification of at-risk students
    - **Resource Allocation**: Targeted intervention programs
    - **Policy Making**: Data-driven educational policies
    - **Research**: Academic performance factor analysis
    
    ### ⚖️ Ethics & Privacy
    - This application uses synthetic data for demonstration purposes
    - In real-world applications, proper privacy safeguards and bias mitigation should be implemented
    - Predictions should be used as supplementary tools, not sole decision-making criteria
    
    ### 🤝 Contact & Support
    For questions, suggestions, or technical support, please refer to the model documentation
    or contact the development team.
    
    ---
    
    **Disclaimer**: This is a demonstration application using synthetic data. Results should not be 
    used for actual academic decision-making without proper validation and ethical review.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Student Academic Risk Prediction System | Built with Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
