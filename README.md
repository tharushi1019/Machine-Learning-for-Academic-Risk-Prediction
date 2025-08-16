# Machine Learning for Academic Risk Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://25xpanhzgvp4rapp9pzhm6r.streamlit.app/)
[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📚 Overview

This project implements machine learning models to predict academic risk and identify students who may need additional support to succeed in their academic journey. By analyzing various student factors including demographics, academic performance, and behavioral patterns, the system can help educational institutions proactively intervene and provide targeted support.

## 🎯 Objectives

- **Early Risk Detection**: Identify students at risk of academic failure or dropout before it happens
- **Data-Driven Insights**: Provide actionable insights to educators and administrators
- **Intervention Planning**: Enable timely and targeted interventions to improve student outcomes
- **Model Comparison**: Evaluate multiple machine learning algorithms to find the best performing model

## 🚀 Live Demo

Try the interactive web application: [Academic Risk Prediction App](https://25xpanhzgvp4rapp9pzhm6r.streamlit.app/)

## 📊 Dataset Features

The model analyzes various student attributes including:

- **Academic Performance**: GPA, test scores, assignment completion rates
- **Attendance**: Class attendance patterns and frequency
- **Demographics**: Age, gender, socioeconomic background
- **Behavioral Factors**: Engagement levels, participation in activities
- **Historical Data**: Previous academic performance and trends

## 🛠️ Technologies Used

- **Programming Language**: Python 3.8+
- **Machine Learning**: scikit-learn, pandas, numpy
- **Data Visualization**: matplotlib, seaborn, plotly
- **Web Framework**: Streamlit
- **Development Environment**: Jupyter Notebook
- **Model Evaluation**: Classification metrics, cross-validation

## 📁 Project Structure

```
Machine-Learning-for-Academic-Risk-Prediction/
├── app/
│   ├── app.py
│   └── requirement.txt
├── artifacts/
│   ├── final_pipeline.joblib
│   ├── label_classes.json
│   └── model_comparison.csv
├── data/
│   └──Student_performance_data _.csv
├── notebooks/
│   └── Explainable_Machine_Learning_for_Academic_Risk_Prediction.ipynb
├── reports/
│   └── Report.md

```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tharushi1019/Machine-Learning-for-Academic-Risk-Prediction.git
   cd Machine-Learning-for-Academic-Risk-Prediction
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**:
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Access the application**:
   Open your browser and go to `http://localhost:8501`

## 📈 Machine Learning Models

The project implements and compares multiple classification algorithms:

### Models Implemented:
- **Logistic Regression**: Baseline linear model for binary classification
- **Random Forest**: Ensemble method with feature importance analysis
- **Support Vector Machine (SVM)**: Effective for high-dimensional data
- **Gradient Boosting**: Advanced ensemble technique (XGBoost/LightGBM)
- **Neural Networks**: Deep learning approach for complex patterns

### Model Evaluation Metrics:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate for at-risk students
- **Recall**: Ability to identify all at-risk students
- **F1-Score**: Balanced measure of precision and recall
- **ROC-AUC**: Model's discriminative ability
- **Confusion Matrix**: Detailed classification results

## 📊 Key Features

### Interactive Web Application:
- **Student Risk Assessment**: Input student data for real-time risk prediction
- **Batch Prediction**: Upload CSV files for multiple student assessments
- **Visualization Dashboard**: Interactive charts and graphs
- **Model Comparison**: Compare performance across different algorithms
- **Feature Importance**: Understand which factors contribute most to risk

### Data Analysis Capabilities:
- **Exploratory Data Analysis**: Comprehensive data insights
- **Feature Engineering**: Create meaningful predictors from raw data
- **Missing Data Handling**: Robust preprocessing pipeline
- **Cross-Validation**: Reliable model performance estimation

## 📋 Usage Guide

### For Educators and Administrators:

1. **Individual Assessment**:
   - Navigate to the prediction interface
   - Input student information
   - Get immediate risk assessment and recommendations

2. **Batch Processing**:
   - Upload a CSV file with student data
   - Download results with risk scores and classifications
   - Use insights for intervention planning

3. **Data Insights**:
   - Explore feature importance to understand risk factors
   - View model performance metrics
   - Analyze prediction confidence levels

### For Developers:

1. **Model Training**:
   ```python
   from src.model_training import train_models
   from src.data_preprocessing import load_and_preprocess_data
   
   # Load and preprocess data
   X_train, X_test, y_train, y_test = load_and_preprocess_data('data/student_data.csv')
   
   # Train models
   models = train_models(X_train, y_train)
   ```

2. **Making Predictions**:
   ```python
   from src.prediction import predict_risk
   
   # Single prediction
   risk_score = predict_risk(model, student_data)
   
   # Batch predictions
   risk_scores = predict_batch(model, students_dataframe)
   ```

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make your changes** and add tests
4. **Commit your changes**: `git commit -m 'Add new feature'`
5. **Push to the branch**: `git push origin feature/new-feature`
6. **Submit a pull request**

### Areas for Contribution:
- Additional machine learning models
- Enhanced data visualization
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## 📝 Future Enhancements

- [ ] **Real-time Data Integration**: Connect with student information systems
- [ ] **Advanced Analytics**: Time-series analysis for trend prediction
- [ ] **Mobile Application**: Native mobile app development
- [ ] **Multi-language Support**: Internationalization features
- [ ] **Advanced Visualizations**: 3D plotting and interactive dashboards
- [ ] **Automated Reporting**: Scheduled reports for administrators
- [ ] **API Development**: RESTful API for system integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **[Tharushi](https://github.com/tharushi1019)** - Project Lead & Developer

## 🙏 Acknowledgments

- Educational institutions that provided anonymized data
- Open-source machine learning community
- Streamlit team for the excellent web framework
- Contributors and beta testers

## 📞 Contact & Support

- **GitHub Issues**: For bug reports and feature requests
- **Email**: [tharushinimnadee19@gmail.com](mailto:tharushinimnadee19@gmail.com) for direct inquiries
- **Documentation**: Check the notebooks for detailed analysis

## 📚 References

1. Academic research papers on student success prediction
2. Machine learning best practices for educational data
3. Ethical considerations in predictive modeling for education

---

**⭐ If you find this project helpful, please consider giving it a star on GitHub!**

*This project aims to support educational institutions in improving student outcomes through data-driven insights and early intervention strategies.*
