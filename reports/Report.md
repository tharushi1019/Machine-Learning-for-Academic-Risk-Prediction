# Student Academic Risk Prediction System

## Overview

This Streamlit application predicts student academic risk (grade classification) using a trained machine learning pipeline. It provides single and batch predictions, visualizes model confidence, and offers feature importance analysis. The app also includes synthetic data visualizations and simulated model performance metrics.

---

## Structure

- **Imports:** Data science, ML, visualization, and Streamlit libraries.
- **Styling:** Custom CSS for headers, cards, and prediction results.
- **Model Loading:** Loads pipeline and label classes from `artifacts/`.
- **Sample Data:** Generates synthetic data for demo and visualization.
- **Navigation:** Sidebar for page selection and dataset info.
- **Pages:**
  - Home - Prediction (single & batch)
  - Data Overview
  - Advanced Visualizations
  - Model Performance
  - Feature Analysis
  - About

---

## Key Features

### 1. Model Loading

- Uses `joblib` to load the trained pipeline.
- Loads `label_classes.json` for grade mapping.
- Handles missing artifacts gracefully.

### 2. Prediction (Home Page)

- **Single Prediction:** Form for user input, maps numeric prediction to letter grade using `label_classes.json`.
- **Batch Prediction:** CSV upload, processes categorical mappings, outputs predicted grades.
- **Confidence Visualization:** Bar chart of prediction probabilities.
- **Feature Importance:** Attempts to show feature importances if available.

### 3. Data Visualization

- Synthetic data used for histograms, pie charts, scatter plots, and correlation heatmaps.
- Interactive filters for exploring data.

### 4. Model Performance

- Simulated metrics (accuracy, precision, recall).
- Confusion matrix and classification report.
- Demographic performance analysis.

### 5. Feature Analysis

- Simulated global feature importances.
- Impact analysis for study time and absences.
- Feature interaction heatmap.

### 6. About Page

- Explains app purpose, dataset, technical stack, and ethical considerations.

---

## Issues

### 1. Feature Importance Analysis

- The code robustly checks for named steps in the pipeline, but if the pipeline structure changes, feature importance extraction may fail. Consider documenting expected pipeline structure.

### 2. Label Mapping

- The mapping from numeric prediction to letter grade is now correct, assuming `label_classes.json` is a list of grade labels (`["A", "B", "C", "D", "F"]`).

### 3. Error Handling

- Use of try/except for model loading and prediction mapping.
- Batch prediction mapping is robust.
