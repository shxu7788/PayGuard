# PayGuard: Machine Learning-Based Fraud Detection for Bank Payments

## Overview

**PayGuard** is a machine learning-driven solution designed to detect fraudulent activities in bank payment systems. By leveraging advanced algorithms and data analysis techniques, PayGuard aims to enhance financial security by identifying suspicious transactions in real-time. This project integrates data preprocessing, exploratory data analysis (EDA), model building, and deployment into an interactive web application for seamless user experience.

## Features

- **Data Analysis & Visualization**
  - Comprehensive EDA using heatmaps, correlation matrices, histograms, and box plots.
  - Visualization of transaction distributions and identification of anomalies.
- **Machine Learning Models**
  - Implementation of various classification algorithms, including:
    - Random Forest Classifier
    - XGBoost Classifier
    - K-Nearest Neighbors (KNN) Classifier
  - Comparison of model performance metrics to select the optimal model.
- **Model Optimization**
  - Hyperparameter tuning to improve model accuracy, precision, recall, F1-score, and AUC-score.
  - Evaluation of models before and after tuning for performance enhancement.
- **Web Application Interface**
  - Integration of models into a Streamlit web app for user-friendly interaction.
  - Features include user authentication (signup and login), data uploading, EDA, model training, and fraud prediction.
- **Prediction and Deployment**
  - Real-time fraud prediction on new transaction data.
  - Visualization of prediction results and performance metrics.

## Project Structure

- **dashboard/app.py**: Streamlit application code handling the web interface, user authentication, data uploading, EDA, model building, and prediction functionalities.
- **notebook/code.ipynb**: Jupyter Notebook containing the detailed steps of data analysis, preprocessing, model training, evaluation, and visualization.
- **script/simulate.py**: Python script for simulating and testing the fraud detection models on synthetic data.
- **data**: Dataset for training and testing.

## Requirements

- **Programming Language**
  - Python 3.x
- **Python Libraries**
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - streamlit
  - xgboost
  - imbalanced-learn
- **Tools**
  - Jupyter Notebook or JupyterLab

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shxu7788/PayGuard.git
   ```
2. **Navigate to the Project Directory**
   ```bash
   cd PayGuard
   ```
3. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   ```
   - Activate the virtual environment:
     - On Windows:
       ```bash
       venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source venv/bin/activate
       ```
4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   - If `requirements.txt` is not provided, install dependencies manually:
     ```bash
     pip install pandas numpy scikit-learn matplotlib seaborn streamlit xgboost imbalanced-learn
     ```

## Usage

### Running the Web Application

1. **Start the Streamlit App**
   ```bash
   streamlit run dashboard/app.py
   ```
2. **Access the Web Interface**
   - Open your web browser and navigate to `http://localhost:8501`.
3. **Interact with the Application**
   - **Signup**: Create a new user account to access the app features.
   - **Login**: Access the application using your credentials.
   - **File Upload**: Upload transaction data for analysis (ensure the data is in the correct format).
   - **EDA**: Perform exploratory data analysis to understand data patterns and anomalies.
   - **Model Building**: Train machine learning models and compare their performance.
   - **Prediction**: Use the trained model to predict fraudulent transactions on new data.

### Running the Jupyter Notebook

1. **Open the Notebook**
   - Navigate to the `notebook` directory.
   - Open `code.ipynb` using Jupyter Notebook or JupyterLab.
2. **Execute the Cells**
   - Run the notebook cells sequentially to perform data analysis, model training, and evaluation.
   - Modify code and parameters as needed for experimentation.

## Project Highlights

- **Data Preprocessing**
  - Handling of imbalanced datasets using Synthetic Minority Over-sampling Technique (SMOTE).
  - Encoding of categorical variables and feature scaling for optimal model performance.
- **Model Evaluation Metrics**
  - Assessment using accuracy, precision, recall, F1-score, and ROC-AUC curves.
  - Visualization of model performance before and after hyperparameter tuning.
- **User-Friendly Interface**
  - Streamlit web app provides an accessible platform for users without programming knowledge.
  - Step-by-step guidance through data upload, analysis, and fraud prediction processes.

## Contribution

Contributions to PayGuard are welcome! If you have ideas for enhancements or encounter any issues, please open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is inspired by various open-source projects and research papers on fraud detection and machine learning.

## Contact

Thank you for exploring PayGuard. We hope this project helps in making financial transactions more secure and reliable. If you have any questions or need further assistance, feel free to reach out through the GitHub repository.

*Empowering financial security through machine learning.*
