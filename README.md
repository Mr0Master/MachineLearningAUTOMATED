An automated machine learning pipeline that performs data preprocessing, applies multiple algorithms, evaluates their performance, and selects the best-performing model.

This project is designed to streamline the typical ML workflow by eliminating manual steps such as data cleaning, feature handling, model selection, and evaluation. It is especially useful for rapid experimentation and benchmarking across different datasets.

🚀 Features
🔄 Automated Data Cleaning
Handles missing values
Encodes categorical variables
Normalizes/scales features
📊 Multi-Algorithm Execution
Runs multiple ML algorithms (e.g., regression, classification models)
🧪 Performance Evaluation
Compares models using relevant metrics (accuracy, F1-score, RMSE, etc.)
🏆 Best Model Selection
Automatically identifies and outputs the top-performing algorithm
📁 File-Based Input
Accepts dataset files (CSV, etc.) for processing
⚡ Fully Automated Pipeline
Minimal user input required
🛠️ Tech Stack
Language: Python
Libraries:
pandas, numpy (data processing)
scikit-learn (ML models & evaluation)
(Optional: matplotlib / seaborn for visualization)
📂 How It Works
Load dataset from file
Automatically clean and preprocess data
Split into training and testing sets
Run multiple machine learning algorithms
Evaluate each model
Return the best-performing model with metrics
▶️ Usage
python main.py --file data.csv
📈 Example Output
Best Model: Random Forest Classifier
Accuracy: 91.2%
F1 Score: 0.89
🎯 Use Cases
Rapid ML prototyping
Model benchmarking
Educational purposes
Automating repetitive ML workflows
🔮 Future Improvements
Add deep learning models
Hyperparameter tuning (GridSearch / Optuna)
Web UI for easier interaction
Support for larger datasets & parallel execution
