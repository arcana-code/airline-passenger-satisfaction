# airline-passenger-satisfaction
“Machine learning project to predict airline passenger satisfaction using Random Forest.”
Airline Passenger Satisfaction Prediction
Overview
This project develops a Random Forest Classifier to predict airline passenger satisfaction based on survey data. The model classifies passengers as satisfied or neutral or dissatisfied, achieving 91.32% test accuracy on a dataset of 103,904 survey responses. The project demonstrates skills in data preprocessing, machine learning, and visualization using Python libraries.
Dataset

Source: Airline Passenger Satisfaction dataset (not included in the repository due to size; download from Kaggle).
Size: 103,904 records (50% training, 50% test split: 51,952 samples each).
Features: 23 attributes, including:
Demographic: Gender, Age, Customer Type (Loyal/Disloyal).
Travel: Type of Travel (Business/Personal), Class (Business/Eco/Eco Plus), Flight Distance.
Services: Ratings (0–5) for Inflight wifi service, Seat comfort, On-board service, etc.
Delays: Departure Delay in Minutes, Arrival Delay in Minutes.


Target: satisfaction (binary: satisfied or neutral or dissatisfied).
Note: The dataset has 310 missing values in Arrival Delay in Minutes, handled via median imputation.

Methodology

Preprocessing:
Handled missing values in Arrival Delay in Minutes using median imputation.
Encoded categorical variables (Gender, Customer Type, Type of Travel, Class) using one-hot encoding.
Scaled numerical features with StandardScaler from scikit-learn.
Dropped irrelevant columns (Unnamed: 0, id).


Model: Random Forest Classifier (n_estimators=100, max_depth=10, random_state=42).
Evaluation: 
Split data 50/50 for training and testing.
Achieved 91.32% accuracy on the test set.
Visualized results with a confusion matrix using Seaborn and Matplotlib.


Tools: Python, pandas, scikit-learn, Matplotlib, Seaborn.

Files

Airline_Passenger_Satisfaction.ipynb: Jupyter notebook containing the full code, including data preprocessing, model training, evaluation, and visualizations.
.gitignore: Excludes large datasets and temporary files (e.g., .ipynb_checkpoints, *.csv).
(Dataset not included; download from Kaggle link above.)

Results

Test Accuracy: 91.32%
Confusion Matrix:[[27889,  1515],
 [ 2995, 19553]]


True Negatives (neutral or dissatisfied, correctly predicted): 27,889
False Positives: 1,515
False Negatives: 2,995
True Positives (satisfied, correctly predicted): 19,553


Visualization: Confusion matrix heatmap generated with Seaborn.

Requirements
To run the notebook, install the required Python libraries:
pip install numpy pandas scikit-learn matplotlib seaborn



Download the dataset: Obtain airline_passenger_satisfaction.csv from Kaggle and place it in the project folder.
Install dependencies:pip install -r requirements.txt

(Alternatively, install libraries listed above.)
Run the notebook:
Open Airline_Passenger_Satisfaction.ipynb in Jupyter Notebook or JupyterLab.
Ensure the dataset is in the same directory or update the file path in the notebook.
Run all cells to preprocess data, train the model, and view results.



Future Improvements

Perform hyperparameter tuning using GridSearchCV to optimize Random Forest parameters.
Conduct feature importance analysis to identify key drivers of passenger satisfaction.
Experiment with other models (e.g., XGBoost, Gradient Boosting) for comparison.
Add cross-validation to assess model robustness.
Include additional visualizations, such as feature importance plots or satisfaction distribution charts.

