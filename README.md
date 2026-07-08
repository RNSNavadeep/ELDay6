Iris Flower Classification — K-Nearest Neighbors (KNN)

An end-to-end Machine Learning classification project that predicts the species of Iris flowers using the K-Nearest Neighbors (KNN) algorithm, built on the classic Iris dataset.


📁 Dataset


File: Iris.csv
Target Column: Species (Setosa, Versicolor, Virginica)
Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
Shape: 150 rows × 6 columns



🔄 Workflow

1. 🔍 Data Exploration

Loaded the dataset and performed initial checks:


Displayed first 20 rows to understand structure
Checked data types and info
Checked for null values and duplicate rows
Printed unique species in the target column
Checked shape of the dataset


2. 🏷️ Label Encoding

Converted the Species column from text to numbers using LabelEncoder:


Iris-setosa → 0
Iris-versicolor → 1
Iris-virginica → 2


pythonle = LabelEncoder()
df["Species"] = le.fit_transform(df[["Species"]])

3. 📊 Data Visualization

Visualized feature distributions before scaling:


Boxplot — to understand spread and detect outliers across all features
Histogram — to understand the distribution of each feature


4. ⚖️ Feature Scaling

Applied StandardScaler to normalize all features — KNN is distance-based, so scaling is essential to prevent features with larger ranges from dominating:

pythonss = StandardScaler()
x_scaled = ss.fit_transform(x)

5. ✂️ Train-Test Split

Split data into 80% training and 20% testing:

pythonxtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=42)

6. 🔢 Optimal K Selection

Instead of guessing K, calculated it mathematically:

pythonk = math.floor(math.sqrt(len(x)))  # Square root of total samples
if k % 2 == 0:
    k = k + 1  # Make it odd to avoid ties


Using an odd K avoids tie-breaking issues in classification.



7. 🤖 Model Training

Trained a KNN Classifier with the calculated optimal K:

pythonmodel = KNeighborsClassifier(n_neighbors=k)
model.fit(xtr, ytr)

8. 📊 Model Evaluation

Evaluated using three key classification metrics:

MetricDescriptionAccuracy ScoreOverall correct predictions out of totalClassification ReportPrecision, Recall, F1-score per classConfusion MatrixTable showing TP, TN, FP, FN for each class

9. 📉 Validation Curve

Used Yellowbrick's ValidationCurve to find the best value of K by plotting accuracy across K values from 1 to 25 with 5-fold cross validation:

pythonvalues = np.arange(1, 25)
visual = ValidationCurve(model, param_name='n_neighbors',
                         param_range=values, scoring='accuracy', cv=5)
visual.fit(xtr, ytr)
visual.show()

This helps visualize underfitting vs overfitting across different K values.


🛠️ Libraries Used

LibraryPurposepandasData loading and manipulationnumpyNumerical operationsmatplotlibBoxplot and histogramseabornStatistical visualizationsscikit-learnEncoding, scaling, model training, evaluationyellowbrickValidation curve visualizationmathSquare root calculation for optimal K


🧠 What is KNN?

K-Nearest Neighbors is a simple, distance-based classification algorithm. For a new data point, it finds the K closest points in the training data and assigns the most common class among them.

New flower → Find K nearest flowers → Majority species = Prediction


Small K → flexible but noisy (overfitting)
Large K → smooth but may miss patterns (underfitting)
Best K → found using the validation curve



✅ Conclusion

This project demonstrates KNN classification on the Iris dataset — from raw data to a tuned model. The smart K selection using square root and the validation curve together ensure the model is neither overfitting nor underfitting, giving reliable species predictions.
