# 🩺 Diabetes Prediction using Decision Tree

This project involves data analysis and prediction of diabetes using a **Decision Tree Classifier** on the **Pima Indians Diabetes Dataset**. It includes data visualization, data cleaning (checking for duplicates), and model training using **scikit-learn**.

---

## 📁 Dataset

- **File**: `diabetes.csv`
- **Source**: Contains diagnostic measurements for Pima Indian women, such as:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (0 = Non-diabetic, 1 = Diabetic)

---

## 🛠️ Tools and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📊 Exploratory Data Analysis

- Head of data: `data.head()`
- Info and statistics: `data.info()`, `data.describe()`
- Duplicate check: Identified and printed duplicate rows

### Visualizations

- ✅ **Scatter Plot**: Age vs BMI
- ✅ **Line Plot**: BloodPressure
- ✅ **Histogram**: BloodPressure distribution
- ✅ **Bar Chart**: Age vs BloodPressure
- ❌ **Pie Chart**: *(Not suitable for continuous numerical data)*
- ✅ **Seaborn Lineplot**: Age vs BloodPressure
- ✅ **Seaborn Scatterplot**: Age vs BMI
- ✅ **Seaborn Distplot**: Insulin distribution

---

## 🧠 Model Building

- **Feature Selection**: Used all columns except `Outcome` as features
- **Train-Test Split**: 80% training, 20% testing
- **Model**: `DecisionTreeClassifier`

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Splitting the data
x = data.iloc[:, :-1]
y = data['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Model training, prediction, and evaluation
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
ypred = model.predict(x_test)
accuracy = accuracy_score(y_test, ypred)
print("Accuracy:", accuracy)
```

## ✅ Output

The model predicts whether a person has diabetes or not based on the given medical attributes.

 
