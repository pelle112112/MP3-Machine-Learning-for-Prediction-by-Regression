import readData
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare the data
dataFrame = readData.loadData('../Data/house-data.csv', 'csv')
print(dataFrame)
print(dataFrame.isnull().sum())
print(dataFrame['sqft_living'].isnull().values.any())
# Optional: Additional data cleaning and exploration steps here

# Feature engineering
dataFrame['year'] = pd.to_datetime(dataFrame['date']).dt.year
dataFrame['month'] = pd.to_datetime(dataFrame['date']).dt.month
dataFrame.drop(['id', 'date'], axis=1, inplace=True)

# Feature selection based on domain knowledge, previous analysis, or automated methods
selected_features = [
    'sqft_living',
    'grade',
    'sqft_above',
    'bathrooms',
    'bedrooms',
    'view',
    'floors',
    'waterfront',
    'condition',
    'year',
    'month',
    'sqft_lot15'
]

X = dataFrame[selected_features]
y = dataFrame['price']

# Data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Model fitting with Ridge regression
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)

# Predictions
y_pred = ridge_reg.predict(X_test)

# Evaluation
print(f'R-squared score: {r2_score(y_test, y_pred)}')

# Visualizing the results (Optional)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
