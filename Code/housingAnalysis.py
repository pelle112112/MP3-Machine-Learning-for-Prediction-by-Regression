import readData
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures


# Data wrangling
dataFrame = readData.loadData('../Data/house-data.csv', 'csv')
print(dataFrame)
print(dataFrame.isnull().sum())
print(dataFrame['sqft_living'].isnull().values.any())
# No missing data 

# Removing less important columns
dataFrame = dataFrame.drop([
    'id',
    'date',
    'sqft_living15',
    'sqft_lot15'
], axis=1)

print(dataFrame.info())
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(dataFrame.describe())


dataFrame.hist()
plt.tight_layout()
plt.show()
sns.heatmap(dataFrame.corr(), annot=True)
plt.tight_layout()
plt.show()

# Features with high correlation to price.
# sqft_living, bathrooms, grade, sqft_above
# Removing obsolete features
dataFrame = dataFrame.drop([
    'bedrooms',
    'sqft_lot',
    'floors',
    'waterfront',
    'condition',
    'sqft_basement',
    'yr_built',
    'yr_renovated',
    'zipcode',
    'lat',
    'long',
    'view'
], axis=1)

print(dataFrame.describe())

# Checking for outliers - standardising data, because value ranges are not comparable.
sc = StandardScaler()
standardizedDataFrame = sc.fit_transform(dataFrame)
standardizedDataFrame = pd.DataFrame(standardizedDataFrame, columns=dataFrame.columns)
standardizedDataFrame.plot(kind='box', figsize=(10, 10))
plt.tight_layout()
plt.show()

# Removing outliers in all features, using IQR method.
columns = list(dataFrame)
print(columns)
def removeOutliersFromColumn(dataFrame, columnName):
    Q1 = dataFrame[columnName].quantile(0.25)
    Q3 = dataFrame[columnName].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataFrame[columnName] = dataFrame[columnName][(dataFrame[columnName] >= lower_bound) & (dataFrame[columnName] <= upper_bound)]

for feature in columns:
    removeOutliersFromColumn(dataFrame, feature)

dataFrame = dataFrame.dropna()
# Standardising the data again to see if outliers have been removed
standardizedDataFrame = sc.fit_transform(dataFrame)
standardizedDataFrame = pd.DataFrame(standardizedDataFrame, columns=dataFrame.columns)
standardizedDataFrame.plot(kind='box', figsize=(10, 10))
plt.tight_layout()
plt.show()

# dataFrame = standardizedDataFrame - Was experimenting with running the rest of the code with standardised data - but it didn't change any results and only made the diagrams and predictions less readable.

# Trying the linear regression method:

# Splitting dataframe in dependent and independent data sets. For the independent set, I have chosen the feature with the highest correlation to price.
X = dataFrame['sqft_living'].values.reshape(-1, 1)
y = dataFrame['price'].values.reshape(-1, 1 )

# Showing datasets on scatter plot
plt.ylabel('price')
plt.xlabel('sqft_living')
plt.scatter(X, y)
plt.ticklabel_format(style='plain')
plt.tight_layout()
plt.show()

# By the looks of it - it seems that linear regression will always have a fairly high error rate.

# Splitting the data into train and test sets.
def trainTestSplit(x, y):

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=10, test_size=0.2)
    # Checking proportions of the data sets, to get an idea if they have been split correctly
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = trainTestSplit(X, y)

lineReg = LinearRegression()
lineReg.fit(X_train, y_train)

# Now i can retrieve the slope and intercept from my instance of LinearRegression
slope = lineReg.coef_
intercept = lineReg.intercept_

# Testing my model with the test data set:
y_predictions = lineReg.predict(X_test)

# The predictions can then be compared to our actual y_test dataset
print(y_predictions)
print(y_test)

# Visualising the linear regression
plt.title('Linear Regression')
plt.scatter(X, y, color='green')
plt.plot(X_train, slope*X_train + intercept, color='blue')
plt.plot(X_test, y_predictions, color='orange')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.ticklabel_format(style='plain')
plt.tight_layout()
plt.show()

# For evaluating the model i will take a look at explained variance score and R-squared score.
def evaluateModel(y_test, y_predictions):
    explainedVariance = round(sm.explained_variance_score(y_test, y_predictions), 3)
    print(f'Explained variance score is: {explainedVariance}')

    print(f'The R-squared score is: {r2_score(y_test, y_predictions)}')

evaluateModel(y_test, y_predictions)

# The scores here are somewhat low - meaning that this method might not be the best fit for our data, as there is a lot of variance not explained by the independent value. 

# Trying out the multilinear method:
X = dataFrame[['sqft_living', 'grade', 'sqft_above', 'bathrooms']]
y = dataFrame['price']

X_train, X_test, y_train, y_test = trainTestSplit(X, y)

multiLineReg = LinearRegression()
multiLineReg.fit(X_train, y_train)

slopes = multiLineReg.coef_
print(list(zip(['sqft_living', 'grade', 'sqft_above', 'bathrooms'], multiLineReg.coef_)))
intercept = multiLineReg.intercept_
print(intercept)

# Testing the model
y_predictions = multiLineReg.predict(X_test)
print(y_predictions)
print(y_test)

evaluateModel(y_test, y_predictions)

# Trying the polynominal method:
X = dataFrame['sqft_living']
y = dataFrame['price']

X_train, X_test, y_train, y_test = trainTestSplit(X, y)

poly_model = PolynomialFeatures()
X_polyTrain = poly_model.fit_transform(X_train.values.reshape(-1, 1))
X_polyTest = poly_model.fit_transform(X_test.values.reshape(-1, 1))
polynomialReg = LinearRegression()
polynomialReg.fit(X_polyTrain, y_train)

# Testing the model
y_predictions = polynomialReg.predict(X_polyTest)
print(y_predictions)
print(y_test)

#Visualising the results
plt.scatter(X, y, color='red')
plt.plot(X_test, y_predictions, color='blue')
plt.title('Polynimial Regression')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.ticklabel_format(style='plain')
plt.tight_layout()
plt.show()

evaluateModel(y_test, y_predictions)
