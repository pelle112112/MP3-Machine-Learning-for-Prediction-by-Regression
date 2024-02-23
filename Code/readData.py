
# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as sm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Datapreparation

def loadData(path, fileType):
    match fileType:
        case "excel":
            return pd.read_excel(path, na_values=['NA'], skiprows=1)
        case "json":
            return pd.read_json(path)
        case "jsonl":
            return pd.read_json(path, lines=True)
        case "csv": 
            return pd.read_csv(path)
        
# Load the dataset
houseDf = loadData("../Data/House-data.csv", "csv")

# Checking for missing data
def checkMissingData(house):
    print(house.isnull().sum())
    
checkMissingData(houseDf)


#Explore the data

def typeOfData (houseData):
    print(houseData.shape)
    print(houseData.dtypes)
    houseDescription = houseData.describe()
    print(houseDescription)
    houseData.sample(5)
    
typeOfData(houseDf)

pd.set_option('display.float_format', lambda x: '%.3f' % x)


def plotCorrelationMatrix(data, title='Correlation Matrix'):
    numericData = data.select_dtypes(include=[np.number])
    correlationMatrix = numericData.corr()

#plot the heatmap

    plt.figure(figsize=(15,10))
    sns.heatmap(correlationMatrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.tight_layout()
    plt.title(title)
    plt.show()

plotCorrelationMatrix(houseDf, title='Correlation Matrix of All Attributes')

# Display the info with over 0.55 correlation to price

# Define colors for each label
colors = {'sqft_living': 'blue', 'grade': 'green', 'sqft_above': 'orange', 'sqft_living15': 'purple'}

# Scatter plot with different colors for each label
plt.scatter(houseDf['price'], houseDf['sqft_living'], color=colors['sqft_living'], label='sqft_living')
plt.scatter(houseDf['price'], houseDf['grade'], color=colors['grade'], label='grade')
plt.scatter(houseDf['price'], houseDf['sqft_above'], color=colors['sqft_above'], label='sqft_above')
plt.scatter(houseDf['price'], houseDf['sqft_living15'], color=colors['sqft_living15'], label='sqft_living15')

# Set labels and legend
plt.xlabel('price')
plt.ylabel('Feature Values')
plt.legend()

# Show the plot
plt.show()

#  distribution plot of the price vs grade

sns.displot(houseDf['price'], label = 'price', kde=True)
plt.legend()
plt.show()

# visualise and remove Outliers

def analyze_and_remove_outliers(data, feature='price', title='Outlier Analysis'):
    plt.figure(figsize=(15,10))
    sns.boxplot(x=data[feature])
    plt.title(f'Boxplot of {feature} - {title}')
    plt.show()

    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
    print(f"Number of outliers: '{feature}': {len(outliers)}")

    print("Descriptive Statistics before removing outliers:")

    data_cleaned = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

    print("\nDescriptive Statistics after removing outliers:")
    print(data_cleaned[feature].describe())

    return data_cleaned

cleanedDF = analyze_and_remove_outliers(houseDf, feature='price', title='Housing price analysis')



# Train the model with Linear Regression

X = cleanedDF['sqft_living'].values.reshape(-1,1)
Y = cleanedDF['price'].values.reshape(-1,1)
Z = cleanedDF['grade'].values.reshape(-1,1)

# display how grade influece the house pricing

plt.xlabel('grade')
plt.ylabel('price')
plt.scatter(Z, Y, color='red')
plt.show()

# display how sqft_living influece the house pricing
plt.ylabel('price')
plt.xlabel('sqft_living')
plt.scatter(X,Y, color='red')
plt.show()


# Split the data into training and testing sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=123)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# Create the model
myreg = LinearRegression()

# fit the model to our data
myreg.fit(X_train, Y_train)
myreg

# Get the calculated coefficients
a = myreg.coef_
b = myreg.intercept_

y_predicted = myreg.predict(X_test)

R2 = myreg.score(X,Y)

# Visualise the linear regression

plt.title('Linear Regression')
plt.scatter(X,Y, color='green')
plt.plot(X_train, a*X_train + b, color='blue')
plt.plot(X_test, y_predicted, color='orange')
plt.xlabel('Sqft')
plt.ylabel('price')
plt.text(0, -0.1, f'Score: {R2:.2f}', ha='left', va='center', transform=plt.gca().transAxes)
plt.show()



# Train the model with Multiple Linear Regression

#prepare the data

feature_cols = ['bathrooms','sqft_living','grade','sqft_above','sqft_living15']
Z = cleanedDF[feature_cols]
Z.head(10)

print(type(Z))
print(Z.shape)

Y = cleanedDF['price']
Y.head(10)

print(type(Y))
print(Y.shape)


#Splitting the data

Z_train,Z_test,Y_train,Y_test = train_test_split(Z,Y, random_state=1)

# default split 75:25
print(Z_train.shape)
print(Y_train.shape)
print(Z_test.shape)
print(Y_test.shape)

# create a model
linreg = LinearRegression()

# fit the model to our training data
linreg.fit(Z_train, Y_train)

# the intercept and coefficients are stored in system variables
print('b0 =', linreg.intercept_)
print('bi =', linreg.coef_)

# pair the feature names with the coefficients
list(zip(feature_cols, linreg.coef_))

Y_predicted = linreg.predict(Z_test)

print(sm.mean_absolute_error(Y_test, Y_predicted))

print(sm.mean_squared_error(Y_test, Y_predicted))

print(np.sqrt(sm.mean_squared_error(Y_test, Y_predicted)))

# Explained variance
eV = round(sm.explained_variance_score(Y_test, Y_predicted), 6)
print('Explained variance score ',eV )

# R-squared
K = r2_score(Y_test, Y_predicted)

# Visualise the regression results
plt.title('Multiple Linear Regression')
plt.scatter(Y_test, Y_predicted, color='blue')
plt.xlabel('best correlation att.')
plt.ylabel('price')
plt.text(0, -0.1, f'Score: {eV}', ha='left', va='center', transform=plt.gca().transAxes)
plt.show()


# Fitting a polynomial Regression to the dataset

poly_model = PolynomialFeatures(degree=4)
X_poly = poly_model.fit_transform(X)
pol_reg =LinearRegression()
pol_reg.fit(X_poly,Y)

Y_predict = pol_reg.predict(X_poly)

# Predicting a new result with Polymonial Regression

Ypredict = pol_reg.predict(X_poly)

eVPoly = r2_score(Y, Ypredict)

def vizPolynomialSmooth():
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    X_poly_grid = poly_model.transform(X_grid)
    plt.scatter(X, Y, color='red')
    plt.plot(X_grid, pol_reg.predict(X_poly_grid), color='blue')
    plt.title('Polynomial Regression')
    plt.xlabel('Sqft Living')
    plt.ylabel('Price')
    plt.text(0, -0.1, f'Score: {eVPoly}', ha='left', va='center', transform=plt.gca().transAxes)
    plt.show()

vizPolynomialSmooth()

