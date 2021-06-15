########################### problem 1 #####################
import pandas as pd
#loading the dataset
startup = pd.read_csv("C:/Users/usach/Desktop/Multi Linear Regression/50_Startups.csv")

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image

#######feature of the dataset to create a data dictionary
description  = ["Money spend on research and development",
                "Administration",
                "Money spend on Marketing",
                "Name of state",
                "Company profit"]

d_types =["Ratio","Ratio","Ratio","Nominal","Ratio"]

data_details =pd.DataFrame({"column name":startup.columns,
                            "data types ":d_types,
                            "description":description,
                            "data type(in Python)": startup.dtypes})

#3.	Data Pre-startupcessing
#3.1 Data Cleaning, Feature Engineering, etc
#details of startup 
startup.info()
startup.describe()          
#rename the columns
startup.rename(columns = {'R&D Spend':'rd_spend', 'Marketing Spend' : 'm_spend'} , inplace = True)  
#data types        
startup.dtypes
#checking for na value
startup.isna().sum()
startup.isnull().sum()
#checking unique value for each columns
startup.nunique()
"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    
EDA ={"column ": startup.columns,
      "mean": startup.mean(),
      "median":startup.median(),
      "mode":startup.mode(),
      "standard deviation": startup.std(),
      "variance":startup.var(),
      "skewness":startup.skew(),
      "kurtosis":startup.kurt()}

EDA
# covariance for data set 
covariance = startup.cov()
covariance

# Correlation matrix 
co = startup.corr()
co

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.
####### graphistartup repersentation 
##historgam and scatter plot
import seaborn as sns
sns.pairplot(startup.iloc[:, :])

#boxplot for every columns
startup.columns
startup.nunique()

startup.boxplot(column=['rd_spend', 'Administration', 'm_spend', 'Profit'])   #no outlier

# here we can see lVO For profit
# Detection of outliers (find limits for RM based on IQR)
IQR = startup['Profit'].quantile(0.75) - startup['Profit'].quantile(0.25)
lower_limit = startup['Profit'].quantile(0.25) - (IQR * 1.5)

####################### 2.Replace ############################
# Now let's replace the outliers by the maximum and minimum limit
#Graphical Representation
import numpy as np
import matplotlib.pyplot as plt # mostly used for visualization purposes 

startup['Profit']= pd.DataFrame( np.where(startup['Profit'] < lower_limit, lower_limit, startup['Profit']))

import seaborn as sns 
sns.boxplot(startup.Profit);plt.title('Boxplot');plt.show()

# rd_spend
plt.bar(height = startup.rd_spend, x = np.arange(1, 51, 1))
plt.hist(startup.rd_spend) #histogram
plt.boxplot(startup.rd_spend) #boxplot

# Administration
plt.bar(height = startup.Administration, x = np.arange(1, 51, 1))
plt.hist(startup.Administration) #histogram
plt.boxplot(startup.Administration) #boxplot

# m_spend
plt.bar(height = startup.m_spend, x = np.arange(1, 51, 1))
plt.hist(startup.m_spend) #histogram
plt.boxplot(startup.m_spend) #boxplot

#profit
plt.bar(height = startup.Profit, x = np.arange(1, 51, 1))
plt.hist(startup.Profit) #histogram
plt.boxplot(startup.Profit) #boxplot

# Jointplot

sns.jointplot(x=startup['Profit'], y=startup['rd_spend'])

# Q-Q Plot
from scipy import stats
import pylab

stats.probplot(startup.Profit, dist = "norm", plot = pylab)
plt.show() 
# startupfit is normally distributed

stats.probplot(startup.Administration, dist = "norm", plot = pylab)
plt.show() 
# administration is normally distributed

stats.probplot(startup.rd_spend, dist = "norm", plot = pylab)
plt.show() 

stats.probplot(startup.m_spend, dist = "norm", plot = pylab)
plt.show() 
#normal
# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(startup.iloc[:,[0,1,2]])
df_norm.describe()

"""
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
sta=startup.iloc[:,[3]]
enc_df = pd.DataFrame(enc.fit_transform(sta).toarray())"""

# Create dummy variables on categorcal columns

enc_df = pd.get_dummies(startup.iloc[:,[3]])
enc_df.columns
enc_df.rename(columns={"State_New York":'State_New_York'},inplace= True)

model_df = pd.concat([enc_df, df_norm, startup.iloc[:,4]], axis =1)

#rename the columns

"""5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform Multi linear regression model and check for VIF, AvPlots, Influence Index Plots.
5.3	Train and Test the data and compare RMSE values tabulate R-Squared values , RMSE for different models in documentation and model_dfvide your explanation on it.
5.4	Briefly explain the model output in the documentation. 
"""

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model

ml1 = smf.ols('Profit ~ State_California+ State_Florida+ State_New_York+ rd_spend + Administration + m_spend ', data = model_df).fit() # regression model

# Summary
ml1.summary2()
ml1.summary()
# p-values for State, Administration are more th no correlation of model_dffit with State and Administrationan 0.05
# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm 

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 49 is showing high influence so we can exclude that entire row

model_df_new = model_df.drop(model_df.index[[49]])

# Preparing model                  
ml_new = smf.ols('Profit ~State_California+ State_Florida+ State_New_York+ rd_spend + Administration + m_spend ', data = model_df_new).fit()    

# Summary
ml_new.summary()
ml_new.summary2()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_rd_spend = smf.ols('rd_spend ~ Administration + m_spend + State_California+ State_Florida+ State_New_York', data = model_df).fit().rsquared  
vif_rd_spend = 1/(1 - rsq_rd_spend) 

rsq_admini = smf.ols(' Administration ~ rd_spend + m_spend + State_California+ State_Florida+ State_New_York ', data = model_df).fit().rsquared  
vif_admini = 1/(1 - rsq_admini)
ml_ad=smf.ols(' Administration ~ rd_spend + m_spend + State_California+ State_Florida+ State_New_York ', data = model_df).fit()
ml_ad.summary() 

rsq_m_spend = smf.ols(' m_spend ~ rd_spend + Administration  + State_California+ State_Florida+ State_New_York', data = model_df).fit().rsquared  
vif_m_spend = 1/(1 - rsq_m_spend) 

rsq_state = smf.ols(' State_California ~ rd_spend + Administration + m_spend  ', data = model_df).fit().rsquared  
vif_state = 1/(1 - rsq_state) 

ml_S= smf.ols(' State_California~ rd_spend + Administration + m_spend  ', data = model_df).fit()
ml_S.summary()

# Storing vif values in a data frame
d1 = {'Variables':['rd_spend' ,'Administration' ,'m_spend ',' State '], 'VIF':[vif_rd_spend, vif_admini, vif_m_spend, vif_state]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

#vif is low 
#model 2 without Administration  column because p value of state >>> 0.5

ml2 = smf.ols('Profit ~ rd_spend + State_California+ State_Florida+ State_New_York + m_spend  ', data = model_df).fit() # regression model
# Summary
ml2.summary()

# administration p value is high 
sm.graphics.influence_plot(ml2)

# Studentized Residuals = Residual/standard deviation of residuals
#model 3 without Administration  and state column because p value of state >>> 0.5
ml3 = smf.ols('Profit ~ rd_spend + m_spend ', data = model_df_new).fit() # regression model

# Summary
ml3.summary()

sm.graphics.influence_plot(ml3)

# Final model
final_ml = smf.ols('Profit ~ rd_spend + m_spend + State_California+ State_Florida+ State_New_York  ', data = model_df).fit() 
final_ml.summary() 

# Prediction
pred = final_ml.predict(model_df)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = model_df.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)
### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
model_df_train, model_df_test = train_test_split(model_df, test_size = 0.2,random_state = 457) # 20% test data

# preparing the model on train data 
model_train = smf.ols('Profit ~ rd_spend + m_spend  + State_California+ State_Florida+ State_New_York  ', data = model_df_train).fit()
model_train.summary()
model_train.summary2()
# prediction on test data set 
test_pred = model_train.predict(model_df_test)

# test residual values 
test_resid = test_pred - model_df_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse
# train_data prediction
train_pred = model_train.predict(model_df_train)

# train residual values 
train_resid  = train_pred - model_df_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

####################### problem 2 #################################
import pandas as pd

#loading the dataset
computer = pd.read_csv("C:/Users/usach/Desktop/Multi Linear Regression/Computer_Data.csv")
#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary
description  = ["Index row number (irrelevant ,does not provide useful Informatiom)",
                "Price of computer(relevant provide useful Informatiom)",
                "computer speed (relevant provide useful Informatiom)",
                "Hard Disk space of computer (relevant provide useful Informatiom)",
                "Random axis momery of computer (relevant provide useful Informatiom)",
                "Screen size of Computer (relevant provide useful Informatiom)",
                "Compact dist (relevant provide useful Informatiom)",
                "Multipurpose use or not (relevant provide useful Informatiom)",
                "Premium Class of computer (relevant provide useful Informatiom)",
                "advertisement expenses (relevant provide useful Informatiom)",
                "Trend position in market (relevant provide useful Informatiom)"]

d_types =["Count","Ratio","Ratio","Ratio","Ratio","Ratio","Binary","Binary","Binary","Ratio","Ratio"]

data_details =pd.DataFrame({"column name":computer.columns,
                            "data types ":d_types,
                            "description":description,
                            "data type(in Python)": computer.dtypes})

#3.	Data Pre-computercessing
#3.1 Data Cleaning, Feature Engineering, etc
#details of computer 
computer.info()
computer.describe()          

#droping index colunms 
computer.drop(['Unnamed: 0'], axis = 1, inplace = True)
#dummy variable creation 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
computer['cd'] = LE.fit_transform(computer['cd'])
computer['multi'] = LE.fit_transform(computer['multi'])
computer['premium'] = LE.fit_transform(computer['premium'])

#data types        
computer.dtypes

#checking for na value
computer.isna().sum()
computer.isnull().sum()

#checking unique value for each columns
computer.nunique()
"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    
EDA ={"column ": computer.columns,
      "mean": computer.mean(),
      "median":computer.median(),
      "mode":computer.mode(),
      "standard deviation": computer.std(),
      "variance":computer.var(),
      "skewness":computer.skew(),
      "kurtosis":computer.kurt()}

EDA
# covariance for data set 
covariance = computer.cov()
covariance

# Correlation matrix 
co = computer.corr()
co

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.
####### graphicomputer repersentation 
##historgam and scatter plot
import seaborn as sns
sns.pairplot(computer.iloc[:, :])

#boxplot for every columns
computer.columns
computer.nunique()

computer.boxplot(column=['price','ads', 'trend'])   #no outlier

#for imputing HVO for Price column
"""
# here we can see lVO For Price
# Detection of outliers (find limits for RM based on IQR)
IQR = computer['Price'].quantile(0.75) - computer['Price'].quantile(0.25)
upper_limit = computer['Price'].quantile(0.75) + (IQR * 1.5)
####################### 2.Replace ############################
# Now let's replace the outliers by the maximum and minimum limit
#Graphical Representation
import numpy as np
import matplotlib.pyplot as plt # mostly used for visualization purposes 
computer['Price']= pd.DataFrame( np.where(computer['Price'] > upper_limit, upper_limit, computer['Price']))
import seaborn as sns 
sns.boxplot(computer.Price);plt.title('Boxplot');plt.show()"""

# Q-Q Plot
from scipy import stats
import pylab
import matplotlib.pyplot as plt

stats.probplot(computer.price, dist = "norm", plot = pylab)
plt.show() 

stats.probplot(computer.ads, dist = "norm", plot = pylab)
plt.show() 

stats.probplot(computer.trend, dist = "norm", plot = pylab)
plt.show() 

#normal

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(computer.iloc[:,[1,2,3,4,8,9]])
df_norm.describe()
"""
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
sta=computer.iloc[:,[3]]
enc_df = pd.DataFrame(enc.fit_transform(sta).toarray())"""

# Create dummy variables on categorcal columns

enc_df = computer.iloc[:,[5,6,7]]

model_df = pd.concat([enc_df, df_norm,computer.iloc[:,[0]] ], axis =1)

#rename the columns

"""5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform Multi linear regression model and check for VIF, AvPlots, Influence Index Plots.
5.3	Train and Test the data and compare RMSE values tabulate R-Squared values , RMSE for different models in documentation and model_dfvide your explanation on it.
5.4	Briefly explain the model output in the documentation. 
"""
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model           
ml1 = smf.ols('price ~ speed + hd + ram + screen +   cd + multi + premium + ads + trend', data = model_df).fit() # regression model

# Summary
ml1.summary()

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables

rsq_hd = smf.ols('hd ~ speed  + ram + screen +   cd + multi + premium + ads + trend', data = model_df).fit().rsquared  
vif_hd = 1/(1 - rsq_hd) 
vif_hd # vif is low 

rsq_ram = smf.ols('ram ~ speed  + hd + screen +   cd + multi + premium + ads + trend', data = model_df).fit().rsquared  
vif_ram = 1/(1 - rsq_ram) 
vif_ram # vif is low 

 # by r squared value
mlhd = smf.ols('hd ~ speed  + ram + screen +   cd + multi + premium + ads + trend', data = model_df).fit()
mlhd.summary()

#model 2 
ml2 = smf.ols('price ~ speed + ram + screen +   cd + multi + premium + ads + trend', data = model_df).fit() # regression model

# Summary
ml2.summary()

#model 3 
ml3 = smf.ols('price ~ speed + hd + screen +   cd + multi + premium + ads + trend', data = model_df).fit() # regression model
# Summary
ml3.summary()

# Final model
final_ml = smf.ols('price ~ speed + hd + ram + screen +   cd + multi + premium + ads + trend', data = model_df).fit()
final_ml.summary() 
final_ml.summary2()
# Prediction
pred = final_ml.predict(model_df)
pred
from scipy import stats
import pylab
import statsmodels.api as sm
# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Jointplot
import seaborn as sns
# Residuals vs Fitted plot
sns.residplot(x = pred, y = model_df.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)
### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
train, test = train_test_split(model_df, test_size = 0.2,random_state = 7) # 20% test data

# preparing the model on train data 
model_train = smf.ols('price ~ speed + hd + ram + screen +   cd + multi + premium + ads + trend', data = train).fit()

# prediction on test data set 
test_pred = model_train.predict(test)

# test residual values 
test_resid = test_pred - test.price
import numpy as np
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(train)

# train residual values 
train_resid  = train_pred - train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
############################# problem 3 #############################
import pandas as pd
#loading the dataset
toyo = pd.read_csv("C:/Users/usach/Desktop/Multi Linear Regression/ToyotaCorolla.csv", encoding ="latin1")

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

data_details =pd.DataFrame({"column name":toyo.columns,
                            "data type(in Python)": toyo.dtypes})

#3.	Data Pre-toyocessing
#3.1 Data Cleaning, Feature Engineering, etc
#details of toyo 
toyo.info()
toyo.describe()          

#droping index colunms 
toyo.drop(["Id"], axis = 1, inplace = True)
#dummy variable creation 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
toyo['Age_08_04'] = LE.fit_transform(toyo['Age_08_04'])
toyo['HP'] = LE.fit_transform(toyo['HP'])
toyo['cc'] = LE.fit_transform(toyo['cc'])
toyo['Doors'] = LE.fit_transform(toyo['Doors'])
toyo['Gears'] = LE.fit_transform(toyo['Gears'])

df= toyo[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

#data types        
df.dtypes
#checking for na value
df.isna().sum()
df.isnull().sum()

#checking unique value for each columns
df.nunique()
"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """

EDA ={"column ": df.columns,
      "mean": df.mean(),
      "median":df.median(),
      "mode":df.mode(),
      "standard deviation": df.std(),
      "variance":df.var(),
      "skewness":df.skew(),
      "kurtosis":df.kurt()}

EDA
# covariance for data set 
covariance = df.cov()
covariance
# Correlation matrix 
co = df.corr()
co
# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.
####### graphidf repersentation 
##historgam and scatter plot
import seaborn as sns
sns.pairplot(df.iloc[:, :])
#boxplot for every columns
df.columns
df.nunique()

df.boxplot(column=['Price', 'Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears','Quarterly_Tax', 'Weight'])   #no outlier

#normal
# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(df.iloc[:,1:9])
df.describe()
"""
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
sta=df.iloc[:,[3]]
enc_df = pd.DataFrame(enc.fit_transform(sta).toarray())"""

# Create dummy variables on categorcal columns

model_df = pd.concat([df,toyo.iloc[:,[1]] ], axis =1)

#rename the columns
"""5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform Multi linear regression model and check for VIF, AvPlots, Influence Index Plots.
5.3	Train and Test the data and compare RMSE values tabulate R-Squared values , RMSE for different models in documentation and model_dfvide your explanation on it.
5.4	Briefly explain the model output in the documentation. 
"""

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model

ml1 = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = model_df).fit() # regression model

# Summary
ml1.summary()

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)

model_df_new = model_df.drop(model_df.index[[960,221]])
#droping row num 960,221 due to outlire

# Preparing model                  
ml_new = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = model_df_new).fit()    

# Summary
ml_new.summary()

# Prediction
pred = ml_new.predict(model_df)
# removing outlier 
# Final model
final_ml = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = model_df_new).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(model_df)

from scipy import stats
import pylab
import statsmodels.api as sm
import matplotlib.pyplot as plt
# Q-Q plot  residuals
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Jointplot
import seaborn as sns
# Residuals vs Fitted plot
sns.residplot(x = pred, y = model_df.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
pro_train, pro_test = train_test_split(model_df_new, test_size = 0.2,random_state = 77) # 20% test data

# preparing the model on train data 
model_train = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = pro_train).fit()
model_train.summary()
# prediction on test data set 
test_pred = model_train.predict(pro_test)

import numpy as np
# test residual values 
test_resid = test_pred - pro_test.Price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse
# train_data prediction
train_pred = model_train.predict(pro_train)
train_pred
# train residual values 
train_resid  = train_pred - pro_train.Price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

########################## problem 4 ################################
import pandas as pd
import numpy as np

# loading the data
Avocado_Price = pd.read_csv("C:/Users/usach/Desktop/Multi Linear Regression/Avacado_Price.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (HistogTOT_AVA2, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

Avocado_Price.drop('type', inplace=True, axis=1)
Avocado_Price.drop('year', inplace=True, axis=1)
Avocado_Price.drop('region', inplace=True, axis=1)

Avocado_Price.rename(columns = {'XLarge Bags': 'XLarge_Bags'}, inplace = True)

Avocado_Price.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# Total_Volume
plt.bar(height = Avocado_Price.Total_Volume, x = np.arange(1, 18249, 1))
plt.hist(Avocado_Price.Total_Volume) #histog
plt.boxplot(Avocado_Price.Total_Volume) #boxplot

# AVERAGEPRICE
plt.bar(height = Avocado_Price.AveragePrice, x = np.arange(1, 18249, 1))
plt.hist(Avocado_Price.AveragePrice) #histogTOT_AVA2
plt.boxplot(Avocado_Price.AveragePrice) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=Avocado_Price['Total_Volume'], y=Avocado_Price['AveragePrice'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(Avocado_Price['Total_Volume'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(Avocado_Price.AveragePrice, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histogTOT_AVA2s
import seaborn as sns
sns.pairplot(Avocado_Price.iloc[:, :])
                             
# Correlation matrix 
Avocado_Price.corr()

# we see there exists High collinearity between input variables especially between
# [TOT_AVA2 & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index  1043 is showing high influence so we can exclude that entire row
 
Avocado_Price_new = Avocado_Price.drop(Avocado_Price.index[[11271]])

# Preparing model                  
ml_new = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_tt = smf.ols('Total_Volume ~ tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_tt= 1/(1 - rsq_tt) 

rsq_tot_ava1 = smf.ols('tot_ava1 ~ Total_Volume + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_tot_ava1 = 1/(1 - rsq_tot_ava1)

rsq_tot_ava2 = smf.ols('tot_ava2 ~ Total_Volume + tot_ava1 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_tot_ava2 = 1/(1 - rsq_tot_ava2) 

rsq_tot_ava3 = smf.ols('tot_ava3 ~ Total_Volume + tot_ava1 + tot_ava2 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_tot_ava3 = 1/(1 - rsq_tot_ava3) 

rsq_tb = smf.ols('Total_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Small_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_tb = 1/(1 - rsq_tb) 

rsq_sb = smf.ols('Small_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_sb = 1/(1 - rsq_sb) 

rsq_lb = smf.ols('Large_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Small_Bags + Total_Bags + + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_lb = 1/(1 - rsq_lb) 

rsq_xl = smf.ols('XLarge_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Small_Bags + Total_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_xl = 1/(1 - rsq_xl) 

# Storing vif values in a data fTOT_AVA2e
d1 = {'Variables':['Total_Volume', 'tot_ava1 + tot_ava2 ', 'tot_ava3', 'Total_Bags', 'Small_Bags', 'Large_Bags', 'XLarge_Bags'], 'VIF':[vif_tt, vif_tot_ava1, vif_tot_ava2, vif_tot_ava3, vif_tb, vif_sb, vif_lb, vif_xl]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As Large_Bags is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('AveragePrice ~ Total_Volume + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + XLarge_Bags', data = Avocado_Price).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(Avocado_Price)
pred

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = Avocado_Price.AveragePrice, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
Avocado_Price_train, Avocado_Price_test = train_test_split(Avocado_Price, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags", data = Avocado_Price_train).fit()

# prediction on test data set 
test_pred = model_train.predict(Avocado_Price_test)

# test residual values 
test_resid = test_pred - Avocado_Price_test.AveragePrice

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(Avocado_Price_train)

# train residual values 
train_resid  = train_pred - Avocado_Price_train.AveragePrice
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
