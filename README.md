## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
NAME: BHARATHGANESH S
REG NO: 212222230022
```
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/9aa60aa9-2074-4ae0-a031-c27f3ec86ec3)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=["Hot","Warm",'Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/b96e02b1-0e30-4e4b-91a5-0674c915f90e)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/a3e49a68-bca1-4cfb-9d3b-1832328b2d08)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/14a55098-160c-467b-a753-3b2b02e954ca)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/de93026a-defb-4304-b5b8-d945fcee035c)
```
pip install category_encoders
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/85c03a65-c2c7-45a3-9480-533d770491de)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```

![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/d47a69ad-1bc3-4391-b24d-5cbdcc78b879)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/04b54263-8607-4478-9a54-2f72a17497ff)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/812561c6-4171-4c9c-bd7b-93043f9c9a1f)
```
df.skew()
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/baa4f8e6-81e8-4069-93de-319cb57f4a54)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/33a4aff1-1af1-4262-aa4d-bb84a36597af)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/b2606b02-8236-4257-95b4-509157cd8676)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/b68b729b-147c-4356-8b50-a20d31f9cfb1)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/55e652d8-f688-4524-9497-a562c43c8264)
```
import matplotlib.pyplot as plt 
import seaborn as sns 
import statsmodels.api as sm 
import scipy.stats as stats
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/f8d40849-d011-4855-9ff9-9eccf7b7dd61)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/65e17f96-430b-4bea-ae53-5fa7a17f4844)
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/a1c9e811-7e8d-4ac5-8352-d49bd2cca195)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-3-DS/assets/119478098/932be1fd-e64a-4745-b4af-76df776eebad)

      
### RESULT:
   Hence performing Feature Encoding and Transformation process is Successful.

       
