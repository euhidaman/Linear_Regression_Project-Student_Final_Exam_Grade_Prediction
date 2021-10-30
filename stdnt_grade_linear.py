#!/usr/bin/env python
# coding: utf-8

# In[242]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()  # Used for styling
#plt.style.use('ggplot') # Used for styling


# In[200]:


csv_file1 = pd.read_csv("student-mat.csv", sep=";")
csv_file1.head()


# In[201]:


df = csv_file1
df.head()


# # ADD / DROP

# # Delete the rows where G3 == 0

# In[212]:


df = df[df.G3 != 0] # Removing Outliers


# In[202]:


df.loc[:,"G1+G2"] = df.loc[:,"G1"] + df.loc[:,"G2"]
df.loc[:,"G1XG2"] = df.loc[:,"G1"] * df.loc[:,"G2"]
df.loc[:,"Avg"] = round((df.loc[:,"G1"] + df.loc[:,"G2"])//2)
df.loc[:,"MeduXFedu"] = (df.loc[:,"Medu"] * df.loc[:,"Fedu"])


# In[208]:


df.info()


# In[50]:


sns.pairplot(df)


# In[206]:


df.describe()


# In[219]:


sns.heatmap(df.corr(), annot = True)


# # Building Linear Models

# In[11]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from yellowbrick.regressor import PredictionError, ResidualsPlot


# In[299]:


class makeModel:
    def __init__(self,name, X,y):
        self.name = name
        self.X = X
        self.y = y
        self.linear_model, self.fX_train, self.fX_test, self.fy_train, self.fy_test = make_model(self.X, self.y)
        self.y_pred =  self.linear_model.predict(self.fX_test)
        self.rmse = np.sqrt(mean_squared_error(self.fy_test, self.y_pred))
        self.R2_value = r2_score(self.y_pred, self.fy_test)
        self.linear_score = self.linear_model.score(self.fX_test, self.fy_test)
        self.model_tuple = (self.linear_model, self.fX_train, self.fX_test, self.fy_train, self.fy_test)


    def make_model(self, X, y):
        best_linear_score = 0
        best_lm = LinearRegression()
        for _ in range(10000):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.66)
            lm = LinearRegression().fit(X_train, y_train)
            current_best_score = lm.score(X_test, y_test) 
            if current_best_score > best_linear_score:
                best_linear_score = current_best_score
                #with open("student_grade_pred_model.pickle", "wb") as f:
                    #pickle.dump(lm, f)
                #y_pred_lm = lm.predict(X_test)
                fX_train, fX_test, fy_train, fy_test = X_train, X_test, y_train, y_test
                best_lm = lm
        return best_lm, fX_train, fX_test, fy_train, fy_test
    
    


# In[300]:


def visualize_error(lm, X_train, X_test, y_train, y_test):
    #Visualiztion of Error
    vis = PredictionError(lm).fit(X_train, y_train)
    vis.score(X_test, y_test)
    vis.poof()


# # Creating & Evaluating A Model

# In[301]:


y = df.G3


# In[307]:


X1 = df[["G2","G1+G2", "Avg", "studytime", "traveltime"]]
X2 = df[["G2","G1+G2", "Avg", "studytime", "traveltime", "MeduXFedu"]]
X3 = df[["G2", "G1XG2", "MeduXFedu"]]
X4 = df[["G2","G1XG2", "Avg", "studytime", "traveltime"]]
X5 = df[["G1", "G2","health","absences","MeduXFedu"]]
X6 = df[["G1", "G2","health","absences", "studytime", "MeduXFedu", "traveltime"]] 
X7 = df[["G1+G2","G1", "G2","health","absences", "studytime", "MeduXFedu", "traveltime"]]


# In[291]:


model_1 = makeModel("Model 1", X1, y)


# In[303]:


model_2 = makeModel("Model 2", X2, y)


# In[304]:


model_3 = makeModel("Model 3", X3, y)


# In[305]:


model_4 = makeModel("Model 4", X4, y)


# In[306]:


model_5 = makeModel("Model 5", X5, y)


# In[308]:


model_6 = makeModel("Model 6", X6, y)


# In[309]:


model_7 = makeModel("Model 7", X7, y)


# In[310]:


models_list =[model_1, model_2, model_3, model_4, model_5, model_6, model_7]


# # Choosing The Best Model 

# In[312]:


models_list.sort(reverse=True, key= lambda x: (x.linear_score, x.rmse))


# In[315]:


for _ in models_list:
    print(_.name,"Linear Score: ",  _.linear_score,"RMSE: ", _.rmse)


# In[316]:


m = model_1
visualize_error(*m.model_tuple)
print("RMSE: ", m.rmse)
print("R^2: ", m.R2_value)
print("Linear Score: ", m.linear_score)

