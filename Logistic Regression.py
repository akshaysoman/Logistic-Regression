#====================TASK 1 ============================================
import pandas as pd
import math
import numpy as np
import statistics

#creating the sigmoid function
def sigma(t):
    a = math.exp(-t)
    b = 1/(1+a)
    return b

#creating F values in the method F

def F(x,lamda):
    sums = 0
    for k in range(len(lamda)):       
        sums += x[k]*lamda[k]
    Fs = sigma(sums)
    Fval = Fs
    return Fval

#creating method to find negative log likelihood 
def objective(x, y, lamda):
    loss = []
    p=0
    for j in x:
       sums = 0
       for k in range(len(lamda)):
           #multiplying through given lamda and summing
            sums += j[k]*lamda[k]
     
       m = sigma(sums)
       loss.append(m)
    p=0
    for n in loss:
       p  += -(math.log(n))
    return p

 
def C(x,y,lamda):
    Cval = []
    cn = (y-(1-y)) / ((y*F(x,lamda)) + ((1-y) * F(x,lamda)))
    Cval =cn
    return Cval                                                       
                                                                                                                  
    
def gradF(x,lamda):
    GFval1 = []
    GFval= (F(x,lamda))* (1 - (F(x,lamda)))
    GFval1=GFval                      
    return GFval1
                          
# method to find gradient
def grad(x,y,lamda):
    l=[]
    gradient =[]
    y = list(y)
    for i in range(len(x)):
        ls=-(gradF(x[i],lamda)*C(x[i],y[i],lamda))     # <-c*gradF
        l.append(ls)
     
    for i in range(len(lamda)):
          sums=0
          for j in range(len(x)):        
              sums += l[j]*x[j][i]
          gradient.append(sums)
    return gradient

# method to find gradient descent   
def gradDescent(x,y,lamda,T,eta):
    
    print("start")
    print("lamda=",lamda)
    print("objective(x)=",objective(x,y,lamda))
    print("grad(x)=",grad(x,y,lamda))
    for i in range(T):
         gradi = grad(x,y,lamda)
         res = []
         a1 =[]
         for k in range(len(gradi)):
             a= eta * -gradi[k]
             a1.append(a)
             ans = lamda[k] + a1[k]
             res.append(ans)
         lamda = res
    print("end")
    print("lamda=",lamda)
    print("objective(x)=",objective(x,y,lamda))
    print("grad(x)=",grad(x,y,lamda))
    return lamda

def predict(X,final_lamda):
    pre = []
    prediction=[]
    for i in range(len(X)):
        a = list(X[i])
        a.insert(0,1)
        b = np.array(a)
        pre.append(b)
    pre = np.array(pre)
    for i in range(len(pre)):
        sum=0
        for j in range(len(final_lamda)):
            sum += final_lamda[j]*pre[i][j]
        if (sum<.5):
            sum = 0
        else:
            sum = 1
        prediction.append(sum)
    prediction=np.array(prediction)
    return prediction

def errorrate(predicted_value,truex):
    l=[]
    for i in truex:
        a = (truex-predicted_value)^2
    l.append(a)
    errors=statistics.mean(a)
    return errors


#====================TASK 2 ============================================
#loading the dataset auto
#removing impurities
#creating labels and attributes as given

from sklearn import preprocessing
df = pd.read_csv("auto.csv")
df.replace('?',np.nan,inplace=True)
df.dropna(inplace=True)
df['horsepower'] = df['horsepower'].astype(float)
df['weight'] = df['weight'].astype(float)
df['year'] = df['year'].astype(float)
df['origin'] = df['origin'].astype(float)
df['high'] = [1 if i>=23 else 0 for i in df.mpg]
df['origin1'] = [0 if i==1 else 1 if i==2 else 0 for i in df.origin]
df['origin2'] = [0 if i==1 else 0 if i==2 else 1 for i in df.origin]

horsepower = df['horsepower']
weight = df['weight']
year = df['year']
origin1 = df['origin1']
origin2 = df['origin2']
data = np.array([horsepower,weight,year,origin1,origin2])
attributes = preprocessing.normalize(data)
attributes = attributes.T
label = df['high']


#====================TASK 3 ======================================  
# Splitting the data set randomly into two equal parts


from sklearn.model_selection import train_test_split

X_train,X_test,y_train, y_test = train_test_split(attributes,label, test_size=0.50,random_state=142)

#====================TASK 4 =======================================
#Creating independent random numbers in the range [−0.7, 0.7] as the initial weights

import random
lambda_new=[]
for i in range(len(data)+1):
    weights = random.uniform(-0.7,0.7)
    lambda_new.append(weights)
lamda_new = np.array(lambda_new)

#Training the Algorithm

res = []
for i in range(len(X_train)):
    a = list(X_train[i])
    a.insert(0,1)
    b = np.array(a)
    res.append(b)
res = np.array(res)
T = 100
eta = 0.0001

#finding the new value of lamda after 100 iterations

final_lamda = gradDescent(res, y_train, lamda_new,T,eta)  
print(final_lamda)

#getting predictions for test and train data
test = predict(X_test,final_lamda)
train = predict(X_train,final_lamda)

#getting error rates for test and train data

a = errorrate(test,y_test)
print("test error : " ,a)
b = errorrate(train,y_train)
print("train error : " ,b)



"""
#we can use the "solution to logistic regression" to check the correctess of the algorithm
    
x = [[1,1],[1,2],[2,2],[2,1]]
xtest = [1,x]
y =[1,0,1,1]
lamda =[1/3,1/3,-1/3]
for i in x:
    i.insert(0,xtest[0])             
final_lamda = gradDescent(x, y, lamda)
print(final_lamda)

"""