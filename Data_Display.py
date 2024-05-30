from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
import numpy as np

def readPrecipitation(name, minvalue, maxvalue, mi, offset):
    f = open(name)
    line = f.readline()   
    k = 0;
    table = []
    while line:
        line = line.rstrip()  

        k = k + 1
        if k > offset: # Get Data after line k
            table.append(line.split(','))
        line = f.readline()

    l = k - offset
    table1 = []
    table2 = []

    for i in range(minvalue - mi,maxvalue):
    
        table2.append(table[i][2])
        if table[i][3] != '-9999': # Fill with 0 if negative value
            table1.append(float(table[i][3])*0.1 )
        else:
            table1.append(float(0) )

    f.close()
    return table1, table2

def readDischarge(name, offset):
    f = open(name)
    line = f.readline()   
    k = 0;
    tableDis = list()
    while line:
        line = line.rstrip() 
    
        k = k + 1
        if k > offset: # Get Data after line k
            tableDis.append(line.split(';'))
        line = f.readline()
    l = k - offset

    tableD = list()
    tableD2 = list()

    for i in range(len(tableDis)): # Create table with discharge data
        tableD.append(tableDis[i][0])
        tableD2.append(float(tableDis[i][2]))

    f.close()
    
    return tableD, tableD2


fig, axs = plt.subplots(1, 2)

mi = 12;

# Read Data
table1, table2 = readPrecipitation('RR_SOUID100551.txt', 36889, 39081, mi, 19)
table3, table4 = readPrecipitation('RR_SOUID100751.txt', 28854, 31046, mi, 22)
table5, table6 = readPrecipitation('RR_SOUID119285.txt', 28854, 31046, mi, 22)
table7, table8 = readPrecipitation('RR_SOUID100553.txt', 51134, 53326, mi, 19)
table9, table10 = readPrecipitation('RR_SOUID100554.txt', 44559, 46751, mi, 19)
table11, table12 = readPrecipitation('RR_SOUID112197.txt', 10591, 12784, mi, 22)
table13, table14 = readPrecipitation('RR_SOUID100550.txt', 60985, 63187, mi, 19)
table15, table16 = readPrecipitation('RR_SOUID100550.txt', 10592, 12784, mi, 19)

tableD, tableD2 = readDischarge('6348800_Q_Day.Cmd.txt', 37)

mini=383
maxi=6330

l = len(table1)
tablea1 = np.zeros((l))
tablea3 = np.zeros((l))
tablea5 = np.zeros((l))
tablea7 = np.zeros((l))
tablea9 = np.zeros((l))
tablea11 = np.zeros((l))
tablea13 = np.zeros((l))
tablea15 = np.zeros((l))

for i in range(len(table1)): # Create mean value for rainfall data
    if i > mi-1:
        tablea1[i] = (table1[i-1]+table1[i-2]+table1[i-3] + table1[i-4]+ table1[i-5])/5
        # # tablea3[i] = (table3[i-5]+table3[i-6]+table3[i-7]+table3[i-8]+table3[i-9]+table3[i-10]+table3[i-11]+table3[i-12])/7
        tablea5[i] = (table5[i-6]+table5[i-7]+table5[i-8]+table5[i-9]+table5[i-10]+table5[i-11]+table5[i-12])/7
        tablea7[i] = (table7[i-8]+table7[i-1]+table7[i-2]+table7[i-3]+table7[i-4]+table7[i-5]+table7[i-6]+table7[i-7])/8
        tablea9[i] = (table9[i-2]+table9[i-3]+table9[i-4]+table9[i-5]+table9[i-6]+table9[i-7]+table9[i-8])/7
        tablea11[i] = (table11[i-3]+table11[i-4]+table11[i-5]+table11[i-6]+table11[i-7])/5
        tablea13[i] = (table13[i-1]+table13[i-2]+table13[i-3]+table13[i-4]+table13[i-5])/5
        tablea15[i] = (table15[i-2]+table15[i-3]+table15[i-4]+table15[i-5]+table15[i-6]+table15[i-7])/6

FTable = np.array([tablea1[mi:],  tablea5[mi:], tablea7[mi:], tablea9[mi:], tablea11[mi:], tablea13[mi:],tablea15[mi:]], float)

X = FTable
X = X.transpose()
y = np.array(tableD2)

y = (y - mini) / (maxi - mini)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1, test_size=0.3)

#scale
scale_X = StandardScaler()
X_trainscaled=scale_X.fit_transform(X_train)
X_testscaled=scale_X.transform(X_test)

'''
q_t = preprocessing.QuantileTransformer(n_quantiles=1500,
output_distribution='uniform', random_state=0)
X_trainscaled = q_t.fit_transform(X_train)
X_testscaled=q_t.transform(X_test)

'''

reg1 = MLPRegressor(solver='lbfgs', activation="relu" ,random_state=0, max_iter=1000).fit(X_trainscaled, y_train)


y_pred=reg1.predict(X_testscaled)
print("The r2 Score for MLP", (r2_score(y_pred, y_test)))

reg2 = GradientBoostingRegressor(random_state = 1,n_estimators= 1000,max_depth=7,min_samples_split= 5,loss= 'ls').fit(X_trainscaled, y_train)

y_pred=reg2.predict(X_testscaled)
print("The r2 Score for GB", (r2_score(y_pred, y_test)))


reg3 =  ExtraTreesRegressor(n_estimators = 1000, random_state = 1,max_depth = 22).fit(X_trainscaled, y_train)

y_pred=reg3.predict(X_testscaled)
print("The r2 Score for ET", (r2_score(y_pred, y_test)))

reg4 = RandomForestRegressor(n_estimators = 1000, random_state = 1,max_depth = 22).fit(X_trainscaled, y_train)

y_pred=reg4.predict(X_testscaled)
print("The r2 Score for RF", (r2_score(y_pred, y_test)))

K = 1
reg5 = KNeighborsRegressor(n_neighbors = K).fit(X_trainscaled, y_train)
y_pred=reg5.predict(X_testscaled)
print("k= ", K," The r2 Score for KNN ", (r2_score(y_pred, y_test)))

reg6 =  VotingRegressor(estimators=[('mlp', reg1),('gb', reg2),('et',reg3),('rf', reg4),('kn', reg5)]).fit(X_trainscaled, y_train)

reg =StackingRegressor(estimators=[('mlp', reg1),('gb', reg2),('et',reg3),('rf', reg4)]).fit(X_trainscaled, y_train)


y_pred=reg.predict(X_testscaled)

for i in range(1,len(y_pred)): # Cut the outliers
    if (y_pred[i] < 0) or (y_pred[i] > 1.25):
        y_pred[i] = y_pred[i-1]

MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)


print("Stacking Regressor: The r2 Score is ", (r2_score(y_pred, y_test)))
print("Stacking Regressor: The MSE Score is ", MSE)

print("Stacking Regressor: The RMSE Score is ", RMSE)

tablea17 = np.arange(len(y_test)) # Reverse the scale to real values
y_new = (y_pred )*(maxi-mini)+ mini
y_newt = (y_test )*(maxi-mini)+ mini

ytes=3500

numP = 0
numT = 0

for i in range(len(y_new)): # Find cases above threshold
    if (y_new[i] > ytes):
        numP = numP +1
    if (y_newt[i] > ytes):
        numT = numT +1

p = (numP/numT)*100

print("The number of instances exceeding the upper limit are" ,numP,"the model predicted",numT,"a percentage of" ,p,"%")

fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_title("Stacking Regressor Predicted Points")
ax.scatter(tablea17,y_newt,s=10, c='b', marker="s", label='real')
ax.plot(tablea17,y_new, c='r', marker="o", label='MLP Prediction')
ax.plot(tablea17,ytes*np.ones(len(tablea17)), c='g', label='MLP Prediction')

plt.show()