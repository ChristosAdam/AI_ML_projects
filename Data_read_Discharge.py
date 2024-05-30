from matplotlib import pyplot as plt
import numpy as np

f = open('6348800_Q_Day.Cmd.txt')
line = f.readline()   # include newline
k = 0;
table = []
while line:
        line = line.rstrip()  
        
        k = k + 1
        #The values on txt file start on line 37
        if k > 37:
        #Split the line and append values to a table
            table.append(line.split(';'))
        line = f.readline()
l = k - 37 
#Two tables one for the values and one for the dates
table1 = []
table2 = []
#Ite table for the plot
ite = []
for i in range(len(table)):
    #Table2 contains the dates which are on the first column of the table
    #Table1 contains the float values of the discharge which are on the third column of the table
    ite.append(i)
    table2.append(table[i][0])
    table1.append(float(table[i][2]))

#Plot the values with the X axis showing every 365 days
plt.title("Pontelagoscuro Mean Daily Discharge Data (m³/s)")
plt.xticks(np.arange(0, len(table1)+1, 365))
plt.plot(table2, table1)
plt.show()


f.close()