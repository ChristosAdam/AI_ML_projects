from matplotlib import pyplot as plt
import numpy as np

f = open('RR_SOUID100550.txt')
line = f.readline()   # include newline
k = 0;
table = []
while line:
        line = line.rstrip()  
        k = k + 1
        # The values on the txt file start after line 19
        if k == 19:
            print(line)
        elif k > 19:
            #Split each line of the values with , and append the values to a new table
            table.append(line.split(','))
        #Read next line
        line = f.readline()

l = k - 19

#Two tables one for the values and one for the dates
table1 = []
table2 = []

#Ite table for the plot
ite = []

#Precipitation data on each station start on different timelines so for each case the index of the values vary
for i in range(60995,63187):
    ite.append(i)
    table2.append(table[i][2])
    
    #If a value is missing marked as '-9999' append 0 to the table
    #Else append the value multiplied by 0.1, which is the value in millimiters
    if table[i][3] != '-9999':
        table1.append(float(table[i][3])*0.1 )
    else:
        table1.append(float(0) )

#Plot the values with the X axis showing every 365 days
plt.title("Bologna Precipitation")
plt.xticks(np.arange(0, len(table1)+1, 365))
plt.plot(table2, table1)
plt.show()

f.close()