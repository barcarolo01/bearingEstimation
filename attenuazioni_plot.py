from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
data = genfromtxt('attenuation_resume_babel.csv', delimiter=',')
data_USA = genfromtxt('attenuation_resume.csv', delimiter=',')

fig = plt.figure(figsize=(10, 8))
plt.tight_layout()
#plt.plot(data[:,0],np.log(data[:,1]),marker='o')
plt.plot(data[:,0],np.log(data[:,2]),marker='o')
plt.plot(data[:,0],np.log(data_USA[:,2]),marker='o',color='red')

#plt.plot(data[:,0],data[:,1]-data_USA[:,])
plt.show()