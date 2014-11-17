'''
import numpy as np
from utils import plot_images
import matplotlib.pyplot as plt

filepath = './error/error.png'
y_lin = np.linspace(0,100,50)
x_coords = np.arange(0,np.shape(y_lin)[0])
fig=plt.figure()
plt.plot(x_coords,y_lin)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('test plot')
plt.grid(True)
fig.savefig(filepath)
'''

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_axes([1,1,1,1])
ax.plot([1,2])

fig.savefig('test.png')





