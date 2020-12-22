from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_csv("C:\\Users\\apply\\Desktop\\JYH\\Quant Research\\Porfolio Optimization\\Optimizer\\Result\\normalization_column_3529-4779_with lag_adjusted.csv")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.log(data["kappa"])
y = data["look_back"]
z = data["frequency"]
c = data["sharpe ratio"]
plt.set_cmap('hot_r')
img = ax.scatter(x, y, z, c=c, cmap=plt.cm.get_cmap('coolwarm'))
ax.set_xlabel('$\kappa$')
ax.set_ylabel('$look back$')
ax.set_zlabel('$frequency$')
cbaxes = fig.add_axes([0.1, 0.05, 0.05, 0.7])
fig.colorbar(img, cax=cbaxes)
plt.show()