import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, y, sigma):
    inv = np.linalg.inv(sigma)
    print(inv)
    return np.exp(-0.5*(x**2*inv[0,0] + 2*x*y*inv[0,1] + y**2*inv[1,1]))/2/np.pi/(np.linalg.det(sigma))**0.5

x = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, x)

print(X.shape)
sigma = np.array([[2, 1], [1, 2]])

Z = gaussian(X, Y, sigma)

fig, ax = plt.subplots(1, 2)
cset1 = ax[0].contourf(X, Y, Z, 100)
ax[0].set_xlabel("$X_1$")
ax[0].set_ylabel("$X_2$")

ax[1]
plt.show()
