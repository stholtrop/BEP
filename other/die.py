import numpy as np
import matplotlib.pyplot as plt

def X(x):
    return np.round(x)
def Y(x):
    v = np.round(x)
    return np.isin(v,[1,2]) * 1.5 + np.isin(v, [3,4])* 3.5 + np.isin(v, [5,6])* 5.5

for i in range(1, 7):
    xs = [i - 0.48, i + 0.5]
    Xs = [X(i), X(i)]
    Ys = [Y(i), Y(i)]
    if i == 1:
        plt.plot(xs, Xs, linewidth=3, color='lightgrey', label="$X$")
        plt.plot(xs, Ys, linewidth=3, color='red', label="$\mathbb{E}[X|\mathscr{F}]$")
    else: 
        plt.plot(xs, Xs, linewidth=3, color='lightgrey')
        plt.plot(xs, Ys, linewidth=3, color='red')

for i in np.linspace(0.5, 6.5, 4):
    xs = [i, i]
    ys = [0.5, 6.5]
    plt.plot(xs, ys, linewidth=2, color='black')
for i in np.linspace(1.5, 5.5, 3):
    xs = [i, i]
    ys = [0.5, 6.5]
    plt.plot(xs, ys, linewidth=2, color='grey', linestyle='--')
plt.legend(bbox_to_anchor=(0.05,1), loc="upper left")
plt.xlabel("$\Omega$")



plt.show()