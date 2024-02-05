import numpy as np
from matplotlib import pyplot as plt

ep_var = np.load("ep_var.npy")
communicate_indices = np.load("communicate_indices.npy")


print(ep_var)
plt.plot(np.arange(len(ep_var)), ep_var, label="variance")
plt.scatter(
    communicate_indices, [ep_var[i] for i in communicate_indices], label="communicate"
)


plt.legend()
plt.show()
