import numpy as np
from matplotlib import pyplot as plt

ep_var = np.load("ep_var.npy")
errors = np.load("errors.npy")
communicate_indices = np.load("communicate_indices.npy")


print(ep_var)
print(len(communicate_indices))
plt.figure()
plt.plot(np.arange(len(ep_var)), ep_var, label="variance")
plt.scatter(
    communicate_indices, [ep_var[i] for i in communicate_indices], label="communicate"
)

plt.figure()
plt.plot(np.arange(len(errors)), errors, label="errors")
plt.scatter(
    communicate_indices, [errors[i] for i in communicate_indices], label="communicate"
)

plt.legend()
plt.show()
