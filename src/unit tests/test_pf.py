
import numpy as np
import matplotlib.pyplot as plt
class ParticleFilter:
    def __init__(self, num_particles, x_range, y_range, theta_range):
        self.num_particles = num_particles
        self.particles = np.empty((num_particles, 3))  # x, y, theta
        self.particles[:, 0] = np.random.uniform(0, x_range, num_particles)  # x
        self.particles[:, 1] = np.random.uniform(0, y_range, num_particles)  # y
        self.particles[:, 2] = np.random.uniform(0, theta_range, num_particles)  # theta
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, move_x, move_y, rotate_theta, std_pos, std_theta):
        """ Move the particles based on the motion model. """
        self.particles[:, 0] += move_x + np.random.randn(self.num_particles) * std_pos
        self.particles[:, 1] += move_y + np.random.randn(self.num_particles) * std_pos
        self.particles[:, 2] += rotate_theta + np.random.randn(self.num_particles) * std_theta

    def update(self, z, std):
        """ Update particle weights based on measurement. """
        distances = np.linalg.norm(self.particles[:, :2] - z, axis=1)
        self.weights = np.exp(-distances ** 2 / (2 * std ** 2))
        self.weights += 1.e-300  # avoid divide by zero
        self.weights /= sum(self.weights)

    def resample(self):
        """ Resample particles based on weights. """
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        """ Estimate the current position and orientation. """
        return np.average(self.particles, weights=self.weights, axis=0)




# Example usage
num_particles = 10000
x_range = 100
y_range = 100
theta_range = 2 * np.pi  # 0 to 360 degrees in radians
pf = ParticleFilter(num_particles, x_range, y_range, theta_range)

# Simulate motion and measurement
for _ in range(100):
    pf.predict(move_x=1, move_y=1, rotate_theta=np.pi/10, std_pos=0.5, std_theta=0.1)
    pf.update(z=np.array([50, 50]), std=10)
    pf.resample()
    estimated_position = pf.estimate()
    print(f"Estimated Position and Orientation: {estimated_position}")

# Optional: Plotting
plt.quiver(pf.particles[:, 0], pf.particles[:, 1], np.cos(pf.particles[:, 2]), np.sin(pf.particles[:, 2]), alpha=0.5)
plt.scatter(50, 50, c='red', marker='x')  # actual position
plt.xlim(0, x_range)
plt.ylim(0, y_range)
plt.show()