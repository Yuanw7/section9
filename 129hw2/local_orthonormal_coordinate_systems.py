import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def generate_sphere_vectors():
    theta_vals = np.linspace(0, np.pi, 10)
    phi_vals = np.linspace(0, 2 * np.pi, 20)
    theta, phi = np.meshgrid(theta_vals, phi_vals)
    
    x, y, z = spherical_to_cartesian(1, theta, phi)
    
    e_r_x, e_r_y, e_r_z = x, y, z
    e_theta_x = np.cos(theta) * np.cos(phi)
    e_theta_y = np.cos(theta) * np.sin(phi)
    e_theta_z = -np.sin(theta)
    e_phi_x = -np.sin(phi)
    e_phi_y = np.cos(phi)
    e_phi_z = np.zeros_like(phi)
    
    return x, y, z, e_r_x, e_r_y, e_r_z, e_theta_x, e_theta_y, e_theta_z, e_phi_x, e_phi_y, e_phi_z

def plot_sphere_vectors():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z, e_r_x, e_r_y, e_r_z, e_theta_x, e_theta_y, e_theta_z, e_phi_x, e_phi_y, e_phi_z = generate_sphere_vectors()
    
    ax.quiver(x, y, z, e_r_x, e_r_y, e_r_z, color='r', label='n')
    ax.quiver(x, y, z, e_theta_x, e_theta_y, e_theta_z, color='b', label='t')
    ax.quiver(x, y, z, e_phi_x, e_phi_y, e_phi_z, color='g', label='c.t')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Surface with Normal Vectors")
    ax.legend()
    
    plt.show()

plot_sphere_vectors()
