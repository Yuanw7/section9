import numpy as np

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) if r != 0 else 0
    phi = np.arctan2(y, x)
    return r, theta, phi

def cylindrical_to_cartesian(rho, psi, z):
    x = rho * np.cos(psi)
    y = rho * np.sin(psi)
    return x, y, z

def cartesian_to_cylindrical(x, y, z):
    rho = np.sqrt(x**2 + y**2)
    psi = np.arctan2(y, x)
    return rho, psi, z

def spherical_to_cylindrical(r, theta, phi):
    rho = r * np.sin(theta)
    psi = phi
    z = r * np.cos(theta)
    return rho, psi, z

def cylindrical_to_spherical(rho, psi, z):
    r = np.sqrt(rho**2 + z**2)
    theta = np.arctan2(rho, z)
    phi = psi
    return r, theta, phi
