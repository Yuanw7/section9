# part a

The Lorenz equations model Rayleigh-Bénard convection - the motion of a fluid
layer heated from below and cooled from above. This creates circulating rolls of fluid.

VARIABLES:
- x: Rate of convective fluid motion (how fast the fluid circulates)
- y: Temperature difference between rising and sinking fluid currents
- z: Deviation from linear vertical temperature profile

PARAMETERS:
- σ (sigma): Prandtl number - ratio of momentum diffusivity to thermal diffusivity
- ρ (rho): Rayleigh number - measures driving force from temperature gradient
- β (beta): Geometric factor related to the fluid layer's aspect ratio

MATHEMATICAL REPRESENTATION:
dx/dt = σ(y - x)  → Convection driven by temperature differences
dy/dt = x(ρ - z) - y → Feedback between motion and temperature
dz/dt = xy - βz → Energy dissipation and nonlinear coupling

KEY INSIGHTS:
1. Demonstrates deterministic chaos - small changes lead to vastly different outcomes
2. Shows how order emerges from turbulent systems
3. Foundation for understanding chaotic systems in weather, climate, and physics

#part b

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Lorenz equations
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Parameters and initial conditions
sigma, rho, beta = 10, 48, 3
initial_state = [1.0, 1.0, 1.0]
t_span = (0, 12)
t_eval = np.linspace(0, 12, 5000)

# Solve ODE
solution = solve_ivp(lorenz, t_span, initial_state, args=(sigma, rho, beta),
                     t_eval=t_eval, method='LSODA')

# Extract coordinates
x, y, z = solution.y

# Plot attractor
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Lorenz Attractor (σ=10, ρ=48, β=3)')
plt.show()

#part c

import matplotlib.animation as animation

# Initialize plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

line, = ax.plot([], [], [], lw=0.5)
points, = ax.plot([], [], [], 'o', markersize=2, color='red')

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    points.set_data([], [])
    points.set_3d_properties([])
    return line, points

def animate(i):
    line.set_data(x[:i], y[:i])
    line.set_3d_properties(z[:i])
    points.set_data(x[i], y[i])
    points.set_3d_properties(z[i])
    ax.view_init(30, 0.3 * i)  # Rotate view
    return line, points

# Create animation
ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(x), interval=10, blit=True)

# Save video
ani.save('lorenz_attractor.mp4', writer='ffmpeg', fps=30)
plt.close()
