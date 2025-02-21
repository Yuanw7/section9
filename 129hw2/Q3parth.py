# New lifting function
f_quad = lambda x, y: x**2 + x*y + y**2
points_3d_quad = lift_to_3d(points, f_quad)

# Update metric tensor (analytical)
def compute_curvatures_quad(x, y):
    # First fundamental form
    dfdx = 2*x + y
    dfdy = 2*y + x
    E = 1 + dfdx**2
    F = dfdx * dfdy
    G = 1 + dfdy**2
    g = np.array([[E, F], [F, G]])
    
    # Second fundamental form
    ddfddx = 2
    ddfddy = 2
    ddfdxdy = 1
    denom = np.sqrt(1 + dfdx**2 + dfdy**2)
    L = ddfddx / denom
    M = ddfdxdy / denom
    N = ddfddy / denom
    II = np.array([[L, M], [M, N]])
    
    # Shape operator and curvatures
    g_inv = np.linalg.inv(g)
    S = g_inv @ II
    k1, k2 = np.linalg.eigvals(S)
    return k1*k2, (k1 + k2)/2

# Repeat all previous steps with f_quad...
