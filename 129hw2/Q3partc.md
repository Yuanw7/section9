# Part (c): Induced Metric for \( z = x^2 + y^2 \)

## Analytical Form
The metric tensor \( g_{ij} \) is derived from the lifting map \( \mathbf{r}(x,y) = (x, y, x^2 + y^2) \).  
Using the ambient Euclidean metric \( \mathbb{R}^3 \), the induced metric components are:
\[
g_{xx} = 1 + (2x)^2, \quad g_{xy} = 4xy, \quad g_{yy} = 1 + (2y)^2
\]
Thus:
\[
g = \begin{bmatrix}
1 + 4x^2 & 4xy \\
4xy & 1 + 4y^2
\end{bmatrix}
\]

## Numerical Verification (Code)
```python
import numpy as np

def induced_metric(x, y):
    """Compute the metric tensor at point (x, y)."""
    g_xx = 1 + (2 * x)**2
    g_xy = 4 * x * y
    g_yy = 1 + (2 * y)**2
    return np.array([[g_xx, g_xy], [g_xy, g_yy]])

# Example usage
x, y = 1.0, 0.5
print(f"Metric at ({x}, {y}):\n", induced_metric(x, y))
```
**Output**:
```
Metric at (1.0, 0.5):
 [[5.  2. ]
 [2.  2. ]]
```
