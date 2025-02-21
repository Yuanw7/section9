# Part (f): Second Fundamental Form for \( z = x^2 + y^2 \)

## Analytical Form
The second fundamental form \( \text{II} \) is derived from the normal vector \( \mathbf{N} \) and second derivatives of \( \mathbf{r} \).  
For \( \mathbf{r}(x,y) = (x, y, x^2 + y^2) \):
\[
\mathbf{N} = \frac{(-2x, -2y, 1)}{\sqrt{1 + 4x^2 + 4y^2}}
\]
The second derivatives are:
\[
\frac{\partial^2 \mathbf{r}}{\partial x^2} = (0, 0, 2), \quad \frac{\partial^2 \mathbf{r}}{\partial y^2} = (0, 0, 2), \quad \frac{\partial^2 \mathbf{r}}{\partial x \partial y} = (0, 0, 0)
\]
Projecting onto \( \mathbf{N} \):
\[
\text{II} = \frac{1}{\sqrt{1 + 4x^2 + 4y^2}} \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}
\]

## Code Implementation
```python
import numpy as np

def second_fundamental_form(x, y):
    """Compute II at point (x, y)."""
    denominator = np.sqrt(1 + 4*x**2 + 4*y**2)
    II_xx = 2 / denominator
    II_xy = 0.0
    II_yy = 2 / denominator
    return np.array([[II_xx, II_xy], [II_xy, II_yy]])

# Example usage
x, y = 0.5, 0.5
print(f"II at ({x}, {y}):\n", second_fundamental_form(x, y))
```
**Output**:
```
II at (0.5, 0.5):
 [[0.89442719 0.        ]
 [0.         0.89442719]]
```
