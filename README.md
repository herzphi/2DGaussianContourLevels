# Two Dimensional Gaussian Contour Levels
Calculates the contours of a two dimensional normal distribution of a desired amount of probability. The idea is to transform the ellipse to a circle and calculate the integral of the PDF, set it equal to a quantile and tranform it back. The main steps are as follows:
```math
(\vec{x}-\vec{\mu})^\intercal \Sigma^{-1} (\vec{x}-\vec{\mu}) = d^2 \\
```
```math
\Sigma^{-1} = U \Lambda U^{\intercal}
```
```math

(\vec{x}-\vec{\mu})^\intercal U \Lambda U^{\intercal} (\vec{x}-\vec{\mu}) = d^2
```
```math
a = d \sqrt{\lambda_1} \text{ and } b = d \sqrt{\lambda_2}
```
```math
q = \int_A P(x,y) d\vec{x}
```
```math
q = \int_0^{R} r\,e^{-r^2/2} dr = 1-e^{-R^2/2}
```
```math
R = \sqrt{-2\ln{(1-q)}}
```
```math
a = R \sqrt{\lambda_1} = \sqrt{-2 \lambda_1\ln{(1-q)}}  \text{ and }  b = R \sqrt{\lambda_2} = \sqrt{-2 \lambda_2\ln{(1-q)}}
```
## Usage
Get a dictionary containing semi-major axis, semi-minor axis and the angle of the ellipse w.r.t to the major axis from example_ellipse_prop.py by the arguments of confidence as list and covariance matrix of your probability distribution function.

```bash
from numpy.random import randint
from numpy import dot, array

#  A random covariance matrix
off_diag = randint(10)
a, b = randint(1, 10), randint(1, 10)
A = array([
    [a, off_diag],
    [off_diag, b]
])
cov = dot(A, A.T)
```
`cov` can now be used in `make_ellipse_parameter_dict`:
```bash
from contourlevels import make_ellipse_parameter_dict

confidence_regions = [.5,.9,.99]  # Your desired confidence regions
ell_props = make_ellipse_parameter_dict(cov, confidence_regions)
```

The variable `cov` can be manipulated to any desired covariance matrix as long the mathmatical necessities are full filled.


For a single ellipse use `get_ellipse_props(cov, confidence)`.
```bash
cov = #  Your covariance matrix
confidence = 0.5

major_axis, minor_axis, angle = get_ellipse_props(cov, confidence)
```
## Visualization
For visualization use `example_ellipse_plot`.
![example](https://user-images.githubusercontent.com/102586476/213469564-109c0caa-fbe0-4c7c-af6c-0e798cc6528d.png)
