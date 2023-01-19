from contourlevels import make_ellipse_parameter_dict
from numpy.random import randint
from numpy import dot, array

#  The properties of the ellipses are calculated
#  based on the quantiles given and the covariance matrix.
#  Calculate a random covariance matrix.
#  For a given covariance matrix manipulate the array
#  'cov'

off_diag = randint(10)
a, b = randint(1, 10), randint(1, 10)
A = array([
    [a, off_diag],
    [off_diag, b]
])
cov = dot(A, A.T)
#  Calculate the major, minor and angle of the ellipses.
ell_props = make_ellipse_parameter_dict(cov, [.5,.9,.99])