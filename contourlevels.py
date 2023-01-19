
from numpy import sqrt, array, pi, log, arctan, sin, cos
from numpy.linalg import det, eig
from matplotlib import patches
from astropy.modeling.functional_models import Gaussian2D as G2D


def g2d_function(x_mean, y_mean, x_std, y_std, rho):
    """
    Returns a Gaussian2D functions and its covariance matrix
    from the model.
        Args:
            x_mean (float): Mean in x.
            y_mean (float): Mean in y.
            x_std (float): Standard deviation in x.
            y_std (float): Standard deviation in y.
            rho_model (float): Rho from the model.
        Returns:
            g2d (G2D): 2D Gaussian function
            cov (numpy.array 2x2): Covariance matrix. 
    """
    cov = array([
        [x_std**2, rho*x_std*y_std],\
        [rho*x_std*y_std, y_std**2]
    ])
    denom = (2*pi*det(cov)**(1/2))           
    g2d = G2D(amplitude=1/denom,x_mean=x_mean, y_mean=y_mean, \
        cov_matrix=cov)
    return g2d, cov
    

def get_ellipse_props(cov, confidence):
    """
    Calculates the properties of the ellipse based on a given confidence.
        Args:
            cov (numpy.array 2x2): Covariance matrix.
            confidence (float): Quantile between 0 and 1.
        Returns:
            major_axis, minor_axis, angle (float): Length of the 
            respective axis'.
    """
    r_prime = -2*log(1-confidence)
    eigenvalues, eigenvectors = eig(cov)
    major_axis = sqrt(eigenvalues[0]*r_prime)
    minor_axis = sqrt(eigenvalues[1]*r_prime)
    angle = arctan(eigenvectors[1, 0]/eigenvectors[1, 1])
    return major_axis, minor_axis, angle


def ellipse_eq(x, angle, a, b):
    A = cos(angle)**2/a**2+sin(angle)**2/b**2
    B = 2*cos(angle)*sin(angle)*(1/a**2-1/b**2)
    C = cos(angle)**2/b**2+sin(angle)**2/a**2
    if (B*x)**2-4*C*(A*x**2-1)>=0:
        y0 = -(B*x)/(2*C)+sqrt((B*x)**2+4*C-4*A*C*x**2)/(2*C)
        y1 = -(B*x)/(2*C)-sqrt((B*x)**2+4*C-4*A*C*x**2)/(2*C)
    else:
        y0, y1 = 0,0
    return max([y0, y1]), min([y0, y1])


def make_ellipse_parameter_dict(cov_matrix, confidence_list):
    """
    Make a dict with the properties of the ellipse
        Args:
            cov_matrix (numpy.array 2x2): Covariance matrix.
            confidence_list (list): E.g. [0.5,0.9,0.99].
        Returns:
            ell_props (dict): Dictionary with parameters of the ellipses.
    """
    ell_props = {}
    for confd in confidence_list:
        ma, mi, ang = get_ellipse_props(cov_matrix, confidence=confd)
        ell_props[f'major_axis_{str(confd)[2:]}'] = ma
        ell_props[f'minor_axis_{str(confd)[2:]}'] = mi
        ell_props[f'angle_{str(confd)[2:]}'] = ang
    return ell_props


def colorcode_dataset(ell_props, df_datapoints, confidence_list):
    """
    Color codes the dataset based on the three quantiles given in confidence_list.
        Args:
            df_datapoints (pandas.DataFrame): Contians the datapoints (x_data, y_data)
            confidence_list (list): E.g. [0.5,0.9,0.99]
        Returns:
            ell_props (dict): Ellipse parameters.
            df_datapoints (pandas.DataFrame): With additional columns with Boolean statements.
    """
    #  Label the datapoints within the ellipse
    circle_0, circle_1, circle_2 = ([] for i in range(3))
    for x, y in zip(df_datapoints['x_data'].values, df_datapoints['y_data'].values):
        y_ellipse0_max, y_ellipse0_min = ellipse_eq(
            x, 
            ell_props[f'angle_{str(confidence_list[0])[2:]}'], 
            ell_props[f'major_axis_{str(confidence_list[0])[2:]}'], 
            ell_props[f'minor_axis_{str(confidence_list[0])[2:]}']
        )
        y_ellipse1_max, y_ellipse1_min = ellipse_eq(
            x, 
            ell_props[f'angle_{str(confidence_list[1])[2:]}'], 
            ell_props[f'major_axis_{str(confidence_list[1])[2:]}'], 
            ell_props[f'minor_axis_{str(confidence_list[1])[2:]}']
        )
        y_ellipse2_max, y_ellipse2_min = ellipse_eq(
            x, 
            ell_props[f'angle_{str(confidence_list[2])[2:]}'], 
            ell_props[f'major_axis_{str(confidence_list[2])[2:]}'], 
            ell_props[f'minor_axis_{str(confidence_list[2])[2:]}']
        )
        if y<y_ellipse0_max and y>y_ellipse0_min:
            circle_0.append(True)
            circle_1.append(False)
            circle_2.append(False)
        elif y<y_ellipse1_max and y>y_ellipse1_min and (y>y_ellipse0_max or y<y_ellipse0_min):
            circle_1.append(True)
            circle_0.append(False)
            circle_2.append(False)
        elif y<y_ellipse2_max and y>y_ellipse2_min and (y>y_ellipse1_max or y<y_ellipse1_min):
            circle_2.append(True)
            circle_1.append(False)
            circle_0.append(False)
        else:
            circle_0.append(False)
            circle_1.append(False)
            circle_2.append(False)

    df_datapoints[f'circle_0'] = circle_0
    df_datapoints[f'circle_1'] = circle_1
    df_datapoints[f'circle_2'] = circle_2
    return df_datapoints