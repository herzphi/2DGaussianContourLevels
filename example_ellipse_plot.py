from contourlevels import make_ellipse_parameter_dict, g2d_function, colorcode_dataset
from numpy.random import randint, multivariate_normal
from numpy import dot, array, pi, sqrt, std, mean
from pandas import DataFrame
from matplotlib.pyplot import subplots, show, legend
from matplotlib.patches import Ellipse


#  Random sample of 5000
#  Draws three contour lines.

mean_g2d = [0, 0]
off_diag = randint(10)
A = array([
    [randint(1, 10), off_diag],
    [off_diag, randint(1, 10)]
])
cov = dot(A, A.T)
x_data, y_data = multivariate_normal(mean_g2d, cov, 5000).T

rhosigma_1sigma_2 = cov[0][1]
rho = rhosigma_1sigma_2/(sqrt(cov[0][0])*sqrt(cov[1][1]))

g2d_test, cov_test = g2d_function(
    mean(x_data),
    mean(y_data),
    std(x_data), 
    std(y_data), 
    rho
)

ell_props = make_ellipse_parameter_dict(cov_test, [.5,.9,.99])

df_test = DataFrame({
    'x_data':x_data,
    'y_data':y_data,
})

df_test = colorcode_dataset(ell_props, df_test, [.5,.9,.99])

fig, axs = subplots(1, figsize=(5,5))
linecolor = 'mediumseagreen'
for confd in [.5, .9, .99]:
    axs.add_patch(
        Ellipse(
            (g2d_test.x_mean.value, g2d_test.y_mean.value), 
            width=2*ell_props[f'major_axis_{str(confd)[2:]}'],
            height=2*ell_props[f'minor_axis_{str(confd)[2:]}'],
            angle=360*ell_props[f'angle_{str(confd)[2:]}']/(2*pi),
            facecolor='none',
            edgecolor=linecolor,
            linewidth=1.5,
        )
    )

for i in range(3):
    pct = len(df_test[df_test[f'circle_{i}']==True])/5000
    axs.scatter(
        df_test[df_test[f'circle_{i}']==True]['x_data'], 
        df_test[df_test[f'circle_{i}']==True]['y_data'],
        s=1.5,
        alpha=.4,
        label=f"{len(df_test[df_test[f'circle_{i}']==True])} ({100*pct:.0f} %) samples"
    )
len_rest = len(df_test[(df_test[f'circle_0']==False) &
                (df_test[f'circle_1']==False) &
                    (df_test[f'circle_2']==False)])
axs.scatter(
        df_test[
            (df_test[f'circle_0']==False) & \
                (df_test[f'circle_1']==False) & \
                    (df_test[f'circle_2']==False)]['x_data'], 
        df_test[
            (df_test[f'circle_0']==False) & \
                (df_test[f'circle_1']==False) & \
                    (df_test[f'circle_2']==False)]['y_data'],
        s=1.5,
        alpha=.4,
        color='gray',
        label=f"{len_rest} samples"
    )


axs.set_title('Normal distributed samples (Total=5000)')
axs.set_xlabel('x')
axs.set_ylabel('y')
limit_axis = max([*x_data, *y_data])
#plt.xlim(-limit_axis,limit_axis)
#plt.ylim(-limit_axis,limit_axis)
legend(markerscale=5)   
#plt.savefig('./plots/norm_dist_quantiles.png', dpi=300, format='png', bbox_inches='tight')
show()