from multispline.gpu_tricubic_spline import AmpInterp3dGPU
from multispline.spline import TricubicSpline
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

def test_function_3d(x, y, z):
    return np.sin(x) * np.cos(4*y) + np.i0(3.2*z)

NX = 33
NY = 65
NZ = 17
sample_points_x = np.linspace(0, 5, NX)
sample_points_y = np.linspace(0, 5, NY)
sample_points_z = np.linspace(-2, 2, NZ)
sample_points_grid = np.meshgrid(sample_points_x, sample_points_y, sample_points_z, indexing='ij')
sample_values_f = test_function_3d(*sample_points_grid)


tspl = TricubicSpline(sample_points_x, sample_points_y, sample_points_z, sample_values_f)

test_points_x = np.linspace(0, 5, 77)
test_points_y = np.linspace(0, 5, 81)
test_points_z = np.linspace(-2, 2, 59)
test_points_grid = np.meshgrid(test_points_x, test_points_y, test_points_z, indexing='ij')
test_values_f = test_function_3d(*test_points_grid)
tspl_values_f = tspl(*test_points_grid)

average_error = np.max(np.abs(tspl_values_f - test_values_f), axis=0)
plot_grid = np.meshgrid(test_points_y, test_points_z, indexing='ij')

# plt.pcolormesh(*plot_grid, average_error, shading='gouraud', norm=LogNorm())
# cbar = plt.colorbar()
# cbar.set_label('Max absolute error in x', rotation=270, labelpad=15)
# plt.xlabel('y')
# plt.ylabel('z')
# plt.show()

coeffs_unshape = tspl.coefficients
coeffs_reshape = coeffs_unshape.reshape(NX-1, NY-1, NZ-1, 64)
coeffs_reshape_in = np.array([coeffs_reshape for i in range(10)])

sample_points_grid = np.meshgrid(sample_points_x, sample_points_y, sample_points_z, indexing='ij')

spline_handler = AmpInterp3dGPU(
    coeffs_reshape_in, 
    sample_points_x[0].item(),
    (sample_points_x[1] - sample_points_x[0]).item(),
    sample_points_x.size - 1,
    sample_points_y[0].item(), 
    (sample_points_y[1] - sample_points_y[0]).item(),
    sample_points_y.size - 1,
    sample_points_z[0].item(),
    (sample_points_z[1] - sample_points_z[0]).item(),
    sample_points_z.size - 1
)

print("IN")

gpu_res = spline_handler.evaluate_3d_spline(
    test_points_grid[0],
    test_points_grid[1], 
    test_points_grid[2],
    )

print(tspl_values_f.shape, gpu_res[0].shape)


plt.pcolormesh(*plot_grid, tspl_values_f[0,:,:], shading='gouraud', norm=LogNorm())
plt.xlabel('y')
plt.ylabel('z')
plt.savefig('cpu_slice.pdf',bbox_inches="tight")
plt.close()

plt.pcolormesh(*plot_grid, gpu_res[0][0,:,:], shading='gouraud', norm=LogNorm())
plt.xlabel('y')
plt.ylabel('z')
plt.savefig('gpu_slice.pdf',bbox_inches="tight")
plt.close()

average_error = np.max(np.abs(tspl_values_f - test_values_f), axis=0)
plot_grid = np.meshgrid(test_points_y, test_points_z, indexing='ij')

plt.pcolormesh(*plot_grid, average_error, shading='gouraud', norm=LogNorm())
cbar = plt.colorbar()
cbar.set_label('Max absolute error in x', rotation=270, labelpad=15)
plt.xlabel('y')
plt.ylabel('z')
plt.savefig('cpu_err.pdf',bbox_inches="tight")
plt.close()

average_error = np.max(np.abs(gpu_res[0] - test_values_f), axis=0)
plot_grid = np.meshgrid(test_points_y, test_points_z, indexing='ij')

plt.pcolormesh(*plot_grid, average_error, shading='gouraud', norm=LogNorm())
cbar = plt.colorbar()
cbar.set_label('Max absolute error in x', rotation=270, labelpad=15)
plt.xlabel('y')
plt.ylabel('z')
plt.savefig('gpu_err.pdf',bbox_inches="tight")
plt.close()

average_error = np.max(np.abs(gpu_res[1] - test_values_f), axis=0)
plot_grid = np.meshgrid(test_points_y, test_points_z, indexing='ij')

plt.pcolormesh(*plot_grid, average_error, shading='gouraud', norm=LogNorm())
cbar = plt.colorbar()
cbar.set_label('Max absolute error in x', rotation=270, labelpad=15)
plt.xlabel('y')
plt.ylabel('z')
plt.savefig('gpu_err_1.pdf',bbox_inches="tight")
plt.close()

