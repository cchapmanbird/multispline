from numba import cuda, njit
import numpy as np
# import cupy as cp

class AmpInterp3dGPU:
    def __init__(self, coeffs, x0, dx, nx, y0, dy, ny, z0, dz, nz):
        self.coeffs = coeffs.reshape(coeffs.shape[0], -1)
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.ng = coeffs.shape[0]
        self.grid_shape = coeffs.shape[1]
    
    def evaluate_3d_spline(self, x, y, z, out=None):
        if out is None:
            out = cuda.device_array((self.ng, x.size))

        threadsperblock = 256
        blockspergrid = ((x.size + (threadsperblock - 1)) // threadsperblock, self.grid_shape)
        return _evaluate_3d_spline_kernel[blockspergrid, threadsperblock, 0, 8*self.grid_shape](
            out,
            x, 
            y, 
            z, 
            self.coeffs,
            self.x0,
            self.dx,
            self.nx,
            self.y0,
            self.dy,
            self.ny,
            self.z0,
            self.dz,
            self.nz,
            self.ng,
            self.grid_shape
            )

@njit
def get_interval(a, a0, da, na):
    idx = int((a - a0) / da)
    if idx < 0:
        return 0
    elif idx >= na:
        return na - 1
    return idx   

@njit
def _compute_tricubic_spline_value(coeffs_here, x, x0, dx, nx, y, y0, dy, ny, z, z0, dz, nz):
    result = 0.

    i = get_interval(x, x0, dx, nx)
    j = get_interval(y, y0, dy, ny)
    k = get_interval(z, z0, dz, nz)
    xbar = (x - x0 - i * dx) / dx
    ybar = (y - y0 - j * dy) / dy
    zbar = (z - z0 - k * dz) / dz

    zvec = cuda.local.array(16, dtype=np.float64)
    yvec = cuda.local.array(4, dtype=np.float64)

    for l in range(4):
        for m in range(4):
            for n in range(4):
                zvec[4 * l + m] = (
                    zvec[4 * l + m] * zbar
                    + coeffs_here[i, j, k, l * 16 + m * 4 + (3 - n)]  # rewrite this to not use a (shared) and to do a 1-d index?
                )

        for n in range(4):
            yvec[l] = yvec[l] * ybar + zvec[4 * l + (3 - n)]

    for n in range(4):
        result = result * xbar + yvec[3 - n]

    return result

@cuda.jit
def _evaluate_3d_spline_kernel(outp, x, y, z, coeffs, x0, dx, nx, y0, dy, ny, z0, dz, nz, ng, c_per_g):

    # NB: we're likely to pass in ~100 points in x,y,z here and evaluate ~ 1E4 amplitudes? 

    # For speed, we should assign shared memory.

    # Here we want to loop over blocks in the grid
    # this should use something like 
    # int start_block = blockIdx.y;
    # int block_inc = gridDim.y;

    # start: which "spline grid slice of blocks" we are on
    start_block = cuda.blockIdx.y
    # inc: length of the spline grid slice of blocks
    inc_block = cuda.gridDim.y

    for a in range(start_block, ng, inc_block):
        # if we are using shared memory, it should be filled here
        # good candidate for shared memory is the coefficients array as it is indexed all over the place by each thread block
        # by doing this we can avoid needing to pass "a"
        # instead the coefficients at "a" get loaded into shared memory here and passed to the device function
        # the coefficients array will have a dynamic size of coeffs.size per grid
        # check the numba documentation to see how to do this

        # dynamic shared memory, picks up from third argument of kernel invocation
        coeffs_here = cuda.shared.array(0, dtype=np.float64)

        # to fill it we loop over indices like these
        # threads should index NEIGHBOURING LOCATIONS IN MEMORY
        # int start_thread = threadIdx.x;
        # int thread_inc = blockDim.x;
        # stop = c_per_g

        #start: which thread we are on
        start_thread = cuda.threadIdx.x

        # inc: length of each thread block
        inc_thread = cuda.blockDim.x

        for fill_i in range(start_thread, c_per_g, inc_thread):
            coeffs_here[fill_i] = coeffs[a, fill_i]

        # then lastly we want to loop over threads in the block
        # this should use something like
        # int full_loop_start = threadIdx.x + blockDim.x * blockIdx.x;
        # int full_loop_inc = blockDim.x * gridDim.x;

        # start: the thread we are on in this block
        start_fill_ind = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x 

        # inc: the number of threads in the grid in this direction (each thread does one value)
        inc_fill_ind = cuda.blockDim.x * cuda.gridDim.x

        for i in range(start_fill_ind, x.size, inc_fill_ind):
            outp[a, i] = _compute_tricubic_spline_value(
                coeffs_here, 
                x[i], 
                x0, 
                dx, 
                nx, 
                y[i], 
                y0, 
                dy, 
                ny, 
                z[i], 
                z0, 
                dz, 
                nz
            )
        
