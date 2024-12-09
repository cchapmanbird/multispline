from numba import cuda, njit
import numpy as np
import cupy as cp

class AmpInterp3dGPU:
    def __init__(self, coeffs, x0, dx, nx, y0, dy, ny, z0, dz, nz):
        breakpoint()
        
        coeff_temp = coeffs.reshape(coeffs.shape[0], -1)

        self.coeffs = cuda.to_device(coeff_temp)

        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.ng = self.coeffs.shape[0]
        self.grid_shape = self.coeffs.shape[1]

    def evaluate_3d_spline(self, x, y, z, out=None):
        if out is None:
            out = cp.zeros(self.ng * x.size)

        x_in = cuda.to_device(x.flatten())
        y_in = cuda.to_device(y.flatten())
        z_in = cuda.to_device(z.flatten())

        # organisation:
            # Each block corresponds to one trajectory point
            # We may be able to adjust threadsperblock based on the number of requested modes (rounded to mult of 32)? Set to 64 for now.
            # blockspergrid is set based on the number of trajectory points (x.size)

            # coefficients array should store things so that the special index is f(i,j,k) + mode_num + [0,64]*nmodes
            # shared array size = threadsperblock * 64 * 8

        threadsperblock = 64
        blockspergrid = ((x.size + (threadsperblock - 1)) // threadsperblock, self.ng // threadsperblock) # y-axis is mode overflow
        print("BPG:", blockspergrid)
                # _evaluate_3d_spline_kernel[blockspergrid, threadsperblock](
        #_evaluate_3d_spline_kernel[blockspergrid, threadsperblock, 0, int(8*self.grid_shape)](
        _evaluate_3d_spline_kernel[blockspergrid, threadsperblock, 0, int( 64 * threadsperblock * 8)](
            out,
            x_in, 
            y_in, 
            z_in, 
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
            x_in.size,
            )

        return out.reshape(self.ng, *x.shape).get()

@njit
def get_interval(a, a0, da, na):
    idx = int((a - a0) / da)
    if idx < 0:
        return 0
    elif idx >= na:
        return na - 1
    return idx   

# @njit
# def get_interval(a, a0, da, na):
#     return int((a - a0) / da) 

@njit
def _compute_tricubic_spline_value(coeffs_here, xbar, x0, dx, nx, ybar, y0, dy, ny, zbar, z0, dz, nz):
    result = 0.

    zvec = cuda.local.array(16, dtype=np.float64)
    yvec = cuda.local.array(4, dtype=np.float64)

    for l in range(4):
        for m in range(4):
            zvec[4 * l + m] = 0.
            for n in range(4):
                zvec[4 * l + m] = (
                    zvec[4 * l + m] * zbar
                    + coeffs_here[l*16 + m*4 + (3 - n)]
                    # + coeffs_here[i, j, k, l * 16 + m * 4 + (3 - n)]
                    
                )
        yvec[l] = 0.
        for n in range(4):
            yvec[l] = yvec[l] * ybar + zvec[4 * l + (3 - n)]

    for n in range(4):
        result = result * xbar + yvec[3 - n]

    return result

@cuda.jit
def _evaluate_3d_spline_kernel(outp, x, y, z, coeffs, x0, dx, nx, y0, dy, ny, z0, dz, nz, ng, nvec):

    vec_ind = cuda.blockIdx.x

    if vec_ind < nvec:  # avoid out-of-bounds indexing

        coeffs_shared = cuda.shared.array(0, dtype=np.float64)
        
        mode_ind = cuda.threadIdx.x
        mode_ind_on_block = mode_ind + cuda.blockDim.y * cuda.blockInd.y # handle the mode blocks on the y axis

        x_here = x[vec_ind]
        y_here = y[vec_ind]
        z_here = z[vec_ind]

        i_here = get_interval(x_here, x0, dx, nx)
        j_here = get_interval(y_here, x0, dx, nx)
        k_here = get_interval(z_here, x0, dx, nx)

        xbar = (x_here - x0 - i_here * dx) / dx
        ybar = (y_here - y0 - j_here * dy) / dy
        zbar = (z_here - z0 - k_here * dz) / dz

        i_here = i_here * nx
        j_here = j_here * ny
        k_here = k_here * nz

        ind_loc = (i_here*nx*ny*nz + j_here*ny*nz + k_here*nz)*64*ng
        shared_fill_ind = ind_loc + mode_ind_on_block

        # sync before loading up
        cuda.syncthreads()

        # now load the coefficients at this (i, j, k) into shared memory
        # index is at (i,j,k) + mode_ind and loops over 64 elements strided by nmodes
        for coeff_fill_ind in range(64):
            coeffs_shared[coeff_fill_ind] = coeffs[shared_fill_ind + ng * coeff_fill_ind]

        # sync again to ensure shared memory fill is complete
        cuda.syncthreads()

        # output fill neigbouring parts: vec_ind * nmodes + mode_ind
        if mode_ind < ng:  # block overflow of threads per block (last block will not be full)
            fill_ind = vec_ind * ng + mode_ind_on_block
            outp[fill_ind] = _compute_tricubic_spline_value(
                coeffs_shared,
                xbar, 
                x0, 
                dx, 
                nx, 
                ybar, 
                y0, 
                dy, 
                ny, 
                zbar, 
                z0, 
                dz, 
                nz
            )
        
