import numpy as np


def fast_sweep_2d(grid, fixed_cells, obstacle, f, dh, iterations=5):
    # this is used for padding the outer boundaries of the domain,
    # so that the min() operations in the upwind scheme choose the inner point.
    large_val = 1e3
    nx, ny = grid.shape
    # 4 directions to sweep along - the range parameters for x and y.
    sweep_dirs = [
        (0, nx, 1, 0, ny, 1),  # Top-left to bottom-right
        (nx - 1, -1, -1, 0, ny, 1),  # Top-right to bottom-left
        (nx - 1, -1, -1, ny - 1, -1, -1),  # Bottom-right to top-left
        (0, nx, 1, ny - 1, -1, -1),  # Bottom-left to top-right
    ]

    # pad with a large value to properly handle boundary conditions in the upwind scheme.
    padded = np.pad(grid, pad_width=1, mode="constant", constant_values=large_val)

    for _ in range(iterations):
        for x_start, x_end, x_step, y_start, y_end, y_step in sweep_dirs:
            for iy in range(y_start, y_end, y_step):
                for ix in range(x_start, x_end, x_step):
                    # dont do anything for fixed cells (interface) or obstacles
                    if fixed_cells[iy, ix] or obstacle[iy, ix]:
                        continue
                    # calculate a,b from eqn 2.3 of Zhao et.al
                    py, px = iy + 1, ix + 1
                    # since it's a padded array and boundary+1 is a large value,
                    # it will choose the interior value at the end, acting like one sided difference.
                    a = np.min((padded[py, px - 1], padded[py, px + 1]))
                    b = np.min((padded[py - 1, px], padded[py + 1, px]))
                    # explicit unique solution to eq 2.3, given by eq 2.4
                    xbar = (
                        large_val  # xbar will be the distance to this cell from front
                    )
                    if np.abs(a - b) >= f * dh:
                        xbar = np.min((a, b)) + f * dh
                    else:
                        # can add small eps to sqrt later for stability
                        xbar = (a + b + np.sqrt(2 * (f * dh) ** 2 - (a - b) ** 2)) / 2
                    # update if new distance is smaller
                    padded[py, px] = np.min((padded[py, px], xbar))
    # return un-padded array
    return padded[1:-1, 1:-1]


# # dh is the grid spacing
# start = time.time()
# out = fast_sweep_2d(dist_grid_np, interface_mask, obstacle_mask, dh=dx, iterations=5)
# print(f"Fast sweep took {time.time() - start}s")
# out[obstacle_mask.astype(bool)] = np.nan

# plt.pcolormesh(X_np, Y_np, out)
# plt.colorbar()
# plt.contour(X_np, Y_np, out, levels=[0], colors="red")
# plt.contour(X_np, Y_np, out, levels=[0, 0.1, 0.2, 0.3])
# plt.gca().set_aspect(1)
