import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=["iterations"])
def fast_sweep_2d(grid, fixed_cells, obstacle, f, dh, iterations=5):
    large_val = 1e3
    ny, nx = grid.shape
    sweep_dirs = [
        (0, nx, 1, 0, ny, 1),  # Top-left to bottom-right
        (nx - 1, -1, -1, 0, ny, 1),  # Top-right to bottom-left
        (nx - 1, -1, -1, ny - 1, -1, -1),  # Bottom-right to top-left
        (0, nx, 1, ny - 1, -1, -1),  # Bottom-left to top-right
    ]
    frozen = jnp.logical_or(fixed_cells, obstacle)
    padded = jnp.pad(grid, pad_width=1, mode="constant", constant_values=large_val)

    def run_sweep(sweep_dir, grid):
        x_start, x_end, x_step, y_start, y_end, y_step = sweep_dir

        def y_loop_body(iy, grid):
            def x_loop_body(ix, grid):
                piy, pix = iy + 1, ix + 1
                a = jnp.minimum(grid[piy, pix - 1], grid[piy, pix + 1])
                b = jnp.minimum(grid[piy - 1, pix], grid[piy + 1, pix])
                updated_val = jnp.where(
                    frozen[iy, ix],
                    grid[piy, pix],  # no change if frozen
                    jnp.minimum(  # min of curr and updated val
                        grid[piy, pix],
                        jnp.where(  # eqn 2.4
                            jnp.abs(a - b) >= f * dh,
                            jnp.minimum(a, b) + f * dh,
                            (a + b + jnp.sqrt(2 * (f * dh) ** 2 - (a - b) ** 2)) / 2,
                        ),
                    ),
                )
                return grid.at[piy, pix].set(updated_val)

            x_indices = jnp.arange(x_start, x_end, x_step)
            return jax.lax.fori_loop(
                0,
                len(x_indices),
                # ix is 0..len(x_indices) - we need to map it to actual range
                lambda ix, grid: x_loop_body(x_indices[ix], grid),
                grid,
            )

        y_indices = jnp.arange(y_start, y_end, y_step)
        return jax.lax.fori_loop(
            0,
            len(y_indices),
            lambda iy, grid: y_loop_body(y_indices[iy], grid),
            grid,
        )

    def iteration_body(_, cur_grid):
        # perform 4 sweeps (2 dimentions)
        grid_s1 = run_sweep(sweep_dirs[0], cur_grid)
        grid_s2 = run_sweep(sweep_dirs[1], grid_s1)
        grid_s3 = run_sweep(sweep_dirs[2], grid_s2)
        grid_s4 = run_sweep(sweep_dirs[3], grid_s3)
        return grid_s4

    final_grid = jax.lax.fori_loop(0, iterations, iteration_body, padded)
    return final_grid[1:-1, 1:-1]


# # dh is the grid spacing
# start = time.time()
# out = fast_sweep_2d(dist_grid_np, interface_mask, obstacle_mask, dh=dx, iterations=5)
# print(f"Fast sweep took {time.time() - start}s")

# start = time.time()
# out = fast_sweep_2d(dist_grid_np, interface_mask, obstacle_mask, dh=dx, iterations=5)
# print(f"Fast sweep took {time.time() - start}s")

# out = np.array(out)
# out[obstacle_mask.astype(bool)] = np.nan

# plt.pcolormesh(X_np, Y_np, out)
# plt.colorbar()
# plt.contour(X_np, Y_np, out, levels=[0], colors="red")
# plt.contour(X_np, Y_np, out, levels=[0, 0.1, 0.2, 0.3])
# plt.gca().set_aspect(1)
