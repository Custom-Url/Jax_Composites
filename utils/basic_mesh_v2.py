''' Basic 3D multi-ply mesh generation WITHOUT GMESH

#################################################
# Run Mesh Generation
#################################################
# Define mesh
lx = 5.0
ly = 1.0
lz = 0.5
plies = 2

nx = 50
ny = 10
nz = 5

mesh, nodes, elements, ply_ids = generate_mesh(lx, ly, lz, plies, nx, ny, nz)
'''

import gmsh

from tatva import Mesh

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
import pyvista as pv


def generate_mesh(length, width, ply_thickness, num_ply, nx, ny, nz_per_ply):
    import numpy as np
    import jax.numpy as jnp

    # Total elements in z
    nz = num_ply * nz_per_ply

    # Coordinate ranges
    x_rng = np.linspace(0.0, length, nx + 1)
    y_rng = np.linspace(-width / 2, width / 2, ny + 1)
    z_rng = np.linspace(0.0, num_ply * ply_thickness, nz + 1)

    # Structured grid
    Z, Y, X = np.meshgrid(z_rng, y_rng, x_rng, indexing="ij")
    nodes = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # Strides
    stride_x = 1
    stride_y = nx + 1
    stride_z = (nx + 1) * (ny + 1)

    # Cell indices
    k_idx, j_idx, i_idx = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )

    n0 = (i_idx * stride_x + j_idx * stride_y + k_idx * stride_z).flatten()

    # Hex nodes
    n1 = n0 + stride_x
    n2 = n0 + stride_y
    n3 = n2 + stride_x
    n4 = n0 + stride_z
    n5 = n4 + stride_x
    n6 = n4 + stride_y
    n7 = n6 + stride_x

    # 6-tet decomposition (same as your original)
    t1 = jnp.stack([n0, n1, n3, n7], axis=-1)
    t2 = jnp.stack([n0, n1, n7, n5], axis=-1)
    t3 = jnp.stack([n0, n5, n7, n4], axis=-1)
    t4 = jnp.stack([n0, n3, n2, n7], axis=-1)
    t5 = jnp.stack([n0, n2, n6, n7], axis=-1)
    t6 = jnp.stack([n0, n6, n4, n7], axis=-1)

    all_tets = jnp.stack([t1, t2, t3, t4, t5, t6], axis=1)
    elements = all_tets.reshape(-1, 4)

    # -------------------------
    # Ply IDs
    # -------------------------
    # Each hex layer corresponds to a ply index
    ply_per_layer = np.repeat(
        np.arange(num_ply),
        nz_per_ply
    )

    # Map each cell (k index) to ply
    ply_ids_cells = ply_per_layer[k_idx.flatten()]

    # Each hex → 6 tets
    ply_ids = np.repeat(ply_ids_cells, 6)

    # Convert to JAX
    nodes = jnp.array(nodes)
    elements = jnp.array(elements)
    ply_ids = jnp.array(ply_ids)

    mesh = Mesh(coords=nodes, elements=elements)

    return mesh, nodes, elements, ply_ids

def plot_mesh(nodes, elements, ply_ids):
    nodes_np = np.array(nodes)
    elements_np = np.array(elements)
    ply_ids_np = np.array(ply_ids)

    # PyVista cell format
    cells = np.hstack([
        np.full((elements_np.shape[0], 1), 4),
        elements_np
    ]).flatten()

    celltypes = np.full(elements_np.shape[0], pv.CellType.TETRA)

    grid = pv.UnstructuredGrid(cells, celltypes, nodes_np)
    grid.cell_data["ply"] = ply_ids_np

    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="ply", show_edges=True)
    plotter.add_axes()
    plotter.show()
