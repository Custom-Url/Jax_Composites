"Generating a simple mesoscale mesh and loading elastically"

import time
import sys
sys.path.append("/mnt/data/dg765/Jax/JaxDG/utils")

from basic_mesh import generate_mesh, plot_mesh 

from tatva import Operator, element, sparse

import pyvista as pv
import numpy as np
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax_autovmap import autovmap
jax.config.update("jax_enable_x64", True)

from petsc4py import PETSc

class Material(NamedTuple):
    mu: float
    lmbda: float

@jax.jit
def total_energy(u_free):
    u_full = jnp.zeros(n_dofs).at[free_dofs].set(u_free)
    u_full = u_full.at[load_dofs].set(applied_u_load)
    u = u_full.reshape(-1, n_dofs_per_node)
    u_grad = op.grad(u)
    psi = neo_hookean_density(u_grad, mat.mu, mat.lmbda)
    return op.integrate(psi)

def snes_jacobian(snes, x, J, P):
    u = jnp.array(x.array_r)

    K_sparse = hessian_fn(u)
    J.zeroEntries()
    J.setValuesCSR(
        np.asarray(K_sparse.indptr, dtype="int32"),
        np.asarray(K_sparse.indices, dtype="int32"),
        np.asarray(K_sparse.data, dtype="float64"),
    )
    J.assemblyBegin()
    J.assemblyEnd()

    return PETSc.Mat.Structure.SAME_NONZERO_PATTERN


def snes_residual(snes, x, f):
    u = jnp.array(x.array_r)
    f.array = np.array(residual_fn(u))

@autovmap(grad_u=2)
def compute_deformation_gradient(grad_u):
    I = jnp.eye(3)
    F = I + grad_u
    return F


@autovmap(grad_u=2, mu=0, lmbda=0)
def neo_hookean_density(grad_u, mu, lmbda):
    F = compute_deformation_gradient(grad_u)
    J = jnp.linalg.det(F)
    C = F.T @ F
    I1 = jnp.trace(C)
    return (mu / 2) * (I1 - 3 - 2 * jnp.log(J)) + (lmbda / 2) * (jnp.log(J)) ** 2

plotting = False 

#######################################################
# Create Mesh and Operator
#######################################################
lx = 10.0
ly = 1.0
lz = 0.5
plies = 2

mesh, nodes, elements, ply_ids = generate_mesh(lx, ly, lz, plies)

if plotting:
    plot_mesh(nodes, elements, ply_ids)

tet_elem = element.Tetrahedron4()
op = Operator(mesh, tet_elem)
mat = Material(mu=500.0, lmbda=1000.0)

n_dofs_per_node = 3
n_nodes = len(nodes)
n_dofs = n_nodes * n_dofs_per_node

print(f"Nodes: {n_nodes}    DOFs: {n_dofs}  Elements: {len(elements)}")

#######################################################
# Define Boundary Conditions
#######################################################

x_min, x_max = jnp.min(mesh.coords[:, 0]), jnp.max(mesh.coords[:, 0])
fixed_nodes = jnp.where(jnp.isclose(mesh.coords[:, 0], x_min))[0]
load_nodes = jnp.where(jnp.isclose(mesh.coords[:, 0], x_max))[0]

fixed_dofs = jnp.concatenate(
    [fixed_nodes * 3, fixed_nodes * 3 + 1, fixed_nodes * 3 + 2]
)
load_dofs = load_nodes * 3+2  # Apply load in z-direction
prescribed_dofs = jnp.unique(jnp.concatenate([fixed_dofs, load_dofs]))
free_dofs = jnp.setdiff1d(jnp.arange(n_dofs), prescribed_dofs)

applied_u_load = 1.0

#######################################################
# Sparse Jacobian Assembly using colouring
#######################################################

residual_fn = jax.jit(jax.grad(total_energy))

sparsity_pattern = sparse.create_sparsity_pattern(mesh, n_dofs_per_node=3)
reduced_sparsity = sparse.reduce_sparsity_pattern(sparsity_pattern, free_dofs)
colored_matrix = sparse.ColoredMatrix.from_csr(reduced_sparsity)

hessian_fn = sparse.jacfwd(
    fn=residual_fn, colored_matrix=colored_matrix, color_batch_size=int(colored_matrix.colors.max()) + 1
)
hessian_fn = jax.jit(hessian_fn)

#######################################################
# Solve System using PETSc SNES
#######################################################
snes = PETSc.SNES().create(comm=PETSc.COMM_SELF)
opts = PETSc.Options()
opts["snes_monitor"] = None
snes.setFromOptions()

x_sol = PETSc.Vec().createSeq(len(free_dofs))
f_res = PETSc.Vec().createSeq(len(free_dofs))
snes.setFunction(snes_residual, f_res)

J = PETSc.Mat().createAIJ([len(free_dofs), len(free_dofs)], comm=PETSc.COMM_SELF)
J.setPreallocationCSR((reduced_sparsity.indptr, reduced_sparsity.indices))
J.setUp()

snes.setJacobian(snes_jacobian, J, J)

snes.setType("newtonls")
ksp = snes.getKSP()
pc = ksp.getPC()

ksp.setType("preonly")
pc.setType("lu")

start_time = time.time()
snes.solve(None, x_sol)
end_time = time.time()
elapsed = end_time - start_time

print(f"Solve completed in {elapsed:.2f} seconds")
print(f"SNES iterations: {snes.getIterationNumber()}")

#######################################################
# Visualisation
#######################################################

cells = np.hstack([np.full((mesh.elements.shape[0], 1), 4), mesh.elements]).flatten()
cell_types = np.full(mesh.elements.shape[0], pv.CellType.TETRA)
grid = pv.UnstructuredGrid(cells, cell_types, np.array(mesh.coords, dtype=np.float64))

u_current = x_sol.array_r.copy()
u_full = jnp.zeros(n_dofs)
u_full = u_full.at[free_dofs].set(u_current)
u_full = u_full.at[load_dofs].set(applied_u_load)
u_current = u_full.reshape(-1, n_dofs_per_node)

grid.point_data["displacement"] = np.array(u_current)

grad_u = op.grad(u_current)
grid = grid.warp_by_vector("displacement", factor=2.0)
plotter = pv.Plotter()
plotter.add_mesh(
    grid,
    show_edges=True,
    scalars="displacement",
    component=2,
    cmap="managua",
)
plotter.add_axes()
plotter.show()
