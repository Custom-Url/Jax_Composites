"Testing a linear elasticity simulation for a composite ply"

from typing import NamedTuple

import pyvista as pv
import os
import numpy as np

import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import Array
from jax_autovmap import autovmap

from tatva import Operator, element
from tatva.element.base import Line2

class TatvaMeshWrapper:
    def __init__(self, pv_mesh, elements):
        coords = np.array(pv_mesh.points)

        # FORCE 2D
#        coords = coords[:, :2]

        self.coords = jnp.array(coords)
        self.elements = jnp.array(elements)

class Material(NamedTuple):
    """Material properties for the elasticity operator."""

    mu: float  
    lmbda: float

    @classmethod
    def from_youngs_poisson_2d(
        cls, E: float, nu: float, plane_stress: bool = False
    ) -> "Material":
        mu = E / 2 / (1 + nu)
        if plane_stress:
            lmbda = 2 * nu * mu / (1 - nu)
        else:
            lmbda = E * nu / (1 - 2 * nu) / (1 + nu)
        return cls(mu=mu, lmbda=lmbda)

@autovmap(grad_u=2)
def compute_strain(grad_u):
    return 0.5 * (grad_u + grad_u.T)


@autovmap(eps=2, mu=0, lmbda=0)
def compute_stress(eps, mu, lmbda):
    return 2 * mu * eps + lmbda * jnp.trace(eps) * jnp.eye(2)


@autovmap(grad_u=2, mu=0, lmbda=0)
def strain_energy_density(grad_u, mu, lmbda):
    eps = compute_strain(grad_u)
    sigma = compute_stress(eps, mu, lmbda)
    return 0.5 * jnp.einsum("ij,ij->", sigma, eps)

#@jax.jit
#def total_energy_full(u_flat: Array) -> Array:
#    """Compute the total energy of the system."""
#    u = u_flat.reshape(-1, 2)
#    u_grad = op.grad(u)
#    e_density = strain_energy_density(u_grad, mat.mu, mat.lmbda)
#    return op.integrate(e_density)
#
#@jax.jit
#def total_energy(u_free: Array) -> Array:
#    """Compute the total energy of the system."""
#    u_full = jnp.zeros(n_dofs).at[free_dofs].set(u_free)
#    return total_energy_full(u_full)



###################################################################
# Problem Solution
###################################################################

# Read geometery from .vtk file
mesh_dir = "/mnt/data/dg765/FFT/micr/code_python/EXP"
file_name = "2D_Fibre_Distribution_Alpha.vtk"

mesh_path = os.path.join(mesh_dir, file_name)

pv_mesh = pv.read(mesh_path)

nx, ny, nz = pv_mesh.dimensions
nx -= 1
ny -= 1

elements = []

for j in range(ny):
    for i in range(nx):
        n0 = j * (nx + 1) + i
        n1 = n0 + 1
        n2 = n0 + (nx + 1)
        n3 = n2 + 1

        elements.append([n0, n1, n3, n2])

elements = np.array(elements)

nodes = pv_mesh.points

# Read materials from .vtk file
mat_id = pv_mesh.cell_data["geom"]

# Define material params
mat_matrix = Material.from_youngs_poisson_2d(1, 0.2)
mat_fibre = Material.from_youngs_poisson_2d(10, 0.3)

mu_field = jnp.where(mat_id == 1, mat_fibre.mu, mat_matrix.mu)
lmbda_field = jnp.where(mat_id == 1, mat_fibre.lmbda, mat_matrix.lmbda)

# expand to quadrature level
mu_q = mu_field[:, None]
lmbda_q = lmbda_field[:, None]

mesh = TatvaMeshWrapper(pv_mesh, elements)

# FEM operator 
quad = element.Quad4()
op = Operator(mesh, quad)

# DOFs
n_dofs_per_node = 2
n_dofs = mesh.coords.shape[0] * n_dofs_per_node

# Boundaries
x_max = jnp.max(mesh.coords[:, 0])
x_min = jnp.min(mesh.coords[:, 0])

fixed_nodes = jnp.where(jnp.isclose(mesh.coords[:, 0], x_min))[0]
load_nodes = jnp.where(jnp.isclose(mesh.coords[:, 0], x_max))[0]


fixed_dofs = jnp.concatenate([
    fixed_nodes * n_dofs_per_node,
    fixed_nodes * n_dofs_per_node +1
])

load_dofs = load_nodes * n_dofs_per_node

prescribed_dofs = jnp.unique(jnp.concatenate([fixed_dofs, load_dofs]))
free_dofs = jnp.setdiff1d(jnp.arange(n_dofs), prescribed_dofs)




quad_points_in_physical_space = op.eval(mesh.coords)

print(quad_points_in_physical_space)
print("Shape of the quadrature points in physical space: ", quad_points_in_physical_space.shape)
print(mesh.elements.shape)
print(mesh.elements[0])


fig, ax = plt.subplots(figsize=(3, 3), layout="constrained")

# -------------------------
# 1. plot quad mesh correctly
# -------------------------
for e in mesh.elements:
    pts = mesh.coords[e]

    # close the quad
    pts = np.vstack([pts, pts[0]])

    ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=0.5)

# -------------------------
# 2. plot quadrature points
# -------------------------
qp = quad_points_in_physical_space.reshape(-1, 3)

ax.scatter(
    qp[:, 0],
    qp[:, 1],
    c="red",
    s=20,
    marker="x",
    label="quadrature points"
)

# -------------------------
# 3. aesthetics
# -------------------------
ax.set_aspect("equal")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.margins(0.0)
ax.legend()
plt.show()

