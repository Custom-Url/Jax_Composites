import jax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)  # use double-precision
import jax.numpy as jnp
from tatva import Mesh, Operator, element

mesh = Mesh.unit_square(n_x=1, n_y=1,  type="triangle", dim=2)
print("Coordinates of the nodes in the mesh: ", mesh.coords)
print("Connectivity of the elements in the mesh: ", mesh.elements)

tri = element.Tri3()
op = Operator(mesh, tri)

quad_points = op.eval(mesh.coords)

print(quad_points)
print("quad_points shape:", quad_points.shape)


quad_points = quad_points.squeeze()



plt.figure(figsize=(3, 3), layout="constrained")
ax = plt.axes()
ax.tripcolor(
    *mesh.coords.T,
    mesh.elements,
    color="gray",
    lw=0.1,
    facecolors=jnp.ones(mesh.elements.shape[0]),
    cmap="managua_r",
)
ax.scatter(
    quad_points[:, 0],
    quad_points[:, 1],
    c='tab:red',
    s=20,
    marker="x",
)
ax.set_aspect("equal")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.margins(0.0, 0.0)
plt.show()
