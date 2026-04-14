"Generating a simple mesoscale mesh and loading elastically"

import sys

sys.path.append("/mnt/data/dg765/Jax/JaxDG/utils")

from basic_mesh import generate_mesh, plot_mesh 

# Define mesh
lx = 5.0
ly = 2.0
lz = 1.0
plies = 2

nodes, elements, ply_ids = generate_mesh(lx, ly, lz, plies)

plot_mesh(nodes, elements, ply_ids)
