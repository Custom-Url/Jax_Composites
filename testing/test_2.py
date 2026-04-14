"Generating a simple mesoscale mesh and loading elastically"

from typing import NamedTuple

import pyvista as pv
import os
import numpy as np
import gmsh

import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import Array
from jax_autovmap import autovmap

from tatva import Mesh, Operator, element
from tatva.element.base import Line2



def generate_mesh(length: float, width: float, ply_thickness: float, num_ply: int) -> Mesh:
    gmsh.initialize()
    gmsh.model.add("multi_ply")
    occ = gmsh.model.occ

    # Mesh base
    x0, x1 = 0.0, length
    y0, y1 = -width/2, width/2

    # Define the geometry
    p1 = occ.addPoint(x0, y0, 0)
    p2 = occ.addPoint(x1, y0, 0)
    p3 = occ.addPoint(x1, y1, 0)
    p4 = occ.addPoint(x0, y1, 0)


    l1 = occ.addLine(p1, p2)
    l2 = occ.addLine(p2, p3)
    l3 = occ.addLine(p3, p4)
    l4 = occ.addLine(p4, p1)

    cl = occ.addCurveLoop([l1, l2, l3, l4])
    surface = occ.addPlaneSurface([cl])

    occ.synchronize()
       
    # Create plies
    volumes = []
    current_surface = surface


    for i in range(num_ply):
        z_offset = i * ply_thickness

        # copy base surface
        s_copy = occ.copy([(2, surface)])
        occ.translate(s_copy, 0, 0, z_offset)

        out = occ.extrude(s_copy, 0, 0, ply_thickness)
        vol = [e[1] for e in out if e[0] == 3][0]
        volumes.append(vol)

    occ.synchronize()

    # Assign physical groups
    for i, vol in enumerate(volumes):
        gmsh.model.addPhysicalGroup(3, [vol], i + 1)
        gmsh.model.setPhysicalName(3, i + 1, f"ply_{i}")

    # Mesh
    gmsh.model.mesh.generate(3)

    # Extract mesh
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    tag_map = {tag: i for i, tag in enumerate(node_tags)}
    nodes = jnp.array(node_coords).reshape(-1, 3)


    # Extract elements per ply
    all_elements = []
    all_ply_ids = []

    for i, vol in enumerate(volumes):
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3, vol)

        elems = np.array(elem_node_tags[0]).reshape(-1, 4)

        # Convert node tags → indices
        elems = np.vectorize(tag_map.get)(elems)

        all_elements.append(elems)
        all_ply_ids.append(np.full((elems.shape[0],), i))

    elements = jnp.array(np.vstack(all_elements))
    ply_ids = jnp.array(np.concatenate(all_ply_ids))

    gmsh.finalize()

    return nodes, elements, ply_ids

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


#################################################
# Run Mesh Generation
#################################################

# Define mesh
lx = 5.0
ly = 2.0
lz = 1.0
plies = 2

nodes, elements, ply_ids = generate_mesh(lx, ly, lz, plies)

plot_mesh(nodes, elements, ply_ids)
