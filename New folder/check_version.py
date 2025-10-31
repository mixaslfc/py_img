import gco, inspect, sys, os

print("module:", gco)
print("file:", gco.__file__)
print("Python:", sys.version)

print("\ndir(gco):")
print([a for a in dir(gco) if "graph" in a.lower()])

print("\nhasattr(gco, 'cut_general_graph'):", hasattr(gco, "cut_general_graph"))
print("hasattr(gco, 'cut_grid_graph_simple'):", hasattr(gco, "cut_grid_graph_simple"))
print("hasattr(gco, 'GCO'):", hasattr(gco, "GCO"))

src = inspect.getsource(gco.cut_general_graph)
print("\n---- cut_general_graph source (first 40 lines) ----")
print("\n".join(src.splitlines()[:40]))
