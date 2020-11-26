import ompy as om
mat = om.Matrix(path="1Gen_3He.m")
std = om.Matrix(path="1Gen_3He_std.m")

rel = std/mat
rel_larger0 = rel[rel.values>0]
print(rel_larger0.min(), rel_larger0.max())
