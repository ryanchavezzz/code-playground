# Rebuild the sorter STLs from the mesh data embedded in the spec page.
# Usage: python3 build_stl.py <page.html> <out_dir>   (needs: numpy trimesh manifold3d scipy networkx)
import re, json, sys, numpy as np, trimesh
html = open(sys.argv[1]).read()
src = re.search(r'const MESH = (\{.*?\});', html, re.S).group(1)
src = re.sub(r'(\w+):', r'"\1":', src)
M = {k: np.array(v, float).reshape(-1, 3, 3) for k, v in json.loads(src).items()}
def mk(name):
    t = M[name]
    m = trimesh.Trimesh(vertices=t.reshape(-1, 3), faces=np.arange(len(t) * 3).reshape(-1, 3), process=True)
    m.merge_vertices(digits_vertex=3)
    trimesh.repair.fix_normals(m)
    print(f"{name}: tris={len(m.faces)} watertight={m.is_watertight} winding={m.is_winding_consistent} vol={m.volume:.1f} bodies={len(m.split(only_watertight=False))} bounds={m.bounds.round(2).tolist()}")
    return m
parts = {k: mk(k) for k in M}
import manifold3d as mf
def tomf(m): return mf.Manifold(mf.Mesh(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32)))
def frommf(x):
    o = x.to_mesh(); return trimesh.Trimesh(o.vert_properties[:, :3], o.tri_verts)
def union(ms):
    acc = tomf(ms[0])
    for m in ms[1:]: acc = acc + tomf(m)
    return frommf(acc)
def report(name, m):
    print(f"{name}: watertight={m.is_watertight} vol={m.volume:.1f} bodies={len(m.split())} bounds={m.bounds.round(2).tolist()}")
# base: legs and skids already overlap, so a plain union fuses them
base = union(list(parts['base'].split()))
report('base', base)
# The fin's underside is exactly coplanar with the rim top, and an exact-coplanar union leaves an
# internal zero-thickness sheet. Sinking the fin 0.01 mm gives the solver real overlap; the
# resulting sliver past the sloped outer wall is ~7 µm, far below anything a slicer resolves.
T, F = tomf(parts['trough']), tomf(parts['fin'])
tr = frommf(T + F.translate([0, 0, -0.01]))
report('trough+fin', tr)
def overhangs(name, m):
    n = m.face_normals; c = m.triangles_center
    bad = (n[:, 2] < -np.cos(np.radians(45)) + 1e-6) & (c[:, 2] > 0.01)
    worst = np.degrees(np.arcsin(np.clip(-n[bad, 2], 0, 1))).max() if bad.any() else 0
    print(f"{name}: overhang faces >45° = {bad.sum()}, area {m.area_faces[bad].sum():.1f} mm², worst {worst:.0f}° below horizontal-ish; z of them {np.unique(c[bad,2].round(1))[:10]}")
overhangs('trough', tr); overhangs('base', base)
import os
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
tr.export(f"{out}/sorter_trough.stl"); base.export(f"{out}/sorter_base.stl")
for f in ("sorter_trough.stl", "sorter_base.stl"):
    r = trimesh.load(f"{out}/{f}"); print(f, os.path.getsize(f"{out}/{f}"), "B, watertight", r.is_watertight, "vol", round(r.volume,1))
