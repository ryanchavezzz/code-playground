# Rebuild the sorter STLs from the mesh data embedded in the spec page.
# Usage: python3 build_stl.py <page.html> <out_dir> <BarlowCondensed-SemiBold.ttf>
# Needs: numpy trimesh manifold3d scipy networkx matplotlib
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
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

def text_section(s, font, cap):
    """CrossSection of s, baseline at y=0, ending at x=0 (right-aligned), cap height `cap` mm."""
    fp = FontProperties(fname=font)
    k = cap / (TextPath((0, 0), "H", size=100, prop=fp).get_extents().height / 100)
    p = TextPath((0, 0), s, size=k, prop=fp)
    rings = [r[:-1] for r in p.to_polygons(closed_only=True) if len(r) > 3]
    cs = mf.CrossSection([np.asarray(r, np.float64) for r in rings], mf.FillRule.EvenOdd)
    x1 = cs.bounds()[2]
    return cs.translate([-x1, 0])

def foot_cutter(text, side, font, cap=3.8, depth=0.5):
    """Deboss one line into the top (z=3) of a skid, between the legs (y 22..58), reading from
    that skid's outer side with the baseline toward the outer edge. side=-1 left, +1 right."""
    cs = text_section(text, font, cap)
    x0, y0, x1, y1 = cs.bounds()
    cs = cs.translate([-(x0 + x1) / 2, -(y0 + y1) / 2])   # centre on glyph extents, descenders included
    assert x1 - x0 <= 34 and y1 - y0 <= 4.6, (text, x1 - x0, y1 - y0)
    m = mf.Manifold.extrude(cs, depth + 1).translate([0, 0, -depth])
    # text frame (right, up, normal) -> world; right x up = +z so it reads from above, not mirrored
    r, u = np.array([0., side, 0]), np.array([-side, 0., 0])
    M = np.stack([r, u, [0, 0, 1]])
    return m.warp_batch(lambda P: np.array([side * 20, 40, 3]) + P @ M)

def tomf(m): return mf.Manifold(mf.Mesh64(np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.uint64)))
def frommf(x):
    # float64 end to end: float32 snaps near-coincident crease vertices together and breaks manifoldness
    o = x.to_mesh64(); return trimesh.Trimesh(o.vert_properties[:, :3], o.tri_verts)
def union(ms):
    acc = tomf(ms[0])
    for m in ms[1:]: acc = acc + tomf(m)
    return frommf(acc)
def report(name, m):
    print(f"{name}: watertight={m.is_watertight} vol={m.volume:.1f} bodies={len(m.split())} bounds={m.bounds.round(2).tolist()}")
# base: legs and skids already overlap, so a plain union fuses them
base = union(list(parts['base'].split()))
# Credit on the left foot: 0.5 mm deep in the 3 mm skid; caps at 3.8 fill the 34 mm between the legs
base = frommf(tomf(base) - foot_cutter("DESIGNED BY BILL", -1, sys.argv[3]))
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
