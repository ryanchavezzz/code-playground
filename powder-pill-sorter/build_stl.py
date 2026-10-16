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

def rot(axis, a):
    axis = axis / np.linalg.norm(axis); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K

def wall_frame():
    """The left outer wall is two flat triangles creased along B-C. Work in tri ABC's plane and
    fold anything past the crease onto tri BDC, so text wraps the crease without stretching."""
    A, B, C, D = map(np.array, ([-2.5, 0, 0], [-21, 0, 26.43], [-2.5, 90, 0], [-7, 90, 10.29]))
    r = np.array([0., -1, 0])                      # viewer outside on -x sees +y running leftward
    u = (B - A) / np.linalg.norm(B - A)            # up the slant, keel to rim
    n = np.cross(r, u)                             # outward normal of ABC
    ax = (C - B) / np.linalg.norm(C - B)
    side = lambda p: np.sign(np.dot(np.cross(p - B, ax), n))
    sA = side(A)
    # dihedral: unfold D into ABC's plane on the far side of the crease
    best = min((abs(np.dot(rot(ax, -a) @ (D - B) + B - A, n)), a) for a in np.linspace(-np.pi, np.pi, 72001)
               if side(rot(ax, -a) @ (D - B) + B) == -sA)
    phi = best[1]; R = rot(ax, phi)
    Dflat = rot(ax, -phi) @ (D - B) + B
    n2 = R @ n
    def fold(p, h=1.0):
        # Surface point folds rigidly; the offset normal blends n -> n2 across a 2h band so the
        # cutter's inner and outer skins stay continuous instead of tearing at the crease.
        w = np.dot(p - A, n); q = p - w * n
        s = -sA * side(q) * np.linalg.norm(np.cross(q - B, ax))      # >0 on the BDC side
        S = (R @ (q - B) + B) if s > 0 else q
        t = np.clip((s + h) / (2 * h), 0, 1); t = t * t * (3 - 2 * t)
        nn = (1 - t) * n + t * n2
        return S + w * nn / np.linalg.norm(nn)
    return dict(A=A, B=B, C=C, D=D, Dflat=Dflat, r=r, u=u, n=n, fold=fold, phi=np.degrees(phi))

def credit_cutter(lines, font, cap=4.0, depth=0.4, y0=3.0, v0=11.0, pitch=1.55):
    """Cutter for debossing `lines` (top to bottom) into the left outer wall, right-aligned at
    y0 from the pill end with the bottom baseline v0 mm up the slant (clear of the V seats)."""
    f = wall_frame()
    secs, base = [], 0.0
    for s in reversed(lines):
        cs = text_section(s, font, cap)
        secs.append(cs.translate([0, base])); base += cap * pitch
    cs = secs[0]
    for c in secs[1:]: cs = cs + c
    out = 1.0  # cutter pokes 1 mm outside the wall so the cut is clean
    m = mf.Manifold.extrude(cs, depth + out).translate([0, 0, -depth]).refine_to_length(0.5)
    O = f['A'] + y0 * np.array([0., 1, 0]) + v0 * f['u']
    M = np.stack([f['r'], f['u'], f['n']])
    def warp(P):
        Q = O + P @ M
        return np.array([f['fold'](q) for q in Q])
    return m.warp_batch(warp), f

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
report('base', base)
# The fin's underside is exactly coplanar with the rim top, and an exact-coplanar union leaves an
# internal zero-thickness sheet. Sinking the fin 0.01 mm gives the solver real overlap; the
# resulting sliver past the sloped outer wall is ~7 µm, far below anything a slicer resolves.
T, F = tomf(parts['trough']), tomf(parts['fin'])
CREDIT = ["Designed by Bill.", "Printed in California."]
# 0.4 mm deboss leaves >= 1.6 mm of the 2 mm wall; cap height 4 keeps strokes ~0.7 mm for a 0.4 nozzle
cut, _ = credit_cutter(CREDIT, sys.argv[3], cap=4.0, depth=0.4)
tr = frommf(T + F.translate([0, 0, -0.01]) - cut)
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
