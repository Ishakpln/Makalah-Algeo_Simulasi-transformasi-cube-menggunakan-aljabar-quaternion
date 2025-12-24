import math
import os
import shutil
import sys
from typing import List, Tuple

Quat = Tuple[float, float, float, float]
Vec3 = Tuple[float, float, float]

def qNorm(q: Quat) -> Quat:
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return (1.0, 0.0, 0.0, 0.0)
    return (w/n, x/n, y/n, z/n)

def qMul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    )

def qConj(q: Quat) -> Quat:
    w, x, y, z = q
    return (w, -x, -y, -z)

def q_from_axis_angle(axis: Vec3, angle_rad: float) -> Quat:
    ax, ay, az = axis
    n = math.sqrt(ax*ax + ay*ay + az*az)
    if n == 0.0 or angle_rad == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    ax, ay, az = ax/n, ay/n, az/n
    half = angle_rad * 0.5
    s = math.sin(half)
    return qNorm((math.cos(half), ax*s, ay*s, az*s))

def q_rotate(q: Quat, v: Vec3) -> Vec3: 
    vx, vy, vz = v
    p = (0.0, vx, vy, vz)                

    r = qMul(qMul(q, p), qConj(q))       
    w, x, y, z = r                        

    return (x, y, z)                      



# Fungsi helper vector
def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )

def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


# Projection 3D to 2D
def project(v: Vec3,w: int,
    h: int,
    fov: float = 4.6,
    cam_dist: float = 6.0,
    y: float = 0.55,  
) -> Tuple[float, float, float]:
    x, y, z = v
    zc = z + cam_dist
    if zc <= 0.1:
        zc = 0.1
    scale = min(w, h)*0.5 * fov
    sx = (x / zc)*scale+(w - 1)*0.5
    sy = (-y / zc)*scale*y + (h - 1) * 0.5
    return sx, sy, zc 


# Rasterization
def tri_area(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)

def rasterize_triangle(
    grid: List[List[str]],
    zbuf: List[List[float]],
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    c: Tuple[float, float, float],
    ch: str,
) -> None:
    H = len(grid)
    W = len(grid[0])
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c

    minx = max(0, int(math.floor(min(ax, bx, cx))))
    maxx = min(W - 1, int(math.ceil(max(ax, bx, cx))))
    miny = max(0, int(math.floor(min(ay, by, cy))))
    maxy = min(H - 1, int(math.ceil(max(ay, by, cy))))

    area = tri_area(ax, ay, bx, by, cx, cy)
    if area == 0:
        return

    for y in range(miny, maxy + 1):
        py = y + 0.5
        for x in range(minx, maxx + 1):
            px = x + 0.5

            w0 = tri_area(bx, by, cx, cy, px, py)
            w1 = tri_area(cx, cy, ax, ay, px, py)
            w2 = tri_area(ax, ay, bx, by, px, py)

            if area > 0:
                inside = (w0 >= 0 and w1 >= 0 and w2 >= 0)
            else:
                inside = (w0 <= 0 and w1 <= 0 and w2 <= 0)

            if not inside:
                continue

            w0n = w0 / area
            w1n = w1 / area
            w2n = w2 / area

            z = w0n * az + w1n * bz + w2n * cz  
            if z < zbuf[y][x]:
                zbuf[y][x] = z
                grid[y][x] = ch


# Render cube
def render_cube_ascii(q: Quat) -> str:
    # Canvas Size
    FIX_W = 140
    FIX_H = 45

    W, H = FIX_W, FIX_H

    grid = [[" " for _ in range(W)] for _ in range(H)]
    zbuf = [[1e9 for _ in range(W)] for _ in range(H)]

    s = 1.0
    vertex = [
        (-s, -s, -s),  # 0
        ( s, -s, -s),  # 1
        ( s,  s, -s),  # 2
        (-s,  s, -s),  # 3
        (-s, -s,  s),  # 4
        ( s, -s,  s),  # 5
        ( s,  s,  s),  # 6
        (-s,  s,  s),  # 7
    ]

    faces = [
        ((0, 3, 2, 1), "@"),  # -Z
        ((4, 5, 6, 7), "#"),  # +Z
        ((0, 1, 5, 4), "%"),  # -Y
        ((2, 3, 7, 6), "*"),  # +Y
        ((1, 2, 6, 5), "+"),  # +X
        ((0, 4, 7, 3), "="),  # -X
    ]

    vrot: List[Vec3] = [q_rotate(q, v) for v in vertex]
    vproj: List[Tuple[float, float, float]] = [project(v, W, H) for v in vrot]

    for (i0, i1, i2, i3), ch in faces:
        a3, b3, c3 = vrot[i0], vrot[i1], vrot[i2]
        n = v_cross(v_sub(b3, a3), v_sub(c3, a3))
        if n[2] > 0:  
            continue

        p0, p1, p2, p3 = vproj[i0], vproj[i1], vproj[i2], vproj[i3]
        rasterize_triangle(grid, zbuf, p0, p1, p2, ch)
        rasterize_triangle(grid, zbuf, p0, p2, p3, ch)

    return "\n".join("".join(row) for row in grid)


def main():
    q: Quat = (1.0, 0.0, 0.0, 0.0)

    while True:
        clear_screen()
        print(render_cube_ascii(q))
        print("\nMasukkan perubahan sudut dx dy dz: | r(reset) | q(quit): ", end="")

        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()

        if line.lower() == "q":
            break
        if line.lower() == "r":
            q = (1.0, 0.0, 0.0, 0.0)
            continue
        if not line:
            continue

        parts = line.split()
        if len(parts) != 3:
            continue

        try:
            dx, dy, dz = map(float, parts)
        except ValueError:
            continue

        rx, ry, rz = math.radians(dx), math.radians(dy), math.radians(dz)

        qx = q_from_axis_angle((1.0, 0.0, 0.0), rx)
        qy = q_from_axis_angle((0.0, 1.0, 0.0), ry)
        qz = q_from_axis_angle((0.0, 0.0, 1.0), rz)

        q_incInput = qMul(qz, qMul(qy, qx))
        q = qNorm(qMul(q_incInput, q))

    clear_screen()
    print("Bye.")

if __name__ == "__main__":
    main()
