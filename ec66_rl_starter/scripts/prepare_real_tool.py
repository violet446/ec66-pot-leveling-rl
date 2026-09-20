"""Generate a separate URDF with real visuals and piecewise concave scoop collision.

Original assets and the Gate 2 proxy are never modified. The generated model
keeps provisional inertial properties and requires physical calibration.
"""

from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]


def rpy(matrix):
    return [math.atan2(matrix[2, 1], matrix[2, 2]),
            math.atan2(-matrix[2, 0], math.hypot(matrix[0, 0], matrix[1, 0])),
            math.atan2(matrix[1, 0], matrix[0, 0])]


def xyz(values):
    return " ".join(f"{v:.10g}" for v in values)


def main():
    config = json.loads((ROOT / "configs/real_tool_demo.json").read_text(encoding="utf-8"))
    assets = ROOT.parent / "assets"
    source = ET.parse(ROOT.parent / "ec66_sand/assets/ec66_simplified.urdf")
    robot = source.getroot()
    official = ET.parse(assets / "ec66_description.urdf").getroot()
    # Verify the real link meshes use the same kinematic frames as the proxy.
    for i in range(1, 7):
        a = robot.find(f"joint[@name='joint{i}']")
        b = official.find(f"joint[@name='joint{i}']")
        for tag, key in (("origin", "xyz"), ("origin", "rpy"), ("axis", "xyz")):
            np.testing.assert_allclose(np.fromstring(a.find(tag).get(key, "0 0 0"), sep=" "),
                                       np.fromstring(b.find(tag).get(key, "0 0 0"), sep=" "), atol=1e-5)
    for name in ["base_link"] + [f"link{i}" for i in range(1, 7)]:
        link = robot.find(f"link[@name='{name}']")
        for visual in list(link.findall("visual")):
            link.remove(visual)
        visual = ET.SubElement(link, "visual")
        ET.SubElement(ET.SubElement(visual, "geometry"), "mesh",
                      filename=(assets / f"ec66/{name}.STL").as_posix())
        ET.SubElement(visual, "material", name="ec66_white")
    tool = robot.find("link[@name='scoop_link']")
    for child in list(tool):
        tool.remove(child)
    inertial = ET.SubElement(tool, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0.06 0.32")
    ET.SubElement(inertial, "mass", value=str(config["scoop_mass_kg"]))
    ET.SubElement(inertial, "inertia", ixx="0.020", iyy="0.029", izz="0.011", ixy="0", ixz="0", iyz="0")
    # Provisional mass and inertia are retained explicitly, not inferred from STL volume.
    path = assets / "铁锹4y向上.STL"
    mesh = trimesh.load(str(path), force="mesh", process=False)
    mount = np.asarray(config["mount_center_raw_m"])
    vertices = np.asarray(mesh.vertices) * config["stl_scale_to_m"] - mount
    triangles = vertices[mesh.faces]
    visual = ET.SubElement(tool, "visual")
    ET.SubElement(visual, "origin", xyz=xyz(-mount))
    ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=path.as_posix(),
                  scale=xyz([config["stl_scale_to_m"]] * 3))
    ET.SubElement(visual, "material", name="ec66_steel")
    robot.find("joint[@name='scoop_mount']/origin").set("rpy", xyz(config["mount_rpy_rad"]))

    def top_y(x, z):
        # Intersect a ray parallel to raw Y with the real triangles, no scipy required.
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        u, v = b - a, c - a
        det = u[:, 0] * v[:, 2] - u[:, 2] * v[:, 0]
        valid = abs(det) > 1e-12
        den = np.where(valid, det, 1)
        alpha = ((x - a[:, 0]) * v[:, 2] - (z - a[:, 2]) * v[:, 0]) / den
        beta = (u[:, 0] * (z - a[:, 2]) - u[:, 2] * (x - a[:, 0])) / den
        valid &= (alpha >= -1e-6) & (beta >= -1e-6) & (alpha + beta <= 1 + 1e-6)
        return None if not valid.any() else float((a[:, 1] + alpha * u[:, 1] + beta * v[:, 1])[valid].max())

    size = config["patch_size_m"]
    count = 0
    max_residual = 0.0
    patches = []
    # Blade only; shaft has its own cylinder. Overlap the boxes to avoid open seams.
    for x in np.arange(vertices[:, 0].min() + size / 2, vertices[:, 0].max(), size):
        for z in np.arange(0.215, vertices[:, 2].max(), size):
            points = []
            for dx, dz in ((0, 0), (-0.4, 0), (0.4, 0), (0, -0.4), (0, 0.4)):
                px, pz = x + dx * size, z + dz * size
                y = top_y(px, pz)
                if y is not None:
                    points.append((px, y, pz))
            if len(points) < 3:
                continue
            points = np.array(points)
            A = np.column_stack((points[:, 0], points[:, 2], np.ones(len(points))))
            coefficients = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]
            sx, sz, intercept = coefficients
            residual = float(abs(A @ coefficients - points[:, 1]).max())
            max_residual = max(max_residual, residual)
            normal = np.array([-sx, 1., -sz]); normal /= np.linalg.norm(normal)
            ex = np.array([1., sx, 0.]); ex /= np.linalg.norm(ex)
            ey = np.cross(normal, ex)
            matrix = np.column_stack((ex, ey, normal))
            thickness = config["panel_thickness_m"] + 2 * residual
            center = np.array([x, sx*x + sz*z + intercept, z]) - normal * thickness / 2
            dimensions = [size * math.sqrt(1 + sx*sx) * 1.10,
                          size * math.sqrt(1 + sz*sz) * 1.10, thickness]
            collision = ET.SubElement(tool, "collision", name=f"blade_patch_{count:03d}")
            ET.SubElement(collision, "origin", xyz=xyz(center), rpy=xyz(rpy(matrix)))
            ET.SubElement(ET.SubElement(collision, "geometry"), "box", size=xyz(dimensions))
            patches.append({"center": center.tolist(), "rotation": matrix.tolist(), "size": dimensions})
            count += 1
    for label, radius, length, z in (("shaft", 0.019, 0.215, 0.1075), ("mount", 0.0325, 0.012, 0.006)):
        collision = ET.SubElement(tool, "collision", name=label)
        ET.SubElement(collision, "origin", xyz=xyz([0, 0, z]))
        ET.SubElement(ET.SubElement(collision, "geometry"), "cylinder", radius=str(radius), length=str(length))
    tip = vertices[vertices[:, 2] > vertices[:, 2].max() - 0.006]
    tcp_local = np.median(tip, axis=0)
    out = ROOT / "generated_assets/real_tool_demo"
    out.mkdir(parents=True, exist_ok=True)
    ET.indent(source, space="  ")
    source.write(out / "ec66_real_visuals.urdf", encoding="utf-8", xml_declaration=True)
    report = {"config": config, "blade_patch_count": count, "fit_max_residual_m": max_residual,
              "tcp_scoop_local_m": tcp_local.tolist(), "patches": patches,
              "robot_collision": "unchanged proxies", "tool_collision": "piecewise fitted boxes and shaft",
              "capacity_2l_validated": False}
    (out / "geometry.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"GENERATED={out}")
    print(f"BLADE_PATCHES={count} FIT_MAX_RESIDUAL_M={max_residual:.6f} TCP_LOCAL={tcp_local}")


if __name__ == "__main__":
    main()
