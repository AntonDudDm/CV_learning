import math
from pathlib import Path

import numpy as np
import cv2
import pybullet as p
import pybullet_data

def add_solid_background_ground_box(
    client_id: int,
    half_extents=(5.0, 5.0, 0.01),      # 10м x 10м, толщина 2см
    z_top=0.0,                          # верхняя поверхность на z=0
    rgba=(0.15, 0.15, 0.15, 1.0)
) -> int:
    """
    Одноцветный фон без шахматки: большая тонкая коробка.
    Верхняя грань лежит на z=z_top (обычно 0).
    """
    hx, hy, hz = half_extents
    col_id = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=client_id
    )
    vis_id = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=list(rgba), physicsClientId=client_id
    )

    # Центр коробки должен быть ниже, чтобы верх был ровно на z_top:
    base_z = z_top - hz

    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[0.0, 0.0, base_z],
        physicsClientId=client_id
    )
    return body_id


def fov_deg_from_fy(height_px: int, fy_px: float) -> float:
    return float(2.0 * math.degrees(math.atan((height_px / 2.0) / fy_px)))


def setup_camera_topdown(
    width: int,
    height: int,
    fy: float,
    near: float,
    far: float,
    cam_height: float,
    target_xy=(0.0, 0.0),
):
    aspect = width / float(height)
    fov = fov_deg_from_fy(height, fy)

    tx, ty = target_xy
    eye = [tx, ty, cam_height]
    target = [tx, ty, 0.0]
    up = [0.0, 1.0, 0.0]  # важно: не (0,0,1) для top-down

    view = p.computeViewMatrix(eye, target, up)
    proj = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=near, farVal=far)
    return view, proj


def render_rgb(width: int, height: int, view, proj, renderer=p.ER_TINY_RENDERER) -> np.ndarray:
    img = p.getCameraImage(width, height, view, proj, renderer=renderer)
    rgba = np.reshape(img[2], (height, width, 4)).astype(np.uint8)
    return rgba[:, :, :3]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def generate_for_model(
    client_id: int,
    urdf_path: str,
    out_dir: Path,
    label_value: int,
    n_images: int,
    width: int,
    height: int,
    fy: float,
    near: float,
    far: float,
    cam_height: float,
    obj_pos=(0.0, 0.0, 0.0),
    global_scaling: float = 1.0,
    seed: int = 0,
    prefix: str = "img",
):
    rng = np.random.default_rng(seed)

    view, proj = setup_camera_topdown(
        width=width,
        height=height,
        fy=fy,
        near=near,
        far=far,
        cam_height=cam_height,
        target_xy=(obj_pos[0], obj_pos[1]),
    )

    ensure_dir(out_dir)

    labels_path = out_dir.parent / "labels.txt"
    with open(labels_path, "a", encoding="utf-8") as f_labels:
        for i in range(n_images):
            p.resetSimulation(physicsClientId=client_id)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
            p.setGravity(0, 0, -9.8, physicsClientId=client_id)

            # фон
            add_solid_background_ground_box(client_id, half_extents=(5.0, 5.0, 0.01), z_top=0.0, rgba=(0.15, 0.15, 0.15, 1.0))

            # случайный yaw
            yaw = rng.uniform(0.0, 2.0 * math.pi)
            orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])

            p.loadURDF(
                urdf_path,
                basePosition=list(obj_pos),
                baseOrientation=orn,
                useFixedBase=True,
                globalScaling=global_scaling,
                physicsClientId=client_id,
            )

            p.stepSimulation(physicsClientId=client_id)

            rgb = render_rgb(width, height, view, proj, renderer=p.ER_TINY_RENDERER)
            fname = f"{prefix}_{i:04d}.png"
            out_path = out_dir / fname

            cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            rel = out_path.relative_to(out_dir.parent)
            f_labels.write(f"{rel.as_posix()} {label_value}\n")


def main():
    # ==========================
    # 1) ПАРАМЕТРЫ — МЕНЯЕШЬ ТОЛЬКО ЭТО
    # ==========================
    MODEL1_URDF = r"C:\URDF\Gaika\urdf\Gaika.urdf"
    MODEL2_URDF = r"C:\urdf\Shaiba\urdf\Shaiba.urdf"

    OUT_ROOT = Path(r"dataset")

    N_PER_CLASS = 500

    # камера/рендер
    WIDTH = 1920
    HEIGHT = 1080
    FY = 1796.1276552044753
    NEAR = 0.001
    FAR = 3.0
    CAM_HEIGHT = 0.02  # метры над объектом

    # масштаб (если URDF слишком большой/маленький)
    SCALE1 = 1.0
    SCALE2 = 1.0

    # сиды для воспроизводимости
    SEED1 = 0
    SEED2 = 1

    # GUI для дебага
    USE_GUI = False
    # ==========================

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    class1_dir = OUT_ROOT / "class1"
    class2_dir = OUT_ROOT / "class2"
    ensure_dir(class1_dir)
    ensure_dir(class2_dir)

    # перезаписываем labels.txt
    (OUT_ROOT / "labels.txt").write_text("", encoding="utf-8")

    client_id = p.connect(p.GUI if USE_GUI else p.DIRECT)

    try:
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=client_id)
    except Exception:
        pass

    generate_for_model(
        client_id=client_id,
        urdf_path=MODEL1_URDF,
        out_dir=class1_dir,
        label_value=1,
        n_images=N_PER_CLASS,
        width=WIDTH,
        height=HEIGHT,
        fy=FY,
        near=NEAR,
        far=FAR,
        cam_height=CAM_HEIGHT,
        obj_pos=(0.0, 0.0, 0.0),
        global_scaling=SCALE1,
        seed=SEED1,
        prefix="c1",
    )

    generate_for_model(
        client_id=client_id,
        urdf_path=MODEL2_URDF,
        out_dir=class2_dir,
        label_value=-1,
        n_images=N_PER_CLASS,
        width=WIDTH,
        height=HEIGHT,
        fy=FY,
        near=NEAR,
        far=FAR,
        cam_height=CAM_HEIGHT,
        obj_pos=(0.0, 0.0, 0.0),
        global_scaling=SCALE2,
        seed=SEED2,
        prefix="c2",
    )

    p.disconnect(client_id)
    print("OK:", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()