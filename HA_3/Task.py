
"""
Полноценный рабочий пример:
1) грузит URDF: C:/urdf/mobile_robot/urdf/robot.urdf
2) создаёт красный маркер и жёстко крепит его "перед роботом"
3) запускает симуляцию в PyBullet GUI (окно видно)
4) пишет кадры top-down ортокамеры
5) считает скорость:
   - ground truth (по координатам робота в PyBullet)
   - CV#1: трекинг красного маркера по HSV
   - CV#2: вычитание кадров (frame differencing)
   - CV#3: оптический поток (Lucas–Kanade)
6) печатает скорости (выделены текстом)

Требуется: pybullet, opencv-python, numpy
"""

import time
import math
import numpy as np
import cv2
import pybullet as p
import pybullet_data


# -----------------------------
# Camera (top-down orthographic)
# -----------------------------
def setup_camera_topdown_ortho(center_xy=(0.0, 0.0), cam_height=3.0, ortho=2.0, img_size=720):
    """
    Ортографическая камера сверху:
    - eye = (cx, cy, cam_height)
    - target = (cx, cy, 0)
    Масштаб метры/пиксель постоянен: mpp = (2*ortho)/W
    """
    cx, cy = center_xy
    eye = [cx, cy, cam_height]
    target = [cx, cy, 0.0]
    up = [0.0, 1.0, 0.0]  # "верх" изображения вдоль +Y мира

    view = p.computeViewMatrix(eye, target, up)
    proj = p.computeProjectionMatrix(left=-ortho, right=ortho, bottom=-ortho, top=ortho, nearVal=900, farVal=1100.0)

    meters_per_pixel = (2.0 * ortho) / float(img_size)
    return view, proj, meters_per_pixel


def get_frame_bgr(view, proj, img_size):
    w = img_size
    h = img_size
    width, height, rgba, depth, seg = p.getCameraImage(
        width=w, height=h,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER
    )
    rgba = np.reshape(rgba, (h, w, 4)).astype(np.uint8)
    rgb = rgba[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


# -----------------------------
# Marker attach (fixed constraint)
# -----------------------------
def attach_red_marker_front(robot_id: int,
                            offset_forward: float = 0.40,
                            offset_up: float = 0.10,
                            radius: float = 0.03,
                            mass: float = 0.01,
                            collision: bool = False) -> int:
    """
    Создаёт красный шар и крепит fixed constraint к базе робота.
    offset_forward/offset_up заданы в ЛОКАЛЬНОЙ системе робота:
      +X = "вперёд", +Z = "вверх".
    Если "вперёд" у твоего URDF не +X, см. ниже в main: можно сменить ось.
    """
    vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1])
    col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius) if collision else -1

    marker_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1]
    )

    # фиксируем к BASE (linkIndex = -1)
    cid = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=-1,
        childBodyUniqueId=marker_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[offset_forward, 0.0, offset_up],
        childFramePosition=[0.0, 0.0, 0.0],
        parentFrameOrientation=[0, 0, 0, 1],
        childFrameOrientation=[0, 0, 0, 1]
    )
    p.changeConstraint(cid, maxForce=500)
    p.changeDynamics(marker_id, -1, linearDamping=0.9, angularDamping=0.9)

    return marker_id


# -----------------------------
# CV method #1: HSV red marker tracking
# -----------------------------
def speed_by_color_marker(frames_bgr, fps, meters_per_pixel,
                          hsv_lo=(0, 120, 120), hsv_hi=(10, 255, 255),
                          min_area=30):
    centers = []
    speeds = []
    prev = None

    for f in frames_bgr:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            centers.append(None)
            speeds.append(np.nan)
            prev = None
            continue

        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            centers.append(None)
            speeds.append(np.nan)
            prev = None
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            centers.append(None)
            speeds.append(np.nan)
            prev = None
            continue

        cur = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=np.float64)
        centers.append(cur)

        if prev is None:
            speeds.append(np.nan)
        else:
            dpix = np.linalg.norm(cur - prev)
            dm = dpix * meters_per_pixel
            speeds.append(dm * fps)

        prev = cur

    return np.array(speeds, dtype=np.float64)


# -----------------------------
# CV method #2: frame differencing
# -----------------------------
def speed_by_frame_diff(frames_bgr, fps, meters_per_pixel, thresh=25, min_area=80):
    speeds = []
    prev_gray = None
    prev_c = None

    for f in frames_bgr:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            speeds.append(np.nan)
            continue

        diff = cv2.absdiff(gray, prev_gray)
        _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            speeds.append(np.nan)
            prev_c = None
            prev_gray = gray
            continue

        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            speeds.append(np.nan)
            prev_c = None
            prev_gray = gray
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            speeds.append(np.nan)
            prev_c = None
            prev_gray = gray
            continue

        cur = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=np.float64)

        if prev_c is None:
            speeds.append(np.nan)
        else:
            dpix = np.linalg.norm(cur - prev_c)
            dm = dpix * meters_per_pixel
            speeds.append(dm * fps)

        prev_c = cur
        prev_gray = gray

    return np.array(speeds, dtype=np.float64)


# -----------------------------
# CV method #3: optical flow (Lucas–Kanade)
# -----------------------------
def speed_by_optical_flow(frames_bgr, fps, meters_per_pixel,
                          max_corners=200, quality_level=0.01, min_distance=7,
                          win_size=(21, 21), max_level=3):
    lk_params = dict(winSize=win_size,
                     maxLevel=max_level,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    speeds = []
    prev_gray = None
    prev_pts = None

    for f in frames_bgr:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners,
                                               qualityLevel=quality_level, minDistance=min_distance)
            speeds.append(np.nan)
            continue

        if prev_pts is None or len(prev_pts) < 10:
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners,
                                               qualityLevel=quality_level, minDistance=min_distance)

        if prev_pts is None:
            speeds.append(np.nan)
            prev_gray = gray
            continue

        next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
        if next_pts is None:
            speeds.append(np.nan)
            prev_gray = gray
            prev_pts = None
            continue

        st = st.reshape(-1).astype(bool)
        good_prev = prev_pts.reshape(-1, 2)[st]
        good_next = next_pts.reshape(-1, 2)[st]

        if len(good_prev) < 10:
            speeds.append(np.nan)
            prev_gray = gray
            prev_pts = None
            continue

        flow = good_next - good_prev
        dx_med = float(np.median(flow[:, 0]))
        dy_med = float(np.median(flow[:, 1]))
        dpix = math.sqrt(dx_med * dx_med + dy_med * dy_med)

        dm = dpix * meters_per_pixel
        speeds.append(dm * fps)

        prev_gray = gray
        prev_pts = good_next.reshape(-1, 1, 2).astype(np.float32)

    return np.array(speeds, dtype=np.float64)


# -----------------------------
# Helpers
# -----------------------------
def robust_mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    q1, q3 = np.percentile(x, [10, 90])
    x = x[(x >= q1) & (x <= q3)]
    return float(np.mean(x)) if x.size else float("nan")


# -----------------------------
# Main
# -----------------------------
def main():
    URDF_PATH = r"C:/urdf/mobile_robot/urdf/robot.urdf"

    # --- параметры эксперимента
    sim_seconds = 3.0
    dt = 1.0 / 240.0
    target_speed_mps = 3.0  # "заданная" скорость (через скорость базы)
    record_fps = 60          # частота записи кадров для CV
    img_size = 720
    cam_height = 1000
    ortho = 1       # половина ширины окна (в метрах)
    realtime = False         # True = шагать примерно в реальном времени (медленнее, но наглядно)

    # --- запуск Bullet GUI (покажет симуляцию)
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(dt)

    # визуальные настройки
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(lightPosition=[2, 2, 5])

    # сцена
    p.loadURDF("plane.urdf")

    # робот
    start_pos = [0.0, 0.0, 0.0]
    start_orn = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF(URDF_PATH, start_pos, start_orn, useFixedBase=False)

    # --- маркер "перед роботом"
    # ВАЖНО: offset_forward предполагает, что "вперёд" у URDF — это +X.
    # Если у твоего URDF "вперёд" это +Y, то самый простой фикс:
    #   заменить parentFramePosition=[0.0, offset_forward, offset_up] в attach_red_marker_front
    # или временно повернуть робота стартовым yaw.
    marker_id = attach_red_marker_front(robot_id, offset_forward=0.40, offset_up=0.10, radius=0.03)

    # --- задаём скорость роботу (так как у URDF нет вращений/суставов)
    p.resetBaseVelocity(robot_id, linearVelocity=[target_speed_mps, 0.0, 0.0], angularVelocity=[0, 0, 0])

    # --- камера
    view, proj, mpp = setup_camera_topdown_ortho(center_xy=(0.0, 0.0),
                                                 cam_height=cam_height,
                                                 ortho=ortho,
                                                 img_size=img_size)

    # --- цикл симуляции + сбор ground truth + запись кадров
    n_steps = int(sim_seconds / dt)
    k = max(1, int(round((1.0 / record_fps) / dt)))  # писать кадр раз в k шагов

    frames = []
    gt_pos = []
    gt_t = []

    t0_wall = time.perf_counter()
    t_sim = 0.0

    for i in range(n_steps):
        p.stepSimulation()
        t_sim += dt

        pos, orn = p.getBasePositionAndOrientation(robot_id)
        gt_pos.append(pos)
        gt_t.append(t_sim)

        if (i % k) == 0:
            frames.append(get_frame_bgr(view, proj, img_size))

        if realtime:
            time.sleep(dt)

    t1_wall = time.perf_counter()

    gt_pos = np.asarray(gt_pos, dtype=np.float64)  # Nx3
    gt_t = np.asarray(gt_t, dtype=np.float64)
    t_sim_total = float(gt_t[-1])
    t_wall = float(t1_wall - t0_wall)

    # ground truth speed (XY)
    dxy = np.linalg.norm(gt_pos[1:, :2] - gt_pos[:-1, :2], axis=1)
    v_gt = dxy / dt
    v_gt_mean = float(np.mean(v_gt))

    # --- CV speeds
    v_color = speed_by_color_marker(frames, record_fps, mpp,
                                    hsv_lo=(0, 120, 120), hsv_hi=(10, 255, 255),
                                    min_area=20)

    # (иногда красный бывает около Hue=180; если будет плохо — добавь второй диапазон 170..180)
    v_diff = speed_by_frame_diff(frames, record_fps, mpp, thresh=25, min_area=80)
    v_flow = speed_by_optical_flow(frames, record_fps, mpp)

    v_color_mean = robust_mean(v_color)
    v_diff_mean = robust_mean(v_diff)
    v_flow_mean = robust_mean(v_flow)

    # --- печать результатов (выделено)
    print("\n" + "=" * 70)
    print("RESULTS (скорости в м/с)")
    print("=" * 70)
    print(f"[GROUND TRUTH]  v_gt_mean   = {v_gt_mean:.4f}  m/s")
    print(f"[CV #1 HSV]     v_color_mean= {v_color_mean:.4f}  m/s")
    print(f"[CV #2 DIFF]    v_diff_mean = {v_diff_mean:.4f}  m/s")
    print(f"[CV #3 FLOW]    v_flow_mean = {v_flow_mean:.4f}  m/s")
    print("-" * 70)
    print(f"t_sim_total = {t_sim_total:.4f} s   |   t_wall = {t_wall:.4f} s")
    print(f"meters_per_pixel = {mpp:.6f} m/px   |   frames = {len(frames)} @ {record_fps} fps")
    print("=" * 70)

    # --- оставим окно симуляции открытым
    print("\nОкно PyBullet GUI оставлено открытым. Закрой его вручную для выхода.")
    while p.isConnected():
        time.sleep(0.1)


if __name__ == "__main__":
    main()