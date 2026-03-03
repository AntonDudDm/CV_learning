import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data


import matplotlib.pyplot as plt

def plot_results(times, v_gt, v_cv_marker, v_cv_diff, v_cv_flow,
                 title="Speed comparison", skip_seconds=0.0, win=15):
    t = np.asarray(times, dtype=float)
    v_gt = np.asarray(v_gt, dtype=float)
    v_m = np.asarray(v_cv_marker, dtype=float)
    v_d = np.asarray(v_cv_diff, dtype=float)
    v_f = np.asarray(v_cv_flow, dtype=float)

    if skip_seconds > 0:
        m = t >= (t[0] + skip_seconds)
        t, v_gt, v_m, v_d, v_f = t[m], v_gt[m], v_m[m], v_d[m], v_f[m]

    def moving_average_nan(x, win_):
        x = np.asarray(x, float)
        mask = np.isfinite(x)
        x0 = np.where(mask, x, 0.0)
        w0 = mask.astype(float)
        k = np.ones(win_, float)
        num = np.convolve(x0, k, mode="same")
        den = np.convolve(w0, k, mode="same")
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    v_m_s = moving_average_nan(v_m, win)
    v_d_s = moving_average_nan(v_d, win)
    v_f_s = moving_average_nan(v_f, win)

    # Окно 1: RAW
    plt.figure()
    plt.plot(t, v_gt, label="GT")
    plt.plot(t, v_m, label="CV1 marker (raw)")
    plt.plot(t, v_d, label="CV2 diff (raw)")
    plt.plot(t, v_f, label="CV3 flow (raw)")
    plt.xlabel("t, s")
    plt.ylabel("speed, m/s")
    plt.title(title + " (RAW)")
    plt.grid(True)
    plt.legend()

    # Ошибки (RAW)
    plt.figure()
    plt.plot(t, v_m - v_gt, label="CV1 - GT")
    plt.plot(t, v_d - v_gt, label="CV2 - GT")
    plt.plot(t, v_f - v_gt, label="CV3 - GT")
    plt.axhline(0.0)
    plt.xlabel("t, s")
    plt.ylabel("error, m/s")
    plt.title("Speed error (RAW)")
    plt.grid(True)
    plt.legend()

    # Окно 2: SMOOTH
    plt.figure()
    plt.plot(t, v_gt, label="GT")
    plt.plot(t, v_m_s, label="CV1 marker (smooth)")
    plt.plot(t, v_d_s, label="CV2 diff (smooth)")
    plt.plot(t, v_f_s, label="CV3 flow (smooth)")
    plt.xlabel("t, s")
    plt.ylabel("speed, m/s")
    plt.title(title + f" (SMOOTH, win={win})")
    plt.grid(True)
    plt.legend()

    plt.show()

# -----------------------------
# 1) Инициализация симуляции
# -----------------------------
def init_sim(gui: bool = True) -> int:
    """
    Запуск PyBullet.
    gui=True -> окно симуляции (нужно по заданию).
    """
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return cid


def setup_scene(dt: float = 1.0 / 240.0) -> int:
    """
    Настройка сцены: шаг времени, гравитация, плоскость.
    """
    p.setTimeStep(dt)
    p.setGravity(0, 0, -9.81)

    # свет/тени
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(lightPosition=[2, 2, 5])

    plane_id = p.loadURDF("plane.urdf")
    return plane_id


def load_robot(urdf_path: str,
               start_pos=(0.0, 0.0, 0.10),
               start_yaw: float = 0.0,
               fixed_base: bool = False) -> int:
    """
    Загрузка URDF.
    """
    start_orn = p.getQuaternionFromEuler([0.0, 0.0, start_yaw])
    robot_id = p.loadURDF(urdf_path, start_pos, start_orn, useFixedBase=fixed_base)
    return robot_id
    
def recolor_body_dark(body_id: int, rgba=(0.08, 0.08, 0.08, 1.0)):
    """
    Перекрашивает ВСЕ визуальные части body в тёмный цвет.
    Важно: если на mesh есть texture, PyBullet может продолжать её показывать.
    Тогда мы явно сбрасываем textureUniqueId = -1 (если поддерживается).
    """
    nj = p.getNumJoints(body_id)
    for link in [-1] + list(range(nj)):
        try:
            # Убираем текстуру (если была), иначе цвет может не применяться визуально.
            p.changeVisualShape(body_id, link, rgbaColor=rgba, textureUniqueId=-1)
        except TypeError:
            # В некоторых сборках textureUniqueId не принимается — тогда просто цвет
            p.changeVisualShape(body_id, link, rgbaColor=rgba)

        # Матовый вид (чтобы не было бликов)
        try:
            p.changeVisualShape(body_id, link, specularColor=[0.0, 0.0, 0.0])
        except Exception:
            pass


# -----------------------------
# 2) Камера (View + Projection + getCameraImage)
# -----------------------------
def create_topdown_camera(
    center_xy=(0.0, 0.0),
    cam_height: float = 2.0,
    ortho: float = 2.0,
    img_wh: int = 720,
    nearVal: float = 0.01,
    farVal: float = 50.0
):
    """
    Реализация:
      view = computeViewMatrix(...)
      proj = computeProjectionMatrix(left/right/bottom/top, nearVal, farVal)

    Важно:
    - nearVal/farVal должны соответствовать масштабу сцены (метры!).
      near=900, far=1100 для робота в 1-5 метрах даст пустой кадр.
    """
    cx, cy = center_xy

    # как в лекции: камера сверху
    view = p.computeViewMatrix(
        cameraEyePosition=[cx, cy, cam_height],
        cameraTargetPosition=[cx, cy, 0.0],
        cameraUpVector=[0.0, 1.0, 0.0]
    )

    # как в лекции: границы окна (в терминах лекции)
    proj = p.computeProjectionMatrix(
        left=-ortho, right=ortho,
        bottom=-ortho, top=ortho,
        nearVal=nearVal, farVal=farVal
    )

    # Для последующего пересчёта пикселей в метры мы будем считать:
    # mpp = (2*ortho)/W
    # (это будет работать "как ортографика" только если использовать этот режим стабильно и камера top-down)
    meters_per_pixel = (2.0 * ortho) / float(img_wh) * (cam_height / nearVal)

    return view, proj, meters_per_pixel

def grab_frame_bgr(view, proj, width: int, height: int):
    w, h, rgba, depth, seg = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER
    )
    rgba = np.reshape(rgba, (h, w, 4)).astype(np.uint8)
    rgb = rgba[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    seg = np.reshape(seg, (h, w)).astype(np.int32)  # важно привести форму
    return bgr, seg

# -----------------------------
# 3) Marker attach (fixed constraint)
# -----------------------------
def attach_red_marker_fixed(robot_id: int,
                            offset_local=(0.50, 0.0, 0.12),
                            radius: float = 0.04,
                            marker_mass: float = 0.05,
                            max_force: float = 5_000.0) -> tuple[int, int]:
    """
    Честное крепление маркера к базе робота через JOINT_FIXED.

    Возвращает (marker_id, constraint_id).

    Условия стабильности:
    - robot_id должен двигаться физически (НЕ resetBasePositionAndOrientation)
    - маркер делаем динамическим (mass>0), но без collision
    - отключаем sleeping для маркера
    """

    # 0) Узнаём текущую позу робота
    r_pos, r_orn = p.getBasePositionAndOrientation(robot_id)
    r_pos = np.array(r_pos, dtype=np.float64)
    R = np.array(p.getMatrixFromQuaternion(r_orn), dtype=np.float64).reshape(3, 3)
    offset_world = R @ np.array(offset_local, dtype=np.float64)

    marker_start_pos = (r_pos + offset_world).tolist()
    marker_start_orn = r_orn  # логично дать ту же ориентацию

    # 1) визуальная форма
    vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1])

    # 2) маркер как динамическое тело (но без collision, чтобы не "толкался" о пол/робота)
    marker_id = p.createMultiBody(
        baseMass=marker_mass,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis_id,
        basePosition=marker_start_pos,
        baseOrientation=marker_start_orn
    )

    # 3) отключаем сон и демпфирование маркера (чтобы solver его всегда "таскал")
    p.changeDynamics(marker_id, -1, linearDamping=0.0, angularDamping=0.0)
    p.changeDynamics(marker_id, -1, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)

    # 4) фиксируем к базе робота
    cid = p.createConstraint(
        parentBodyUniqueId=robot_id, parentLinkIndex=-1,
        childBodyUniqueId=marker_id, childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=list(offset_local),
        childFramePosition=[0.0, 0.0, 0.0],
        parentFrameOrientation=[0, 0, 0, 1],
        childFrameOrientation=[0, 0, 0, 1]
    )
    p.changeConstraint(cid, maxForce=max_force)

    # 5) принудительная активация тел (важно)
    p.resetBaseVelocity(marker_id, [0, 0, 0], [0, 0, 0])  # "будим"
    p.resetBaseVelocity(robot_id,  [0, 0, 0], [0, 0, 0])

    return marker_id, cid

# -----------------------------
# 4) Robot motion (two modes)
# -----------------------------
def set_robot_motion(robot_id: int, v_xy=(0.5, 0.0), mode: str = "velocity"):
    """
    Задаёт движение робота.
    mode:
      - "velocity": физический режим (resetBaseVelocity)
      - "kinematic": кинематический режим (будем двигать через resetBasePositionAndOrientation в цикле)

    Возвращает dict state для kinematic, иначе None.
    """
    vx, vy = float(v_xy[0]), float(v_xy[1])

    if mode == "velocity":
        p.resetBaseVelocity(robot_id, linearVelocity=[vx, vy, 0.0], angularVelocity=[0, 0, 0])
        return None

    if mode == "kinematic":
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        state = {
            "pos": np.array(pos, dtype=np.float64),
            "orn": orn,
            "v": np.array([vx, vy, 0.0], dtype=np.float64)
        }
        return state

    raise ValueError("mode must be 'velocity' or 'kinematic'")

def step_kinematic(robot_id: int, state: dict, dt: float):
    """
    Один шаг кинематического движения: x <- x + v*dt.
    """
    state["pos"] = state["pos"] + state["v"] * dt
    p.resetBasePositionAndOrientation(robot_id, state["pos"].tolist(), state["orn"])


# -----------------------------
# 5) Реальное время и скорость в симуляции
# -----------------------------

# -----------------------------
# Таймеры: wall-clock время
# -----------------------------
def start_timers():
    """
    Старт измерения реального времени выполнения (wall-clock).
    """
    return time.perf_counter()


def stop_timers(t0_wall: float) -> float:
    """
    Останов измерения wall-clock.
    """
    return time.perf_counter() - t0_wall


# -----------------------------
# Ground truth скорость (по позициям)
# -----------------------------
def gt_step(robot_id: int, prev_pos_xy: np.ndarray, dt: float):
    """
    Один шаг измерения ground truth скорости.

    prev_pos_xy: np.array([x_prev, y_prev])
    Возвращает:
      cur_pos_xy, v_gt
    """
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    cur_pos_xy = np.array([pos[0], pos[1]], dtype=np.float64)

    # скорость по конечной разности
    d = np.linalg.norm(cur_pos_xy - prev_pos_xy)
    v_gt = d / dt

    return cur_pos_xy, float(v_gt)


# -----------------------------
# CV #1: HSV tracking of red marker (single frame)
# -----------------------------
def detect_red_center_px(frame_bgr, min_area: int = 20):
    """
    Возвращает (center_xy_px, mask) или (None, mask), если не найден.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # красный = два диапазона по Hue
    lo1, hi1 = (0, 120, 120), (10, 255, 255)
    lo2, hi2 = (170, 120, 120), (180, 255, 255)

    m1 = cv2.inRange(hsv, np.array(lo1, np.uint8), np.array(hi1, np.uint8))
    m2 = cv2.inRange(hsv, np.array(lo2, np.uint8), np.array(hi2, np.uint8))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask

    c = max(cnts, key=cv2.contourArea)

    (x, y), r = cv2.minEnclosingCircle(c)
    center = np.array([x, y], dtype=np.float64)
    return center, mask
    
    '''
    if cv2.contourArea(c) < min_area:
        return None, mask

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, mask

    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    return np.array([cx, cy], dtype=np.float64), mask
    '''


# -----------------------------
# CV #2: Frame differencing (single frame pair)
# -----------------------------
def detect_motion_center_px(gray_prev, gray_cur, thresh: int = 15, min_pixels: int = 200):
    """
    CV2 diff: центр движения как центр масс бинарной маски.
    Возвращает (center_xy_px, mask) или (None, mask).

    thresh: порог по яркости после blur
    min_pixels: минимальное число белых пикселей в маске движения
    """
    # подавляем шахматный шум/алиасинг
    gray_prev = cv2.GaussianBlur(gray_prev, (7, 7), 0)
    gray_cur  = cv2.GaussianBlur(gray_cur,  (7, 7), 0)

    diff = cv2.absdiff(gray_cur, gray_prev)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    nz = cv2.countNonZero(mask)
    if nz < min_pixels:
        return None, mask

    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] == 0:
        return None, mask

    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    return np.array([cx, cy], dtype=np.float64), mask


# -----------------------------
# CV #3: оптический поток Lucas–Kanade (KLT)
# -----------------------------

def of_init_points(gray, mask=None, max_corners=200, quality=0.01, min_dist=7):
    pts = cv2.goodFeaturesToTrack(
        gray,
        mask=mask,  # <-- ключевое
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_dist
    )
    return pts


def of_step_lk(prev_gray, cur_gray, prev_pts):
    """
    Один шаг KLT (Lucas–Kanade).
    Возвращает (next_pts, median_flow_px) или (None, None) если не удалось.
    median_flow_px = np.array([dx, dy]) в пикселях.
    """
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_pts, None, **lk_params)
    if next_pts is None or st is None:
        return None, None

    st = st.reshape(-1).astype(bool)
    good_prev = prev_pts.reshape(-1, 2)[st]
    good_next = next_pts.reshape(-1, 2)[st]

    if good_prev.shape[0] < 10:
        return None, None

    flow = good_next - good_prev
    med = np.median(flow, axis=0)  # [dx, dy]
    return good_next.reshape(-1, 1, 2).astype(np.float32), med.astype(np.float64)


def draw_flow_arrows(img_bgr, prev_pts, next_pts, step=10, scale=1.0):
    """
    Рисует стрелки потока для части точек.
    step=10 -> рисуем примерно каждую 10-ю точку.
    """
    if prev_pts is None or next_pts is None:
        return img_bgr
    p0 = prev_pts.reshape(-1, 2)
    p1 = next_pts.reshape(-1, 2)

    out = img_bgr
    for k in range(0, min(len(p0), len(p1)), step):
        x0, y0 = int(p0[k, 0]), int(p0[k, 1])
        x1 = int(p0[k, 0] + (p1[k, 0] - p0[k, 0]) * scale)
        y1 = int(p0[k, 1] + (p1[k, 1] - p0[k, 1]) * scale)
        cv2.arrowedLine(out, (x0, y0), (x1, y1), (0, 255, 255), 1, tipLength=0.3)
    return out

# -----------------------------
# Simulation runner: GT + CV speed + preview
# -----------------------------
def run_simulation_collect_gt_and_cv_preview(robot_id: int,
                                             dt: float,
                                             sim_seconds: float,
                                             view, proj,
                                             img_wh: int,
                                             meters_per_pixel: float,
                                             v_xy=(0.5, 0.0),
                                             realtime: bool = True,
                                             print_every_s: float = 0.5,
                                             warmup_steps: int = 10):
    """
    Внутри одной функции:
    - поддерживаем скорость робота (resetBaseVelocity каждый шаг)
    - считаем GT скорость
    - считаем CV скорость по маркеру (HSV)
    - показываем кадр с двумя скоростями
    - сохраняем массивы v_gt, v_cv

    Возвращает dict:
      times, v_gt, v_cv, t_sim_total, t_wall_total
    """
    total_steps = int(sim_seconds / dt)
    n_steps = total_steps - warmup_steps

    if n_steps <= 10:
        raise ValueError("warmup_steps too large for sim_seconds/dt")

    # --- warmup: двигаем и шагаем, но НЕ измеряем
    for _ in range(warmup_steps):
        p.resetBaseVelocity(robot_id, linearVelocity=[v_xy[0], v_xy[1], 0.0], angularVelocity=[0, 0, 0])
        p.stepSimulation()
        if realtime:
            time.sleep(dt)

    # GT init
    pos0, _ = p.getBasePositionAndOrientation(robot_id)
    prev_xy = np.array([pos0[0], pos0[1]], dtype=np.float64)

    # CV init
    prev_center = None
    prev_gray = None
    prev_motion_center = None

    times = np.zeros(n_steps, dtype=np.float64)
    v_gt = np.zeros(n_steps, dtype=np.float64)
    v_cv_marker  = np.full(n_steps, np.nan, dtype=np.float64)
    v_cv_diff = np.full(n_steps, np.nan, dtype=np.float64)
    v_cv_flow = np.full(n_steps, np.nan, dtype=np.float64)

    t_sim = 0.0
    t0_wall = time.perf_counter()
    print_every_steps = max(1, int(round(print_every_s / dt)))

    centers = []  # list of (t_sim, center) for CV1

    motion_centers = []  # list of (t_sim, c_motion) for CV2
    K_DIFF = 10          # окно для скорости CV2 (10*dt ~ 0.042s)

    # --- CV#3 optical flow state ---
    prev_pts = None
    FLOW_REINIT_EVERY = 30  # переинициализация точек раз в ~30 шагов

    for i in range(n_steps):
        # 1) поддержка скорости каждый шаг (иначе физика может "съедать" скорость)
        p.resetBaseVelocity(robot_id, linearVelocity=[v_xy[0], v_xy[1], 0.0], angularVelocity=[0, 0, 0])

        # 2) шаг физики
        p.stepSimulation()
        t_sim += dt

        # 3) GT скорость
        cur_xy, v = gt_step(robot_id, prev_xy, dt)
        prev_xy = cur_xy
        times[i] = t_sim
        v_gt[i] = v

        # 4) кадр камеры (статичная камера)
        frame, seg = grab_frame_bgr(view, proj, img_wh, img_wh)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 5) CV#1 скорость по маркеру
        center, mask = detect_red_center_px(frame, min_area=20)

        if center is not None:
            centers.append((t_sim, center))
            # держим последние M точек
            M = 10
            if len(centers) > M:
                centers.pop(0)

            # если есть хотя бы 2 точки, считаем скорость по "длинной" разности
            if len(centers) >= 2:
                t_old, c_old = centers[0]
                t_new, c_new = centers[-1]
                dpix = float(np.linalg.norm(c_new - c_old))
                dt_cv = float(t_new - t_old)
                if dt_cv > 0:
                    v_cv_marker [i] = (dpix * meters_per_pixel) / dt_cv


        # 6) CV#2 frame differencing (через функцию)
        if prev_gray is not None:
            c_motion, diff_mask = detect_motion_center_px(prev_gray, gray, thresh=15, min_pixels=200)

                # Показ "сырого" diff, чтобы видеть хоть что-то
            cv2.imshow("DIFF_RAW", diff_mask)

            if c_motion is not None:
                motion_centers.append((t_sim, c_motion))
                if len(motion_centers) > K_DIFF + 1:
                    motion_centers.pop(0)

                if len(motion_centers) >= K_DIFF + 1:
                    t_old, c_old = motion_centers[0]
                    t_new, c_new = motion_centers[-1]
                    dpix2 = float(np.linalg.norm(c_new - c_old))
                    dt2 = float(t_new - t_old)
                    if dt2 > 0:
                        v_cv_diff[i] = (dpix2 * meters_per_pixel) / dt2
            else:
                # если движение не найдено — сбросим буфер, чтобы не было ложных скачков
                motion_centers.clear()


        # 7) --- CV#3 optical flow (KLT) ---
        robot_mask = (seg == robot_id).astype(np.uint8) * 255

        if prev_gray is not None:
            # переинициализация точек иногда нужна (теряются/деградируют)
            if (prev_pts is None) or (i % FLOW_REINIT_EVERY == 0):
                prev_pts = of_init_points(prev_gray, mask=robot_mask, max_corners=400, quality=0.005, min_dist=5)

            if prev_pts is not None:
                next_pts, med_flow = of_step_lk(prev_gray, gray, prev_pts)
                if next_pts is not None and med_flow is not None:
                    dx, dy = float(med_flow[0]), float(med_flow[1])
                    dpix = (dx * dx + dy * dy) ** 0.5
                    v_cv_flow[i] = (dpix * meters_per_pixel) / dt
                    prev_pts = next_pts
                else:
                    prev_pts = None


        if prev_pts is not None:
            for pt in prev_pts.reshape(-1, 2)[::10]:
                cv2.circle(disp, (int(pt[0]), int(pt[1])), 2, (255, 0, 255), -1)

        # prev_gray обновляем после обработки
        prev_gray = gray.copy()

        #  overlay в окно
        disp = frame.copy()
        if center is not None:
            cv2.circle(disp, (int(center[0]), int(center[1])), 6, (0, 255, 0), 2)
        cv2.putText(disp, f"t={t_sim:.2f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(disp, f"GT: {v_gt[i]:.3f} m/s", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        txt1 = f"CV1 marker: {v_cv_marker[i]:.3f} m/s" if np.isfinite(v_cv_marker[i]) else "CV1 marker: ---"
        txt2 = f"CV2 diff  : {v_cv_diff[i]:.3f} m/s" if np.isfinite(v_cv_diff[i]) else "CV2 diff  : ---"
        txt3 = f"CV3 flow : {v_cv_flow[i]:.3f} m/s" if np.isfinite(v_cv_flow[i]) else "CV3 flow : ---"

        cv2.putText(disp, txt1, (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(disp, txt2, (10,118), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(disp, txt3, (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("TopDown (3 speeds)", disp)
        if cv2.waitKey(1) == 27:
            times = times[:i+1]
            v_gt = v_gt[:i+1]
            v_cv_marker = v_cv_marker[:i+1]
            v_cv_diff = v_cv_diff[:i+1]
            v_cv_flow = v_cv_flow[:i+1]
            break

        if realtime:
            time.sleep(dt)

    t_wall = time.perf_counter() - t0_wall
    return {
        "times": times,
        "v_gt": v_gt,
        "v_cv_marker": v_cv_marker,
        "v_cv_diff": v_cv_diff,
        "v_cv_flow": v_cv_flow,
        "t_sim_total": float(t_sim),
        "t_wall_total": float(t_wall),
    }
# -----------------------------
# Main: демонстрация маркера + движения + preview камеры
# -----------------------------
def main():

    URDF_PATH = r"C:/URDF/mobile_robot/urdf/robot.urdf"

    dt = 1.0 / 240.0
    sim_seconds = 1.0
    n_steps = int(sim_seconds / dt)

    img_wh = 720

    # камера
    cam_height = 1000.0
    ortho = 1.0
    nearVal = 900
    farVal = 1100

    # скорость (м/с)
    v_xy = (1.0, 0.0)

    # режим движения: "velocity" или "kinematic"
    motion_mode = "velocity"  # <- если URDF не физический, ставь kin

    init_sim(gui=True)
    setup_scene(dt=dt)

    robot_id = load_robot(URDF_PATH, start_pos=(-0.8, 0.0, 0.0), start_yaw=0.0, fixed_base=False)
    recolor_body_dark(robot_id, rgba=(0.08, 0.08, 0.08, 1.0))

   
    # отключаем трение/демпфирование/сон ВЕЗДЕ (на всех линках)
    nj = p.getNumJoints(robot_id)
    for link in [-1] + list(range(nj)):
        p.changeDynamics(robot_id, link, linearDamping=0.0, angularDamping=0.0)
        p.changeDynamics(robot_id, link, lateralFriction=0.0, rollingFriction=0.0, spinningFriction=0.0)
        p.changeDynamics(robot_id, link, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)

    # честно крепим маркер
    marker_id, marker_cid = attach_red_marker_fixed(
        robot_id,
        offset_local=(0.20, 0.0, 0.12),
        radius=0.04,
        marker_mass=0.05,
        max_force=5000.0
    )


    #скорость
    motion_state = set_robot_motion(robot_id, v_xy=v_xy, mode=motion_mode)

    view, proj, mpp = create_topdown_camera(
        center_xy=(0.0, 0.0),
        cam_height=cam_height, ortho=ortho,
        img_wh=img_wh, nearVal=nearVal, farVal=farVal
    )

    # --- подготовка GT измерений
    pos0, _ = p.getBasePositionAndOrientation(robot_id)
    prev_xy = np.array([pos0[0], pos0[1]], dtype=np.float64)

    times = np.zeros(n_steps, dtype=np.float64)
    v_gt = np.zeros(n_steps, dtype=np.float64)

    t_sim = 0.0
    t0_wall = time.perf_counter()

    print("robot_id:", robot_id, "marker_id:", marker_id, "mode:", motion_mode)
    print(f"Camera: h={cam_height}, ortho={ortho}, near={nearVal}, far={farVal}, mpp={mpp:.6f} m/px")


    res = run_simulation_collect_gt_and_cv_preview(
        robot_id=robot_id,
        dt=dt,
        sim_seconds=sim_seconds,
        view=view, proj=proj,
        img_wh=img_wh,
        meters_per_pixel=mpp,
        v_xy=v_xy,
        realtime=True,
        print_every_s=0.5
    )

    cv2.destroyAllWindows()

    # итоговые числа (средние) — удобно для проверки
    v_gt_mean = float(np.mean(res["v_gt"]))
    v_m_mean = float(np.nanmean(res["v_cv_marker"]))
    v_d_mean = float(np.nanmean(res["v_cv_diff"]))
    v_f_mean = float(np.nanmean(res["v_cv_flow"]))

    print("\n=== SUMMARY ===")
    print(f"t_sim_total  = {res['t_sim_total']:.3f} s")
    print(f"t_wall_total = {res['t_wall_total']:.3f} s")
    print(f"v_gt_mean        = {v_gt_mean:.4f} m/s")
    print(f"v_cv_marker_mean = {v_m_mean:.4f} m/s")
    print(f"v_cv_diff_mean   = {v_d_mean:.4f} m/s")
    print(f"v_cv_flow_mean   = {v_f_mean:.4f} m/s")

    plot_results(
        res["times"],
        res["v_gt"],
        res["v_cv_marker"],
        res["v_cv_diff"],
        res["v_cv_flow"],
        title="GT vs CV (marker + diff + flow)",
        skip_seconds=0.2
    )


    '''


    # --- основной цикл: видно движение + считаем скорость на каждом шаге
    for i in range(n_steps):
        # поддержка движения
        if motion_mode == "kinematic":
            # если используешь кинематику — нужно motion_state = set_robot_motion(...)
            # но для честного constraint лучше velocity
            pass
        else:
            # ВАЖНО: поддерживаем скорость КАЖДЫЙ шаг, иначе физика может её "съедать"
            p.resetBaseVelocity(robot_id, linearVelocity=[v_xy[0], v_xy[1], 0.0], angularVelocity=[0, 0, 0])

        # шаг физики
        p.stepSimulation()
        t_sim += dt

        # ground truth скорость (по позиции)
        cur_xy, v = gt_step(robot_id, prev_xy, dt)
        prev_xy = cur_xy

        times[i] = t_sim
        v_gt[i] = v

        # видео/preview
        frame = grab_frame_bgr(view, proj, img_wh, img_wh)
        cv2.putText(frame, f"t={t_sim:.2f}s  v_gt={v:.3f} m/s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("TopDown", frame)

        # печать раз в ~0.5 сек
        if (i % int(0.5 / dt)) == 0:
            print(f"t={t_sim:6.2f}s | v_gt={v:7.3f} m/s")

        # ESC — выйти
        if cv2.waitKey(1) == 27:
            break

        # чтобы глазами видеть движение (и чтобы wall-time был близок к sim-time)
        time.sleep(dt)

    t_wall = time.perf_counter() - t0_wall
    cv2.destroyAllWindows()

    # итоги
    v_mean = float(np.mean(v_gt[:i+1]))
    v_min = float(np.min(v_gt[:i+1]))
    v_max = float(np.max(v_gt[:i+1]))
    rtf = (t_sim / t_wall) if t_wall > 0 else float("inf")

    print("\n=== SUMMARY ===")
    print(f"t_sim_total  = {t_sim:.3f} s")
    print(f"t_wall_total = {t_wall:.3f} s")
    print(f"RTF          = {rtf:.3f} (sim_seconds / real_seconds)")
    print(f"v_gt_mean    = {v_mean:.4f} m/s")
    print(f"v_gt_min/max = {v_min:.4f} / {v_max:.4f} m/s")

    # times и v_gt — готовы для графика
    # (график сделаем на следующем шаге)

    '''


if __name__ == "__main__":
    main()