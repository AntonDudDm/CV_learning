import time
import numpy as np
import cv2
import pybullet as p


# --------------------------
# 1) Камера: fov, матрицы
# --------------------------
def fy_to_fov_deg(height_px: int, fy_px: float) -> float:
    fov_rad = 2.0 * np.arctan(height_px / (2.0 * fy_px))
    return float(np.degrees(fov_rad))


def mat4_from_pybullet(m16):
    """
    PyBullet возвращает 16 чисел в стиле OpenGL (column-major).
    Чтобы корректно умножать в numpy, делаем reshape(order='F').
    """
    return np.array(m16, dtype=np.float64).reshape((4, 4), order='F')


# --------------------------
# 2) Pixel + depth -> World
# --------------------------
def pixel_to_world(u, v, depth_buf, width, height, proj_m16, view_m16):
    """
    Реализация ровно по идее со слайда 16:
    pixel -> NDC -> clip -> invProj -> view -> invView -> world
    depth_buf: значение из depth buffer (0..1)
    """
    proj = mat4_from_pybullet(proj_m16)
    view = mat4_from_pybullet(view_m16)
    inv_proj = np.linalg.inv(proj)
    inv_view = np.linalg.inv(view)

    # NDC: x in [-1,1], y in [-1,1] (y вверх!)
    x_ndc = (2.0 * u / (width - 1)) - 1.0
    y_ndc = 1.0 - (2.0 * v / (height - 1))

    # z in [-1,1] из depth buffer [0,1]
    z_ndc = 2.0 * depth_buf - 1.0

    clip = np.array([x_ndc, y_ndc, z_ndc, 1.0], dtype=np.float64)

    view_h = inv_proj @ clip
    view_h /= view_h[3]

    world_h = inv_view @ view_h
    world_h /= world_h[3]

    return world_h[:3]

def filter_by_curvature(pts, step=6, min_angle_deg=10.0):
    """
    pts: (N,2) точки контура (int)
    Возвращает точки, где есть заметный поворот (дуги), прямые боковины отбрасываем.
    """
    pts = np.asarray(pts)
    N = len(pts)
    if N < 2*step + 5:
        return pts

    keep = []
    # Для прямой cos~1. Хотим отбрасывать почти прямые => оставлять где cos < thr.
    # Угол > ~min_angle_deg  <=>  cos < cos(min_angle_deg).
    thr = np.cos(np.deg2rad(min_angle_deg))

    for i in range(step, N-step):
        p0 = pts[i-step].astype(np.float64)
        p1 = pts[i].astype(np.float64)
        p2 = pts[i+step].astype(np.float64)

        v1 = p1 - p0
        v2 = p2 - p1
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue

        cosang = float(np.dot(v1, v2) / (n1*n2))
        # cosang ~ 1 => почти прямая, оставляем только "поворотные" точки:
        if cosang < thr:
            keep.append(pts[i])

    if len(keep) < 5:
        return pts
    return np.array(keep, dtype=np.int32)

def split_top_bottom(pts, top_frac=0.35, bottom_frac=0.35):
    pts = np.asarray(pts)
    y = pts[:, 1]
    ymin, ymax = int(y.min()), int(y.max())
    h = max(1, ymax - ymin)
    y_top = ymin + int(top_frac * h)
    y_bot = ymax - int(bottom_frac * h)
    top = pts[y <= y_top]
    bot = pts[y >= y_bot]
    return top, bot

def find_top_ellipse_center_debug(rgb_bgr, depth_map=None, min_area=800,
                                  step=6, min_angle_deg=10.0,
                                  top_frac=0.45,
                                  pick_top2=True):
    """
    Возвращает:
      - (u,v) центра (пока как среднее двух лучших эллипсов, если найдены)
      - ellipses: список эллипсов кандидатов (outer/inner)
      - debug_images: dict с картинками:
           'edges'
           'contours_raw'       (1) контуры ДО фильтра кривизны
           'contours_curv'      (2) все дуги ПОСЛЕ фильтра кривизны
           'top2_arcs'          (3) две самые крупные дуги сверху (inner/outer кандидаты)
    """

    H, W = rgb_bgr.shape[:2]

    # --- как у тебя: gray + clahe + blur ---
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # --- Canny ---
    vmed = np.median(gray)
    edges = cv2.Canny(gray, 90, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # ========== (1) Визуализация контуров ДО фильтрации ==========
    vis_raw = np.zeros_like(rgb_bgr)
    cv2.drawContours(vis_raw, contours, -1, (255, 255, 255), 1)

    # Для устойчивого "верхнего" выбора ограничим ROI по y:
    y_cut = int(top_frac * H)

    # ========== (2) Визуализация ВСЕХ дуг ПОСЛЕ фильтрации ==========
    vis_curv = np.zeros_like(rgb_bgr)

    # Собираем кандидаты дуг верхнего отверстия:
    # будем оценивать дугу по числу точек + площади bbox и располагаться "сверху"
    arc_candidates = []

    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < min_area or len(c) < 20:
            continue

        pts = c.reshape(-1, 2)

        arc_pts = filter_by_curvature(pts, step=step, min_angle_deg=min_angle_deg)
        if arc_pts is None or len(arc_pts) < 5:
            continue

        # рисуем все дуги
        for pxy in arc_pts[::2]:
            vis_curv[pxy[1], pxy[0]] = (255, 255, 255)

        # берём только верхние дуги (для отверстия)
        top_pts = arc_pts[arc_pts[:, 1] <= y_cut]
        if len(top_pts) < 20:
            continue

        # оценка "крупности" дуги сверху
        x, y, w, h = cv2.boundingRect(top_pts.reshape(-1, 1, 2))
        score = len(top_pts) * (w * h)

        print(f"[c{idx}] area={area:.1f} arc={len(arc_pts)} top={len(top_pts)} bbox={w}x{h} score={score:.1e}")

        arc_candidates.append({
            "score": score,
            "top_pts": top_pts,
            "all_arc_pts": arc_pts,
            "src_contour_idx": idx
        })

    if len(arc_candidates) == 0:
        return {
            "edges": edges,
            "contours_raw": vis_raw,
            "contours_curv": vis_curv,
            "top2_arcs": np.zeros_like(rgb_bgr),
            "note": "no arc candidates"
        }

    arc_candidates.sort(key=lambda d: d["score"], reverse=True)

    # ========== (3) Две самые крупные дуги сверху ==========
    vis_top2 = np.zeros_like(rgb_bgr)

    picked = arc_candidates[:2] if pick_top2 else arc_candidates[:1]

    ellipses = []
    centers = []

    for k, item in enumerate(picked):
        pts2 = item["top_pts"].reshape(-1, 1, 2).astype(np.int32)
        if len(pts2) < 5:
            continue

        ell = cv2.fitEllipse(pts2)
        ellipses.append(ell)

        (cx, cy), (MA, ma), ang = ell
        centers.append((cx, cy))

        # рисуем дуги разными оттенками (чтобы отличать 1 и 2)
        color = (255, 255, 255) if k == 0 else (180, 180, 180)
        for pxy in item["top_pts"][::2]:
            vis_top2[pxy[1], pxy[0]] = color

    # Финальный центр: среднее центров двух эллипсов (если 2 найдены), иначе первый
    if len(centers) == 0:
        u = v = None
    elif len(centers) == 1:
        u = int(round(centers[0][0]))
        v = int(round(centers[0][1]))
    else:
        u = int(round((centers[0][0] + centers[1][0]) / 2.0))
        v = int(round((centers[0][1] + centers[1][1]) / 2.0))

    # Сводная визуализация поверх исходного кадра (для удобства)
    vis_overlay = rgb_bgr.copy()
    for ell in ellipses:
        cv2.ellipse(vis_overlay, ell, (0, 255, 0), 2)
    if u is not None:
        cv2.circle(vis_overlay, (u, v), 6, (0, 0, 255), 2)

    return (u, v), ellipses, {
        "edges": edges,
        "contours_raw": vis_raw,     # (1)
        "contours_curv": vis_curv,   # (2)
        "top2_arcs": vis_top2,       # (3)
        "overlay": vis_overlay
    }



# --------------------------
# Main
# --------------------------
def main():
    # --- PyBullet init ---
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)


    # --- Load URDF ---
    urdf_path = "C:/URDF/Vtulka/urdf/Vtulka.urdf"
    startPos = [0, 0, 0]
    startOrn = p.getQuaternionFromEuler([0, 0, 0])
    vtulka_id = p.loadURDF(urdf_path, startPos, startOrn, useFixedBase=True, globalScaling=1.0)
    p.changeVisualShape(vtulka_id, -1, rgbaColor=[0.4, 0.4, 0.4, 1.0])  # тёмно-серая

    # --- Camera params  ---
    width, height = 1920, 1080
    fy = 1796.1276552044753
    near, far = 0.001, 3.0

    ## Находит Feild of view
    fov = fy_to_fov_deg(height, fy) 
    aspect = width / height

    proj = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=near, farVal=far)

    # Поставили камеру под 45° сверху-сбоку
    eye = [0.0, 0.30, 0.4]
    target = [0.0, 0.0, 0.1]
    up = [0.0, 0.0, 1.0]
    view = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=up)

    # Дадим симуляции шаг
    for i in range(1):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

        img = p.getCameraImage(width, height, 
                            view, 
                            proj, 
                            renderer=p.ER_TINY_RENDERER,
                            lightDirection=[0, 2, 3],             # направление света
                            lightColor=[1, 1, 1],                   # цвет света
                            shadow=1
                            )

        # --- unpack ---

        # raw RGBA (int32)
        rgba = np.reshape(img[2], (height, width, 4)).astype(np.uint8)
        rgb = rgba[:, :, :3]                         # RGB
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)   # OpenCV wants BGR

        ## Глубина
        depth_buf = np.reshape(img[3], (height, width)).astype(np.float32)  # 0..1

        cv2.imwrite(f"rgb.png", bgr)


    

    # depth debug (картинка для отчёта)
    depth_vis = (255 * (depth_buf - depth_buf.min()) / (np.ptp(depth_buf) + 1e-9)).astype(np.uint8)
    cv2.imwrite("depth_vis.png", depth_vis)

    
    res = find_top_ellipse_center_debug(bgr, step=6, min_angle_deg=10, min_area=0, top_frac=0.65)
    if res is None:
        print("ничего не найдено")
    else:
        (u,v), ellipses, dbg = res
        cv2.imwrite("01_edges.png", dbg["edges"])
        cv2.imwrite("02_contours_raw.png", dbg["contours_raw"])
        cv2.imwrite("03_contours_curv.png", dbg["contours_curv"])
        cv2.imwrite("04_top2_arcs.png", dbg["top2_arcs"])
        cv2.imwrite("05_overlay.png", dbg["overlay"])
        print("center:", u, v, "num_ellipses:", len(ellipses))
        




if __name__ == "__main__":
    main()