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

# --------------------------
# 2) Pixel + depth -> World
# --------------------------

def mat4_from_pybullet(m16):
    """
    PyBullet: 16 чисел OpenGL column-major.
    Numpy умножение: делаем reshape(order='F').
    """
    return np.array(m16, dtype=np.float64).reshape((4, 4), order='F')

def pixel_depth_to_world(x, y, depth_buf, width, height, proj_m16, view_m16):
    """
    (x,y) - пиксели (как в OpenCV), depth_buf in [0,1] из getCameraImage
    Возвращает world xyz.
    Шаги:
      pixel -> NDC -> clip -> invProj -> view -> invView -> world
    """
    proj = mat4_from_pybullet(proj_m16)
    view = mat4_from_pybullet(view_m16)

    inv_proj = np.linalg.inv(proj)
    inv_view = np.linalg.inv(view)

    # 1) pixel -> NDC
    # x_ndc in [-1,1], y_ndc in [-1,1], y вверх => переворачиваем ось Y
    x_ndc = ((2.0 * x + 1.0) / (width-1)) - 1.0
    y_ndc = 1.0 - ((2.0 * y + 1.0) / (height-1))

    # 2) depth buffer [0,1] -> z_ndc [-1,1]
    z_ndc = 2.0 * float(depth_buf) - 1.0

    # 3) NDC -> clip
    clip = np.array([x_ndc, y_ndc, z_ndc, 1.0], dtype=np.float64)

    # 4) clip -> view (camera) space
    view_h = inv_proj @ clip
    view_h /= view_h[3]

    # 5) view -> world
    world_h = inv_view @ view_h
    world_h /= world_h[3]

    return world_h[:3]


# ---------- helpers ----------
def filter_by_curvature(pts, step=6, min_angle_deg=10.0):
    """
    pts: (N,2) int точки контура.
    Оставляет точки, где контур "поворачивает" (дуга), прямые выкидывает.
    """
    pts = np.asarray(pts)
    N = len(pts)
    if N < 2 * step + 5:
        return pts

    keep = []
    thr = np.cos(np.deg2rad(min_angle_deg))  # чем меньше min_angle, тем больше точек оставим

    for i in range(step, N - step):
        p0 = pts[i - step].astype(np.float64)
        p1 = pts[i].astype(np.float64)
        p2 = pts[i + step].astype(np.float64)

        v1 = p1 - p0
        v2 = p2 - p1
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue

        cosang = float(np.dot(v1, v2) / (n1 * n2))
        # прямые: cosang ~ 1, дуги: cosang заметно меньше
        if cosang < thr:
            keep.append(pts[i])

    if len(keep) < 5:
        return pts
    return np.array(keep, dtype=np.int32)




# --------------------------
# 3) Find elipses
# --------------------------
def find_top_ellipse_center_debug(rgb_bgr, depth_map=None,
                                  step=6, min_angle_deg=10.0,
                                  top_frac=0.65,
                                  min_top_pts=12,
                                  max_ratio=10.0,
                                  close_edges=True,
                                  debug_print=False):
    """
    Возвращает:
      (u,v), [outer_ellipse, inner_ellipse], debug_images

    Важно: собирает внутренний/внешний эллипс, даже если дуги разбросаны по разным контурам.
    """

    H, W = rgb_bgr.shape[:2]

    #Перевод в ЧБ
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    #Предобработка изображения
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    #Canny - карта границ
    edges = cv2.Canny(gray, 90, 120)

    # Склейка границ
    if close_edges:
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    # Поиск конутров
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # (1) raw contours
    vis_raw = np.zeros_like(rgb_bgr)
    cv2.drawContours(vis_raw, contours, -1, (255, 255, 255), 1)

    # bbox самого большого контура -> y_cut (стабильнее, чем top_frac*H)
    c_big = max(contours, key=cv2.contourArea)
    xB, yB, wB, hB = cv2.boundingRect(c_big)
    # обрезка по высоте (не брать нижний полуэллипс)
    y_cut = yB + int(top_frac * hB)

    # (2) all arcs after curvature
    vis_curv = np.zeros_like(rgb_bgr)

    # кандидаты-фрагменты эллипсов
    frags = []
    for idx, c in enumerate(contours):
        if len(c) < 20:
            continue

        ## ФИЛЬТРАЦИЯ ПО КРИВИЗНЕ убираем все прямые линии
        pts = c.reshape(-1, 2)
        arc_pts = filter_by_curvature(pts, step=step, min_angle_deg=min_angle_deg)
        if arc_pts is None or len(arc_pts) < 5:
            continue

        # рисуем дуги после кривизны
        for pxy in arc_pts[::2]:
            vis_curv[pxy[1], pxy[0]] = (255, 255, 255)

        # верхние точки дуги
        top_pts = arc_pts[arc_pts[:, 1] <= y_cut]
        if len(top_pts) < min_top_pts:
            continue

        # OpenCV формат массива
        cnt = top_pts.reshape(-1, 1, 2).astype(np.int32)
        if len(cnt) < 5:
            continue

        # Эллипс по данному кусу конутра
        ell = cv2.fitEllipse(cnt)
        (cx, cy), (MA, ma), ang = ell

        # фильтры от мусора
        ratio = max(MA, ma) / (min(MA, ma) + 1e-9)
        if ratio > max_ratio:
            continue

        area_ell = float(MA * ma)
        score = float(len(top_pts) / (ratio + 1e-9))  # простая "надёжность"

        frags.append({
            "idx": idx,
            "ell": ell,
            "center": (float(cx), float(cy)),
            "area": area_ell,
            "score": score,
            "pts": top_pts
        })

        if debug_print:
            x, y, w, h = cv2.boundingRect(cnt)
            print(f"[frag c{idx}] top={len(top_pts)} ratio={ratio:.2f} ell_area={area_ell:.1f} bbox={w}x{h} score={score:.2f}")

    if len(frags) < 2:
        return None

    # ---- A) Оставляем доминирующую группу по общему центру ----
    # берём seed = самый "надёжный" фрагмент, вокруг него собираем соседей по центру
    frags.sort(key=lambda d: d["score"], reverse=True)
    seed = frags[0]
    c0x, c0y = seed["center"]
    # порог по расстоянию до центра: привязан к размеру seed-эллипса
    (s_cx, s_cy), (s_MA, s_ma), _ = seed["ell"]
    dist_thr = 0.25 * max(s_MA, s_ma) + 10.0  # +10 пикс для запаса

    good = []
    for f in frags:
        cx, cy = f["center"]
        if (cx - c0x) ** 2 + (cy - c0y) ** 2 <= dist_thr ** 2:
            good.append(f)

    if len(good) < 2:
        # fallback: берём всё, но это редко
        good = frags

    # ---- B) Делим good на inner/outer по площади эллипса ----
    areas = np.array([g["area"] for g in good], dtype=np.float64)
    order = np.argsort(areas)
    areas_sorted = areas[order]

    # ищем самый большой разрыв между соседями
    gaps = areas_sorted[1:] - areas_sorted[:-1]
    if len(gaps) == 0:
        return None
    j = int(np.argmax(gaps))
    thr = 0.5 * (areas_sorted[j] + areas_sorted[j + 1])

    inner_frags = [g for g in good if g["area"] <= thr]
    outer_frags = [g for g in good if g["area"] > thr]

    # если порог плохой (все в одну сторону), fallback на медиану
    if len(inner_frags) == 0 or len(outer_frags) == 0:
        thr = float(np.median(areas))
        inner_frags = [g for g in good if g["area"] <= thr]
        outer_frags = [g for g in good if g["area"] > thr]

    if len(inner_frags) == 0 or len(outer_frags) == 0:
        return None

    # ---- C) Объединяем точки и фитим финальные эллипсы ----
    inner_pts = np.vstack([g["pts"] for g in inner_frags]).astype(np.int32)
    outer_pts = np.vstack([g["pts"] for g in outer_frags]).astype(np.int32)

    if len(inner_pts) < 5 or len(outer_pts) < 5:
        return None

    ell_in = cv2.fitEllipse(inner_pts.reshape(-1, 1, 2))
    ell_out = cv2.fitEllipse(outer_pts.reshape(-1, 1, 2))

    # упорядочим
    Ain = ell_in[1][0] * ell_in[1][1]
    Aout = ell_out[1][0] * ell_out[1][1]
    if Ain > Aout:
        ell_in, ell_out = ell_out, ell_in
        inner_pts, outer_pts = outer_pts, inner_pts

    # центр как среднее центров
    (xo, yo), _, _ = ell_out
    (xi, yi), _, _ = ell_in
    u = int(round((xo + xi) / 2.0))
    v = int(round((yo + yi) / 2.0))

    # ---- DEBUG картинка: фрагменты, кластеры, финал ----
    vis_clusters = np.zeros_like(rgb_bgr)
    # inner белым, outer серым
    for p in inner_pts[::2]:
        vis_clusters[p[1], p[0]] = (255, 255, 255)
    for p in outer_pts[::2]:
        vis_clusters[p[1], p[0]] = (180, 180, 180)
    cv2.ellipse(vis_clusters, ell_out, (0, 255, 0), 2)
    cv2.ellipse(vis_clusters, ell_in, (0, 255, 0), 2)
    cv2.circle(vis_clusters, (u, v), 6, (0, 0, 255), 2)

    vis_overlay = rgb_bgr.copy()
    cv2.ellipse(vis_overlay, ell_out, (0, 255, 0), 2)
    cv2.ellipse(vis_overlay, ell_in, (0, 255, 0), 2)
    cv2.circle(vis_overlay, (u, v), 6, (0, 0, 255), 2)

    # (3) все верхние точки, которые вообще попали в frags
    vis_top_all = np.zeros_like(rgb_bgr)
    all_pts = np.vstack([g["pts"] for g in good]).astype(np.int32)
    for p in all_pts[::1]:
        vis_top_all[p[1], p[0]] = (255, 255, 255)

    return (u, v), [ell_out, ell_in], {
        "edges": edges,
        "contours_raw": vis_raw,
        "contours_curv": vis_curv,
        "top_all": vis_top_all,
        "clusters": vis_clusters,
        "overlay": vis_overlay
    }


# --------------------------
# 4) Find depth
# --------------------------


def robust_depth_at(depth_map, x, y, r=4, mode="p10"):
    """
    depth_map: (H,W) float32 depth buffer 0..1 (как из pybullet)
    x,y: int pixel
    r: радиус окна (2r+1)x(2r+1)
    mode:
      - "min"   : самый ближний (агрессивно)
      - "p10"   : 10-й перцентиль (обычно лучше для кромки)
      - "median"
    """
    H, W = depth_map.shape
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    y0, y1 = max(0, y - r), min(H, y + r + 1)

    patch = depth_map[y0:y1, x0:x1].astype(np.float64)
    vals = patch[np.isfinite(patch)]

    # отсекаем "пустоту"/дальний фон
    vals = vals[(vals > 0.0) & (vals < 0.999999)]
    if vals.size == 0:
        return None

    if mode == "min":
        return float(np.min(vals))
    if mode == "median":
        return float(np.median(vals))
    return float(np.quantile(vals, 0.10))

def _major_axis_endpoints(ellipse, scale=0.92):
    """
    ellipse: ((cx,cy),(MA,ma),ang_deg) как у cv2.fitEllipse
    Возвращает две точки (x,y) на концах большой оси.
    scale < 1 уводит точку внутрь, чтобы не попадать в край/искажения.
    """
    (cx, cy), (MA, ma), ang = ellipse
    theta = np.deg2rad(ang)

    # гарантируем, что theta указывает вдоль большой оси
    major = MA
    theta_major = theta
    if ma > MA:
        major = ma
        theta_major = theta + np.pi/2

    a = 0.5 * major * scale
    dx, dy = np.cos(theta_major), np.sin(theta_major)

    p_plus  = (cx + a*dx, cy + a*dy)
    p_minus = (cx - a*dx, cy - a*dy)
    return [p_plus, p_minus]

def _pair_by_nearest(A2, B2):
    """
    A2=[A1,A2], B2=[B1,B2] (points)
    Возвращает пары (A1,Bk), (A2,Bother) по ближайшему соответствию.
    """
    A1, A2p = A2
    B1, B2p = B2
    d11 = (A1[0]-B1[0])**2 + (A1[1]-B1[1])**2
    d12 = (A1[0]-B2p[0])**2 + (A1[1]-B2p[1])**2
    if d11 <= d12:
        return (A1, B1), (A2p, B2p)
    else:
        return (A1, B2p), (A2p, B1)

def pick_two_rim_points_between_ellipses(depth_map, outer_ellipse, inner_ellipse, center_uv,
                                        between_alpha=0.5,
                                        scale_outer=0.92, scale_inner=0.92,
                                        win_r=4, depth_mode="p10"):
    """
    Возвращает две точки на большой оси:
      P1=(x1,y1,d1), P2=(x2,y2,d2)
    где (x,y) лежит между внутренним и внешним эллипсом с коэффициентом between_alpha:
      P = inner + alpha*(outer-inner)
      alpha=0.5 => середина толщины
      alpha=0.7 => ближе к внешнему
      alpha=0.3 => ближе к внутреннему

    center_uv нужен только для sanity/будущей отладки (можно не использовать).
    """

    # 1) концы большой оси у outer и inner
    O = _major_axis_endpoints(outer_ellipse, scale=scale_outer)
    I = _major_axis_endpoints(inner_ellipse, scale=scale_inner)

    # 2) сопоставляем “стороны” (чтобы не перепутать плюс/минус)
    (O1, I1), (O2, I2) = _pair_by_nearest(O, I)

    # 3) точка между inner и outer на каждой стороне
    a = float(between_alpha)
    M1 = (I1[0] + a*(O1[0]-I1[0]), I1[1] + a*(O1[1]-I1[1]))
    M2 = (I2[0] + a*(O2[0]-I2[0]), I2[1] + a*(O2[1]-I2[1]))

    H, W = depth_map.shape
    pts_out = []
    for (xf, yf) in (M1, M2):
        x = int(round(xf))
        y = int(round(yf))
        if not (0 <= x < W and 0 <= y < H):
            return None
        d = robust_depth_at(depth_map, x, y, r=win_r, mode=depth_mode)
        if d is None:
            return None
        pts_out.append((x, y, d))

    return pts_out[0], pts_out[1]


def interpolate_depth_at_center(P1, P2, center_uv, clamp=True):
    """
    P1=(x1,y1,d1), P2=(x2,y2,d2)
    center_uv=(x0,y0)
    Возвращает (x0,y0,depth0) и параметр t.

    depth0 = d1 + t*(d2-d1), где t — проекция центра на отрезок P1->P2 в пикселях.
    """
    x1, y1, d1 = P1
    x2, y2, d2 = P2
    x0, y0 = center_uv
    x0 = float(x0); y0 = float(y0)

    vx = float(x2 - x1)
    vy = float(y2 - y1)
    denom = vx*vx + vy*vy
    if denom < 1e-9:
        return None

    # t = proj((C-P1) onto (P2-P1)) / ||P2-P1||^2
    t = ((x0 - x1)*vx + (y0 - y1)*vy) / denom
    if clamp:
        t = max(0.0, min(1.0, t))

    depth0 = float(d1 + t*(d2 - d1))
    return (int(round(x0)), int(round(y0)), depth0), t


def draw_depth_debug_07(bgr, center_uv, ell_out, ell_in, P1, P2, Cxyz, t,
                        color_outer=(0, 255, 0),
                        color_inner=(0, 255, 0),
                        color_pts=(255, 0, 0),
                        color_center=(0, 0, 255)):
    """
    07:
      - outer/inner ellipse
      - P1/P2 + линия между ними
      - центр
      - подписи d1,d2,d0,t без налезания
    """
    vis = bgr.copy()
    H, W = vis.shape[:2]

    # эллипсы
    cv2.ellipse(vis, ell_out, color_outer, 2)
    cv2.ellipse(vis, ell_in,  color_inner, 2)

    x0, y0 = int(round(center_uv[0])), int(round(center_uv[1]))
    x1, y1, d1 = P1
    x2, y2, d2 = P2
    xc, yc, d0 = Cxyz

    # точки и линия
    cv2.circle(vis, (x1, y1), 7, color_pts, 2)
    cv2.circle(vis, (x2, y2), 7, color_pts, 2)
    cv2.line(vis, (x1, y1), (x2, y2), color_pts, 1)

    # центр
    cv2.circle(vis, (x0, y0), 7, color_center, 2)

    # ---- умные подписи без налезания ----
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.6
    thk = 2
    pad = 4
    line_gap = 6

    placed = []  # список прямоугольников [x0,y0,x1,y1] уже поставленных

    def rects_intersect(a, b):
        return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

    def place_label(text, anchor_xy, side, text_color):
        ax, ay = anchor_xy
        (tw, th), base = cv2.getTextSize(text, font, fs, thk)

        # начальная позиция (baseline)
        if side == "left":
            x = ax - tw - 16
            y = ay
        elif side == "right":
            x = ax + 16
            y = ay
        elif side == "above":
            x = ax - tw // 2
            y = ay - 16
        else:  # "below"
            x = ax - tw // 2
            y = ay + th + 16

        # clamp внутрь кадра (для baseline y)
        x = int(np.clip(x, pad, W - tw - pad))
        y = int(np.clip(y, th + pad, H - pad))

        # прямоугольник под фон (в координатах изображения)
        def make_rect(xb, yb):
            # yb — baseline
            return [xb - pad, yb - th - pad, xb + tw + pad, yb + base + pad]

        rect = make_rect(x, y)

        # если пересекается с уже размещёнными — сдвигаем вниз
        max_tries = 40
        tries = 0
        while any(rects_intersect(rect, r) for r in placed) and tries < max_tries:
            y += (th + base + line_gap)
            # если ушли вниз — попробуем прыгнуть вверх относительно якоря
            if y > H - pad:
                y = int(np.clip(ay - 16 - tries * (th + base + line_gap), th + pad, H - pad))
            x = int(np.clip(x, pad, W - tw - pad))
            rect = make_rect(x, y)
            tries += 1

        placed.append(rect)

        # фон-плашка
        #cv2.rectangle(vis, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), -1)
        # текст
        cv2.putText(vis, text, (x, y), font, fs, text_color, thk, cv2.LINE_AA)

        # leader line (якорь -> ближайшая точка прямоугольника)
        bx = int(np.clip(ax, rect[0], rect[2]))
        by = int(np.clip(ay, rect[1], rect[3]))
        cv2.line(vis, (ax, ay), (bx, by), text_color, 1)

    # подписи: размещаем в разных предпочтительных сторонах
    place_label(f"P1  d1={d1:.6f}", (x1, y1), "left",  color_pts)
    place_label(f"P2  d2={d2:.6f}", (x2, y2), "right", color_pts)
    place_label(f"C   d0={d0:.6f}  t={t:.3f}", (x0, y0), "above", color_center)

    return vis

# --------------------------
# 5) debug_08 pic
# --------------------------

def draw_world_point_debug_08(overlay_bgr, pixel_xy, world_xyz,
                              out_path="08_overlay_center.png",
                              point_color=(0, 0, 255),
                              font_scale=0.6,
                              thickness=2):
    """
    overlay_bgr: картинка (BGR), например dbg["overlay"]
    pixel_xy: (x, y) в пикселях
    world_xyz: (X, Y, Z) в мировых координатах (метры)
    Рисует точку и подпись "(X,Y,Z) m" с 3 знаками, сохраняет out_path.
    """
    vis = overlay_bgr.copy()
    H, W = vis.shape[:2]

    px = int(round(pixel_xy[0]))
    py = int(round(pixel_xy[1]))

    cv2.circle(vis, (px, py), 9, point_color, 2)

    label = f"({world_xyz[0]:.3f}, {world_xyz[1]:.3f}, {world_xyz[2]:.3f}) m"

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th_text), base = cv2.getTextSize(label, font, font_scale, thickness)

    # позиция подписи (справа-сверху от точки)
    tx = px + 12
    ty = py - 12

    # если вылезает за правый край — переносим влево
    if tx + tw + 8 > W:
        tx = px - 12 - tw
    # если вылезает за верх — переносим вниз
    if ty - th_text - 8 < 0:
        ty = py + 12 + th_text

    pad = 4
    x0 = max(0, tx - pad)
    y0 = max(0, ty - th_text - pad)
    x1 = min(W - 1, tx + tw + pad)
    y1 = min(H - 1, ty + base + pad)

    #cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.putText(vis, label, (tx, ty), font, font_scale, point_color, thickness, cv2.LINE_AA)

    cv2.imwrite(out_path, vis)
    return vis


# --------------------------
# Main
# --------------------------
def main():
    # --- PyBullet init ---
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)


    # --- Load URDF ---
    urdf_path = "C:/URDF/Vtulka_2/urdf/Vtulka_2.urdf"
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

    res = find_top_ellipse_center_debug(
        bgr,
        step=6,
        min_angle_deg=10,
        top_frac=0.65,
        min_top_pts=10,
        max_ratio=12.0,
        close_edges=False,
        debug_print=True
    )

    if res is None:
        print("ничего не найдено")
    else:
        (u, v), ellipses, dbg = res
        cv2.imwrite("01_edges.png", dbg["edges"])
        cv2.imwrite("02_contours_raw.png", dbg["contours_raw"])
        cv2.imwrite("03_contours_curv.png", dbg["contours_curv"])
        cv2.imwrite("04_top_all.png", dbg["top_all"])
        cv2.imwrite("05_clusters.png", dbg["clusters"])
        cv2.imwrite("06_overlay.png", dbg["overlay"])
        print("center:", u, v, "ellipses:", len(ellipses))
        

    # --------------------------
    # 07) Depth по двум точкам на большой оси (между inner/outer)
    # --------------------------
    ell_out, ell_in = ellipses[0], ellipses[1]
    center_uv = (u, v)

    pts2 = pick_two_rim_points_between_ellipses(
        depth_map=depth_buf,
        outer_ellipse=ell_out,
        inner_ellipse=ell_in,
        center_uv=center_uv,
        between_alpha=0.7,     # середина толщины между эллипсами
        scale_outer=0.92,      # уводим от края
        scale_inner=0.92,
        win_r=5,               # окно для depth
        depth_mode="p10"       # устойчиво, не так шумно как min
    )

    if pts2 is None:
        print("07: не удалось взять depth в точках на кромке (pts2=None)")
    else:
        P1, P2 = pts2
        out = interpolate_depth_at_center(P1, P2, center_uv, clamp=True)
        if out is None:
            print("07: не удалось интерполировать depth0 (отрезок вырожден)")
        else:
            Cxyz, t = out  # (x0,y0,depth0), t
            print("07: P1=", P1, "P2=", P2, "Center(x,y,d)=", Cxyz, "t=", t)

            vis07 = draw_depth_debug_07(bgr, center_uv, ell_out, ell_in, P1, P2, Cxyz, t)
            cv2.imwrite("07_depth_line.png", vis07)


    # --------------------------
    # 08) Pixel(x0,y0,depth0) -> World XYZ (по лекции)
    # --------------------------
    x0p, y0p, depth0 = Cxyz  # depth0 именно depth buffer [0..1]

    Pw = pixel_depth_to_world(
        x=x0p, y=y0p, depth_buf=depth0,
        width=width, height=height,
        proj_m16=proj, view_m16=view
    )

    print("08: world center =", Pw)

    # Для отчёта/проверки сохраним в файл
    with open("08_world_point.txt", "w", encoding="utf-8") as f:
        f.write(f"x0={x0p} y0={y0p} depth0={depth0:.9f}\n")
        f.write(f"world_x={Pw[0]:.9f} world_y={Pw[1]:.9f} world_z={Pw[2]:.9f}\n")

    # Pw = pixel_depth_to_world(...)
    draw_world_point_debug_08(
        overlay_bgr=dbg["overlay"],
        pixel_xy=(x0p, y0p),
        world_xyz=Pw,
        out_path="08_overlay_center.png"
    )


if __name__ == "__main__":
    main()