import cv2
from me_bench import bench

def count_magenta_via_white_dfs(image_bgr, target_rgb=(255, 0, 255), white_rgb=(255, 255, 255)):
    if image_bgr is None:
        raise ValueError("Не удалось прочитать изображение. Проверь путь к файлу.")

    h, w, ch = image_bgr.shape
    if ch != 3:
        raise ValueError("Ожидалось 3 канала (BGR).")

    # target_rgb = (R,G,B). В image_bgr хранится (B,G,R).
    tr, tg, tb = target_rgb
    wr, wg, wb = white_rgb

    def is_magenta(y, x):
        b, g, r = image_bgr[y, x]
        return (r == tr) and (g == tg) and (b == tb)

    def is_white(y, x):
        b, g, r = image_bgr[y, x]
        return (r == wr) and (g == wg) and (b == wb)

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-связность

    visited_white = [[False] * w for _ in range(h)]
    visited_magenta = [[False] * w for _ in range(h)]

    components = 0

    # Чтобы не зависеть от того, белый ли (0,0), стартуем с любого белого пикселя:
    # (Можно заменить на (0,0), если фон гарантированно белый по краям.)
    start_points = []
    for y in range(h):
        for x in range(w):
            if is_white(y, x):
                start_points.append((y, x))
                break
        if start_points:
            break

    if not start_points:
        return 0  # белого нет => не по чему ходить

    # Если белые области могут быть разорваны “стенами”, то одной стартовой точки мало.
    # Тогда сделаем так: внешний цикл найдёт все белые компоненты и для каждой запустит DFS.
    # (Это всё равно “ходим по изображению DFS-ом”, просто иногда перезапускаем.)
    for sy in range(h):
        for sx in range(w):
            if not is_white(sy, sx) or visited_white[sy][sx]:
                continue

            # DFS по белой области
            stack = [(sy, sx)]
            visited_white[sy][sx] = True

            while stack:
                cy, cx = stack.pop()

                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue

                    # Ходим только по белому
                    if is_white(ny, nx) and not visited_white[ny][nx]:
                        visited_white[ny][nx] = True
                        stack.append((ny, nx))
                        continue

                    # Если упёрлись в magenta-соседа — запускаем DFS заливку magenta-компоненты
                    if is_magenta(ny, nx) and not visited_magenta[ny][nx]:
                        components += 1

                        mstack = [(ny, nx)]
                        visited_magenta[ny][nx] = True

                        while mstack:
                            my, mx = mstack.pop()
                            for ddy, ddx in neighbors:
                                ty, tx = my + ddy, mx + ddx
                                if 0 <= ty < h and 0 <= tx < w:
                                    if is_magenta(ty, tx) and not visited_magenta[ty][tx]:
                                        visited_magenta[ty][tx] = True
                                        mstack.append((ty, tx))

            # конец DFS по белой компоненте

    return components


def main():
    img_path = "circles.png"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    bench(lambda: count_magenta_via_white_dfs(img, target_rgb=(255, 0, 255)),
          n=5, warmup=1)

    n = count_magenta_via_white_dfs(img, target_rgb=(255, 0, 255))
    print(f"Magenta circles count: {n}")


if __name__ == "__main__":
    main()
