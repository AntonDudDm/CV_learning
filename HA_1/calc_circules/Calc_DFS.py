from collections import deque
import cv2

from pathlib import Path
import os

from me_bench import bench

def count_components_exact_color_rgb(image_bgr, target_rgb=(255, 0, 255)):
    
    """
    Считает компоненты связности пикселей, которые имеют ТОЧНО заданный цвет target_rgb (RGB),
    используя DFS/BFS (здесь стек/дек).
    """

    if image_bgr is None:
        raise ValueError("Не удалось прочитать изображение. Проверь путь к файлу.")

    h, w, ch = image_bgr.shape
    if ch != 3:
        raise ValueError("Ожидалось 3 канала (BGR).")

    tr, tg, tb = target_rgb  # target in RGB

    visited = [[False] * w for _ in range(h)]


    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    

    def is_target(y, x):
        b, g, r = image_bgr[y][x]
        return (r == tr) and (g == tg) and (b == tb)

    components = 0

    for y in range(h):
        row_v = visited[y]
        for x in range(w):
            if row_v[x]:
                continue
            if not is_target(y, x):
                continue

            # нашли новый кружок (новую компоненту)
            components += 1

            # DFS (можно deque как стек)
            stack = [(y, x)]
            visited[y][x] = True

            while stack:
                cy, cx = stack.pop()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny][nx]:
                        if is_target(ny, nx):
                            visited[ny][nx] = True
                            stack.append((ny, nx))
                        else:
                            visited[ny][nx] = True  # помечаем, чтобы второй раз не проверять

    return components


def main():
    img_path = "circles.png"  
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
    # Считаем компоненты точного цвета (255,0,255) в RGB

    ## Замер скорости алгоритма
    bench(lambda: count_components_exact_color_rgb(img, target_rgb=(255, 0, 255)),
          n=5, warmup=1)
    
    ## Результат
    n = count_components_exact_color_rgb(img, target_rgb=(255, 0, 255))
    print(f"Magenta circles count: {n}")


if __name__ == "__main__":
    main()
