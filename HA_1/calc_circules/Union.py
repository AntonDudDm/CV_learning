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

    def is_target(y, x):
        b, g, r = image_bgr[y][x]
        return (r == tr) and (g == tg) and (b == tb)

    components = 0

    prev_segments = []
    

    for y in range(h):
        curr_segments = []

        ## Создание current line - массива линий для пересечения с prev
        x=0
        while x < w:

            if not is_target(y, x):
                x+=1
                continue
            
            # Нашли компоненту
            x0 = x

            while x < w and is_target(y, x):
                x+=1

            x1 = x-1

            curr_segments.append((x0,x1))

        
        ## Проверка пересечений:
        ## Если сверзу ни с кем не пересекается - то новая компонента, если пересекается, то продолжение старой уже известной

        
        i = 0 #Указатель на предыдущий сегмент
        for (x0, x1) in curr_segments:

            # пропускаем prev, которые точно слева и не пересекаются даже с допуском delta
            while i < len(prev_segments) and prev_segments[i][1] < x0:
                i += 1
            # никто не пересекся -> новая компонента
            if i == len(prev_segments) or prev_segments[i][0] > x1:
                components += 1


        prev_segments = curr_segments


            

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
