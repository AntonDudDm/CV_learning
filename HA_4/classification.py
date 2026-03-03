# ha4_train_convex_classifier.py
# Один файл: 1) загрузка + сжатие + примеры, 2) train/test split, 3) convex optimization (hard-margin -> fallback soft-margin),
# 4) метрики на train/test.
#
# Требования:
#   pip install pybullet opencv-python numpy cvxpy
#
# Ожидаемая структура датасета (рядом со скриптом):
#   HA_4/
#     ha4_train_convex_classifier.py
#     dataset/
#       labels.txt
#       class1/...
#       class2/...
#
# labels.txt строки вида:
#   class1/c1_0000.png 1
#   class2/c2_0000.png -1

import os
import math
from pathlib import Path

import numpy as np
import cv2
import cvxpy as cp


# ==========================
# ПАРАМЕТРЫ (меняй тут)
# ==========================
DATASET_DIRNAME = "dataset"
LABELS_FILENAME = "labels.txt"

# препроцессинг
RESIZE_WH = (64, 64)       # (W, H) для cv2.resize
#BINARY_THRESHOLD = 30      # 0..255 (после grayscale). Поставь 1..5 если картинки уже почти бинарные.
BINARY_THRESHOLD = None  # Otsu

# split
TEST_RATIO = 0.25
RANDOM_SEED = 0

# оптимизация
TRY_HARD_MARGIN = True
SOFT_MARGIN_C = 10.0       # чем больше C, тем сильнее штраф за ошибки/нарушение margin
SOLVER = "SCS"             # "SCS" обычно есть. Если есть OSQP/ECOS, можно заменить.

# вывод/артефакты
RESULTS_DIRNAME = "results"
# ==========================

def imread_unicode(path: Path, flags=cv2.IMREAD_GRAYSCALE):
    """
    Надёжный cv2.imread для путей с кириллицей на Windows.
    """
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)   # читает байты независимо от Unicode
    img = cv2.imdecode(data, flags)
    return img

def imwrite_unicode(path: Path, bgr: np.ndarray) -> None:
    """Надёжная запись PNG в любой путь на Windows (в т.ч. с кириллицей)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    buf.tofile(str(path))


def preprocess_image_to_feature(gray: np.ndarray, resize_wh=(64, 64), thr=None):
    """
    gray: uint8 grayscale (H,W)

    Возвращает:
      feat: float32 vector (d,) in {0,1}
      vis_gray: uint8 resized grayscale (H_small,W_small)
      vis_bin:  uint8 resized binary image (0/255) for debug
    """
    vis_gray = cv2.resize(gray, resize_wh, interpolation=cv2.INTER_AREA)

    # порог: если thr=None -> Otsu (рекомендую)
    if thr is None:
        _, vis_bin = cv2.threshold(vis_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, vis_bin = cv2.threshold(vis_gray, int(thr), 255, cv2.THRESH_BINARY)

    # авто-инверсия: хотим фон чёрный, объект белый
    b = 4
    border = np.concatenate([
        vis_bin[:b, :].ravel(),
        vis_bin[-b:, :].ravel(),
        vis_bin[:, :b].ravel(),
        vis_bin[:, -b:].ravel(),
    ])
    if np.mean(border) > 127:
        vis_bin = 255 - vis_bin

    feat = (vis_bin > 0).astype(np.float32).flatten()
    return feat, vis_gray, vis_bin


def load_dataset_from_labels(dataset_root: Path, labels_file: Path, resize_wh=(64, 64), thr=30):
    """
    Читает labels.txt, грузит изображения, делает preprocess, возвращает:
      X: (N,d) float32
      y: (N,) int32 in {-1,+1}
      paths: list[Path] абсолютные пути к исходным картинкам
      y_raw: исходные метки (int) из файла (можно 1/-1)
      previews: dict {+1: (src_path, vis_bin_img), -1: (...)} первые примеры классов
    """
    if not labels_file.exists():
        raise FileNotFoundError(f"labels.txt not found: {labels_file}")

    X_list = []
    y_list = []
    paths = []
    previews = {}

    with open(labels_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rel, label_str = line.split()
            label = int(label_str)
            y = 1 if label == 1 else -1

            img_path = dataset_root / rel
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found (line {line_no}): {img_path}")

            gray = imread_unicode(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError(f"cv2.imread failed (line {line_no}): {img_path}")

            feat, vis_gray, vis_bin = preprocess_image_to_feature(gray, resize_wh=resize_wh, thr=thr)

            X_list.append(feat)
            y_list.append(y)
            paths.append(img_path)

            if y not in previews:
                previews[y] = (img_path, vis_gray, vis_bin)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)

    return X, y, paths, previews


def train_test_split(X, y, test_ratio=0.25, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_ratio))
    n_test = max(1, min(n - 1, n_test))

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx], train_idx, test_idx


def fit_hard_margin_svm_primal(X: np.ndarray, y: np.ndarray, solver="SCS"):
    """
    min ||K||_2  s.t.  y_i (x_i^T K + b) >= 1
    """
    n_samples, n_features = X.shape
    K = cp.Variable(n_features)
    b = cp.Variable()

    constraints = [y[i] * (X[i] @ K + b) >= 1 for i in range(n_samples)]
    objective = cp.Minimize(cp.norm(K, 2))
    prob = cp.Problem(objective, constraints)

    prob.solve(solver=solver, verbose=False)

    return prob, K, b


def fit_soft_margin_svm_primal(X: np.ndarray, y: np.ndarray, C=10.0, solver="SCS"):
    """
    Soft-margin (выпукло):
      min 0.5||K||^2 + C * sum(xi_i)
      s.t. y_i(x_i^T K + b) >= 1 - xi_i, xi_i >= 0
    """
    n_samples, n_features = X.shape
    K = cp.Variable(n_features)
    b = cp.Variable()
    xi = cp.Variable(n_samples)

    constraints = [
        y * (X @ K + b) >= 1 - xi,
        xi >= 0
    ]
    objective = cp.Minimize(0.5 * cp.sum_squares(K) + C * cp.sum(xi))
    prob = cp.Problem(objective, constraints)

    prob.solve(solver=solver, verbose=False)

    return prob, K, b


def predict_scores(X, K, b):
    return X @ K + b


def predict_labels(X, K, b):
    s = predict_scores(X, K, b)
    return np.where(s >= 0, 1, -1).astype(np.int32)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Метрики для бинарной классификации, положительный класс = +1.
    """
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))

    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def print_metrics(title: str, m: dict):
    print(f"\n=== {title} ===")
    print(f"Confusion: TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print(f"Accuracy : {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall   : {m['recall']:.4f}")
    print(f"F1       : {m['f1']:.4f}")


def main():
    script_dir = Path(__file__).resolve().parent
    dataset_root = script_dir / DATASET_DIRNAME
    labels_file = dataset_root / LABELS_FILENAME
    results_dir = script_dir / RESULTS_DIRNAME
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Script dir :", script_dir)
    print("Dataset dir:", dataset_root)
    print("Labels file:", labels_file)

    # 1) Загрузка + сжатие + вывод 2 примеров разных классов
    X, y, paths, previews = load_dataset_from_labels(
        dataset_root=dataset_root,
        labels_file=labels_file,
        resize_wh=RESIZE_WH,
        thr=BINARY_THRESHOLD
    )

    print(f"\nLoaded: X shape={X.shape}, y shape={y.shape}")
    print(f"Resize: {RESIZE_WH}, threshold={BINARY_THRESHOLD}")

    # сохраним два примера (если оба класса есть)
    for cls in (+1, -1):
        if cls in previews:
            src_path, vis_gray, vis_bin = previews[cls]

            # сохраним уменьшенную серую
            out1 = results_dir / f"preview_gray_class_{cls:+d}.png"
            imwrite_unicode(out1, cv2.cvtColor(vis_gray, cv2.COLOR_GRAY2BGR))

            # сохраним бинарную
            out2 = results_dir / f"preview_bin_class_{cls:+d}.png"
            imwrite_unicode(out2, cv2.cvtColor(vis_bin, cv2.COLOR_GRAY2BGR))

            print(f"Saved previews for class {cls:+d}:")
            print(" ", out1)
            print(" ", out2)
        else:
            print(f"WARNING: no sample for class {cls:+d}")

    # 2) Разделение выборки
    Xtr, ytr, Xte, yte, train_idx, test_idx = train_test_split(
        X, y, test_ratio=TEST_RATIO, seed=RANDOM_SEED
    )
    print(f"\nSplit: train={Xtr.shape[0]}  test={Xte.shape[0]}  (test_ratio={TEST_RATIO})")

    # 3) Решение оптимизации
    K_val = None
    b_val = None

    if TRY_HARD_MARGIN:
        prob, K, b = fit_hard_margin_svm_primal(Xtr, ytr, solver=SOLVER)
        print(f"\nHard-margin status: {prob.status}")

        if prob.status in ("optimal", "optimal_inaccurate") and K.value is not None and b.value is not None:
            K_val = np.array(K.value).astype(np.float32)
            b_val = float(b.value)
        else:
            print("Hard-margin infeasible/failed -> fallback to soft-margin.")

    if K_val is None:
        prob, K, b = fit_soft_margin_svm_primal(Xtr, ytr, C=SOFT_MARGIN_C, solver=SOLVER)
        print(f"\nSoft-margin status: {prob.status}  (C={SOFT_MARGIN_C})")

        if prob.status not in ("optimal", "optimal_inaccurate") or K.value is None or b.value is None:
            raise RuntimeError(f"Optimization failed: status={prob.status}")

        K_val = np.array(K.value).astype(np.float32)
        b_val = float(b.value)

    # 4) Метрики train/test
    ytr_pred = predict_labels(Xtr, K_val, b_val)
    yte_pred = predict_labels(Xte, K_val, b_val)

    m_tr = classification_metrics(ytr, ytr_pred)
    m_te = classification_metrics(yte, yte_pred)

    print_metrics("TRAIN", m_tr)
    print_metrics("TEST", m_te)

    # Сохраним параметры модели и немного отладочной инфы
    model_txt = results_dir / "model.txt"
    with open(model_txt, "w", encoding="utf-8") as f:
        f.write(f"resize_wh={RESIZE_WH}\n")
        f.write(f"binary_threshold={BINARY_THRESHOLD}\n")
        f.write(f"test_ratio={TEST_RATIO}\n")
        f.write(f"seed={RANDOM_SEED}\n")
        f.write(f"solver={SOLVER}\n")
        f.write(f"hard_margin_tried={TRY_HARD_MARGIN}\n")
        f.write(f"soft_margin_C={SOFT_MARGIN_C}\n")
        f.write(f"n_features={X.shape[1]}\n")
        f.write(f"b={b_val}\n")
        f.write("K (first 50):\n")
        f.write(" ".join([f"{v:.6g}" for v in K_val[:50]]) + "\n")
        f.write("\nTRAIN metrics:\n")
        for k, v in m_tr.items():
            f.write(f"{k}={v}\n")
        f.write("\nTEST metrics:\n")
        for k, v in m_te.items():
            f.write(f"{k}={v}\n")

    print(f"\nSaved: {model_txt}")
    print(f"Saved previews in: {results_dir}")


if __name__ == "__main__":
    main()