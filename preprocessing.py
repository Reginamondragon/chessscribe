"""Módulo 1 — Preprocesado (OpenCV).

En modo escaneo:

1. Detecta los bordes de la página y corrige la perspectiva.
2. Normaliza iluminación y contraste.
3. Localiza el tablero y lo divide en 64 casillas.
4. Detecta si la imagen es en color o en blanco y negro.

Si el tablero no aparece completo, lo comunica con un mensaje accesible.

OpenCV es una dependencia opcional (extra ``vision``). El módulo la importa de
forma perezosa: si no está instalada, las funciones lanzan un error claro en
lugar de romper el import de la librería.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class VisionUnavailableError(RuntimeError):
    """OpenCV no está instalado (instala el extra 'vision')."""


def _cv2():
    try:
        import cv2  # noqa: F401
        return cv2
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise VisionUnavailableError(
            "El preprocesado requiere OpenCV. Instálalo con "
            "'pip install chesscribe[vision]'."
        ) from exc


@dataclass
class BoardExtraction:
    """Resultado del preprocesado.

    Atributos:
        ok: el tablero se localizó completo.
        message: mensaje accesible (motivo si ``ok`` es False).
        is_color: True si la imagen es en color; False si es en blanco y negro.
        board_image: recorte rectificado del tablero (o None si falla).
        cells: lista de 64 recortes de casilla en orden a8..h1 (fila 8 primero),
            o vacía si falla.
        orientation_hint: 'white' por defecto (las blancas abajo).
    """

    ok: bool
    message: str
    is_color: bool = False
    board_image: Optional[np.ndarray] = None
    cells: list = field(default_factory=list)
    orientation_hint: str = "white"


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como [sup-izq, sup-der, inf-der, inf-izq]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image, pts):
    cv2 = _cv2()
    rect = _order_corners(pts)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = int(max(width_a, width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = int(max(height_a, height_b))
    side = max(max_w, max_h, 8)
    dst = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (side, side))


def _detect_is_color(bgr: np.ndarray) -> bool:
    """Heurística: la imagen es en color si los canales difieren notablemente."""
    cv2 = _cv2()
    if bgr.ndim == 2 or bgr.shape[2] == 1:
        return False
    b, g, r = cv2.split(bgr[:, :, :3])
    # Diferencia media entre canales.
    diff = (
        np.mean(np.abs(b.astype(int) - g.astype(int)))
        + np.mean(np.abs(g.astype(int) - r.astype(int)))
    ) / 2.0
    return diff > 8.0  # umbral prudente


def _normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """Normaliza iluminación/contraste con CLAHE."""
    cv2 = _cv2()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _find_board_quad(gray: np.ndarray):
    """Busca el mayor contorno cuadrilátero (candidato a tablero)."""
    cv2 = _cv2()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = gray.shape[0] * gray.shape[1]
    for cnt in contours[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 0.20 * img_area:
            return approx.reshape(4, 2).astype("float32")
    return None


def _split_into_cells(board_img: np.ndarray) -> list:
    """Divide el tablero rectificado en 64 casillas (fila 8 arriba)."""
    h, w = board_img.shape[:2]
    ch, cw = h // 8, w // 8
    cells = []
    for row in range(8):       # fila 8 (arriba) -> fila 1 (abajo)
        for col in range(8):   # columna a -> h
            y0, y1 = row * ch, (row + 1) * ch
            x0, x1 = col * cw, (col + 1) * cw
            cells.append(board_img[y0:y1, x0:x1].copy())
    return cells


def extract_board(image_bytes: bytes, scan_mode: bool = True) -> BoardExtraction:
    """Preprocesa una imagen y devuelve el tablero dividido en 64 casillas.

    Args:
        image_bytes: contenido del archivo de imagen subido.
        scan_mode: si True aplica corrección de perspectiva de página.

    Returns:
        BoardExtraction con el resultado y un mensaje accesible.
    """
    cv2 = _cv2()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return BoardExtraction(
            ok=False,
            message="No se pudo leer la imagen. Comprueba el formato del archivo.",
        )

    is_color = _detect_is_color(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = _normalize_contrast(gray)

    working = bgr
    if scan_mode:
        quad = _find_board_quad(gray)
        if quad is None:
            return BoardExtraction(
                ok=False,
                is_color=is_color,
                message=(
                    "No se ha localizado el tablero completo en la imagen. "
                    "Encuadra la página entera, con buena luz y sin recortes, "
                    "y vuelve a intentarlo."
                ),
            )
        working = _four_point_transform(bgr, quad)
    else:
        # Sin escaneo: asumimos que la imagen ya es el tablero recortado.
        side = min(bgr.shape[:2])
        working = bgr[:side, :side]

    if working.shape[0] < 8 or working.shape[1] < 8:
        return BoardExtraction(
            ok=False,
            is_color=is_color,
            message="El tablero detectado es demasiado pequeño para analizarlo.",
        )

    cells = _split_into_cells(working)
    return BoardExtraction(
        ok=True,
        message="Tablero localizado y dividido en 64 casillas.",
        is_color=is_color,
        board_image=working,
        cells=cells,
    )
