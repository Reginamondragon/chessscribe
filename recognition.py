"""Módulo 2 — Reconocimiento (VLM + visión clásica).

Un modelo de visión multimodal (VLM) recibe el tablero con las coordenadas
superpuestas y devuelve un JSON con la pieza de cada casilla, las flechas, la
orientación y el número del diagrama.

En paralelo, la visión clásica determina qué casillas están ocupadas (análisis
de trazo) y, en imágenes en color, qué casillas están resaltadas y en qué color
(tono y saturación).

**Fusión de confianza:** una casilla en la que ambos métodos no coinciden (el
VLM dice que hay pieza y el trazo dice que está vacía, o viceversa) recibe
confianza baja y se marca para revisión (product #4).

Las claves de API se leen de variables de entorno; nunca del código (tech #16).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from chesscribe.canonical import (
    Arrow,
    CanonicalPosition,
    Highlight,
    Turn,
)
from chesscribe.preprocessing import BoardExtraction


# Orden de casillas producido por preprocessing._split_into_cells:
# fila 8 (arriba) a fila 1 (abajo), columnas a..h.
def _cell_index_to_square(index: int) -> str:
    row = index // 8          # 0 = fila 8
    col = index % 8           # 0 = columna a
    file_ = "abcdefgh"[col]
    rank = 8 - row
    return f"{file_}{rank}"


# --------------------------------------------------------------------------- #
# Interfaz del VLM
# --------------------------------------------------------------------------- #

class VLMClient(Protocol):
    """Contrato mínimo de un cliente VLM.

    ``recognize`` recibe la imagen del tablero (bytes PNG/JPEG) y devuelve un
    dict JSON con, al menos: ``pieces`` (mapa casilla→símbolo FEN o vacío),
    ``arrows`` (lista de {src,dst}), ``orientation`` y ``diagram_number``.
    """

    def recognize(self, board_png: bytes) -> dict: ...


class EnvVLMClient:
    """Cliente VLM real, configurado por variables de entorno.

    Lee ``VLM_API_KEY`` y ``VLM_MODEL``. La implementación concreta del proveedor
    se deja como punto de extensión; aquí solo se valida la presencia de la
    clave para no exponerla nunca en el código.
    """

    def __init__(self) -> None:
        self.api_key = os.environ.get("VLM_API_KEY")
        self.model = os.environ.get("VLM_MODEL", "default-vlm")

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def recognize(self, board_png: bytes) -> dict:  # pragma: no cover - red
        if not self.configured:
            raise RuntimeError(
                "VLM no configurado: define la variable de entorno VLM_API_KEY."
            )
        # Punto de integración con el proveedor real (HTTP). Se omite aquí para
        # no acoplar la librería a un SDK concreto ni exponer credenciales.
        raise NotImplementedError(
            "Integra aquí el proveedor VLM usando self.api_key y self.model."
        )


# --------------------------------------------------------------------------- #
# Visión clásica
# --------------------------------------------------------------------------- #

def _occupancy_by_stroke(cell: np.ndarray) -> bool:
    """¿La casilla está ocupada? Heurística por densidad de trazo.

    Una pieza introduce bordes/contraste dentro de la casilla. Medimos la
    desviación estándar tras igualar el fondo: una casilla vacía es casi uniforme.
    """
    if cell is None or cell.size == 0:
        return False
    gray = cell if cell.ndim == 2 else np.mean(cell[:, :, :3], axis=2)
    h, w = gray.shape[:2]
    # Recortamos un margen para ignorar el borde entre casillas.
    my, mx = max(1, h // 8), max(1, w // 8)
    inner = gray[my:h - my, mx:w - mx]
    if inner.size == 0:
        inner = gray
    return float(np.std(inner)) > 18.0


def _highlight_of(cell: np.ndarray) -> Optional[str]:
    """Color de resaltado de una casilla en color, o None.

    Solo se llama en imágenes en color (notacion-once #12). Detecta tonos con
    saturación alta que no correspondan a las casillas neutras del tablero.
    """
    try:
        import cv2
    except ImportError:
        return None
    if cell is None or cell.size == 0 or cell.ndim != 3:
        return None
    hsv = cv2.cvtColor(cell[:, :, :3], cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if float(np.mean(s)) < 60:  # poco saturado -> casilla neutra, no resaltado
        return None
    mean_h = float(np.mean(h))  # OpenCV: H en [0,180]
    if mean_h < 15 or mean_h >= 165:
        return "rojo"
    if 35 <= mean_h < 85:
        return "verde"
    if 85 <= mean_h < 135:
        return "azul"
    if 20 <= mean_h < 35:
        return "amarillo"
    return "resaltado"


# --------------------------------------------------------------------------- #
# Fusión
# --------------------------------------------------------------------------- #

@dataclass
class RecognitionResult:
    position: CanonicalPosition
    low_confidence: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _placement_from_pieces(pieces: dict) -> str:
    """Construye el campo de colocación FEN a partir de casilla→símbolo."""
    import chess

    board = chess.Board.empty()
    for square_name, symbol in pieces.items():
        if not symbol:
            continue
        sq = chess.parse_square(square_name.lower())
        board.set_piece_at(sq, chess.Piece.from_symbol(symbol))
    return board.board_fen()


def recognize(
    extraction: BoardExtraction,
    board_png: bytes,
    vlm: Optional[VLMClient] = None,
) -> RecognitionResult:
    """Combina VLM y visión clásica en un modelo canónico con confianzas.

    Args:
        extraction: salida del preprocesado (64 casillas, color/BN).
        board_png: imagen del tablero (con coordenadas) para el VLM.
        vlm: cliente VLM; si es None se usa EnvVLMClient.
    """
    if not extraction.ok:
        raise ValueError(extraction.message)

    vlm = vlm or EnvVLMClient()
    vlm_data = vlm.recognize(board_png)

    pieces: dict = {k.lower(): v for k, v in vlm_data.get("pieces", {}).items()}
    arrows_raw = vlm_data.get("arrows", [])
    orientation = vlm_data.get("orientation", "white")
    diagram_number = vlm_data.get("diagram_number")

    low_conf: list = []
    notes: list = []

    # Cotejar ocupación VLM vs trazo clásico.
    if extraction.cells and len(extraction.cells) == 64:
        for idx, cell in enumerate(extraction.cells):
            square = _cell_index_to_square(idx)
            vlm_occupied = bool(pieces.get(square))
            stroke_occupied = _occupancy_by_stroke(cell)
            if vlm_occupied != stroke_occupied:
                low_conf.append(square)
                notes.append(
                    f"Casilla {square.upper()}: el reconocimiento y el análisis "
                    "de trazo no coinciden."
                )

    # Resaltados solo en color.
    highlights: list = []
    if extraction.is_color and extraction.cells and len(extraction.cells) == 64:
        for idx, cell in enumerate(extraction.cells):
            color = _highlight_of(cell)
            if color:
                highlights.append(Highlight(_cell_index_to_square(idx), color))

    arrows = [
        Arrow(a["src"], a["dst"])
        for a in arrows_raw
        if "src" in a and "dst" in a
    ]

    position = CanonicalPosition(
        placement=_placement_from_pieces(pieces),
        turn=Turn.UNKNOWN,  # el usuario lo indica después
        diagram_number=diagram_number,
        orientation=orientation,
        highlights=highlights,
        arrows=arrows,
        low_confidence=sorted(set(low_conf)),
    )
    return RecognitionResult(position=position, low_confidence=low_conf, notes=notes)


# --------------------------------------------------------------------------- #
# Cliente simulado para pruebas sin red ni imágenes
# --------------------------------------------------------------------------- #

class StubVLMClient:
    """VLM simulado que devuelve un JSON fijo (para tests y demo)."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def recognize(self, board_png: bytes) -> dict:
        return json.loads(json.dumps(self._payload))  # copia defensiva


# --------------------------------------------------------------------------- #
# Reconocedor por visión clásica (sin VLM ni clave)
# --------------------------------------------------------------------------- #

def _piece_color_of(cell) -> Optional[bool]:
    """Devuelve True (blanca) / False (negra) / None si la casilla está vacía.

    Heurística por brillo del objeto sobre el fondo de la casilla: se aísla la
    zona central (donde está la pieza) y se compara su brillo medio con el del
    borde (fondo de la casilla). Un objeto más claro que su entorno se asume
    pieza blanca; más oscuro, pieza negra.
    """
    if cell is None or getattr(cell, "size", 0) == 0:
        return None
    gray = cell if cell.ndim == 2 else np.mean(cell[:, :, :3], axis=2)
    h, w = gray.shape[:2]
    if h < 6 or w < 6:
        return None
    cy0, cy1 = int(h * 0.30), int(h * 0.70)
    cx0, cx1 = int(w * 0.30), int(w * 0.70)
    center = gray[cy0:cy1, cx0:cx1]
    if center.size == 0:
        return None
    # Fondo: media de las cuatro esquinas.
    m = max(2, h // 6)
    corners = np.concatenate([
        gray[:m, :m].ravel(), gray[:m, -m:].ravel(),
        gray[-m:, :m].ravel(), gray[-m:, -m:].ravel(),
    ])
    bg = float(np.mean(corners))
    obj = float(np.mean(center))
    # El signo de la diferencia indica si la pieza es más clara u oscura.
    if obj >= bg:
        return True   # más clara que el fondo -> pieza blanca
    return False      # más oscura -> pieza negra


class ClassicalVLMClient:
    """Reconocedor compatible con ``VLMClient`` que NO usa servicios externos.

    Determina, por visión clásica, qué casillas están ocupadas y de qué color
    es cada pieza. **No identifica el tipo** de pieza (rey, dama…): para eso hace
    falta un VLM. Por eso marca cada casilla ocupada con un símbolo genérico y la
    deja como dudosa para que el usuario confirme el tipo, sin inventar nada.

    Como marcador genérico usa 'P'/'p' (peón) solo como marca de posición y color;
    el flujo de revisión pedirá el tipo real.
    """

    def __init__(self, extraction: BoardExtraction) -> None:
        self._extraction = extraction

    def recognize(self, board_png: bytes) -> dict:
        cells = self._extraction.cells
        pieces: dict = {}
        if cells and len(cells) == 64:
            for idx, cell in enumerate(cells):
                if not _occupancy_by_stroke(cell):
                    continue
                color = _piece_color_of(cell)
                square = _cell_index_to_square(idx)
                # Marcador genérico: mayúscula=blanca, minúscula=negra.
                pieces[square] = "P" if color else "p"
        return {
            "pieces": pieces,
            "arrows": [],
            "orientation": self._extraction.orientation_hint,
            "diagram_number": None,
            "_generic": True,  # señal de que el tipo no está identificado
        }


def recognize_auto(
    extraction: BoardExtraction, board_png: bytes
) -> "RecognitionResult":
    """Reconoce usando el VLM si está configurado; si no, visión clásica.

    En el modo clásico, todas las casillas ocupadas se marcan de confianza baja
    porque el tipo de pieza no está identificado (product #4: no inventar).
    """
    env_vlm = EnvVLMClient()
    if env_vlm.configured:
        return recognize(extraction, board_png, env_vlm)

    classical = ClassicalVLMClient(extraction)
    result = recognize(extraction, board_png, classical)
    # Marcar todas las casillas con pieza como dudosas (tipo sin confirmar).
    occupied = sorted(result.position.piece_map().keys())
    existing = set(result.position.low_confidence)
    for sq in occupied:
        existing.add(sq)
    result.position.low_confidence = sorted(existing)
    if occupied:
        result.notes.append(
            "Reconocimiento sin modelo de visión: se han detectado las casillas "
            "ocupadas y el color de cada pieza, pero no el tipo. Revisa y "
            "confirma cada pieza. Para identificación automática del tipo, "
            "configura la variable de entorno VLM_API_KEY."
        )
    return result
