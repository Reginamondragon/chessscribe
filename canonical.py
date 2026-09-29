"""Módulo 3 — Modelo canónico.

La posición reconocida se guarda como **FEN de colocación de piezas** más un
JSON de anotaciones (resaltados, flechas, turno, número de diagrama, confianzas
y correcciones). Este modelo es la **única fuente** de todas las salidas: ningún
módulo de salida vuelve a mirar la imagen (steering: product #2).

El backend no guarda estado; este objeto se serializa a JSON, viaja al navegador
y vuelve en cada petición.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import chess


# --------------------------------------------------------------------------- #
# Tipos auxiliares
# --------------------------------------------------------------------------- #

FILES = "abcdefgh"
RANKS = "12345678"


class Turn(str, Enum):
    """Bando que mueve. El usuario lo indica; un diagrama no lo dice."""

    WHITE = "w"
    BLACK = "b"
    UNKNOWN = "unknown"


def is_square(name: str) -> bool:
    """¿``name`` es una casilla algebraica válida como 'e4'? (acepta 'E4')."""
    if not isinstance(name, str) or len(name) != 2:
        return False
    f, r = name[0].lower(), name[1]
    return f in FILES and r in RANKS


def normalize_square(name: str) -> str:
    """Normaliza a minúscula-columna + fila, p. ej. 'E4' -> 'e4'."""
    if not is_square(name):
        raise ValueError(f"Casilla inválida: {name!r}")
    return name[0].lower() + name[1]


@dataclass
class Highlight:
    """Casilla resaltada y su color (solo en imágenes en color)."""

    square: str
    color: str  # nombre libre del color detectado, p. ej. "rojo", "verde"

    def __post_init__(self) -> None:
        self.square = normalize_square(self.square)


@dataclass
class Arrow:
    """Flecha con origen y destino."""

    src: str
    dst: str

    def __post_init__(self) -> None:
        self.src = normalize_square(self.src)
        self.dst = normalize_square(self.dst)


@dataclass
class Correction:
    """Corrección del usuario, revalidada y confirmada antes de aplicarse."""

    square: str
    old: Optional[str]  # símbolo FEN previo o None si estaba vacía
    new: Optional[str]  # símbolo FEN nuevo o None para vaciar
    confirmed: bool = False

    def __post_init__(self) -> None:
        self.square = normalize_square(self.square)


# --------------------------------------------------------------------------- #
# Modelo canónico
# --------------------------------------------------------------------------- #

# FEN de posición inicial de colocación (solo el campo de piezas).
EMPTY_PLACEMENT = "8/8/8/8/8/8/8/8"


@dataclass
class CanonicalPosition:
    """Modelo canónico único.

    Atributos:
        placement: campo de colocación FEN (solo piezas), p. ej.
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'.
        turn: bando que mueve (indicado por el usuario).
        diagram_number: número del diagrama del enunciado, si se conoce.
        orientation: 'white' si las blancas están abajo (por defecto).
        highlights: casillas resaltadas (solo en color).
        arrows: flechas origen→destino.
        low_confidence: casillas marcadas con confianza baja (necesitan revisión).
        corrections: historial de correcciones del usuario.
    """

    placement: str = EMPTY_PLACEMENT
    turn: Turn = Turn.UNKNOWN
    diagram_number: Optional[int] = None
    orientation: str = "white"
    highlights: list[Highlight] = field(default_factory=list)
    arrows: list[Arrow] = field(default_factory=list)
    low_confidence: list[str] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)

    # ---- Construcción -------------------------------------------------- #

    @classmethod
    def from_board(cls, board: chess.Board, **kwargs) -> "CanonicalPosition":
        """Crea el modelo a partir de un ``chess.Board`` (usa su colocación)."""
        placement = board.board_fen()
        return cls(placement=placement, **kwargs)

    @classmethod
    def from_fen(cls, fen: str, **kwargs) -> "CanonicalPosition":
        """Crea el modelo desde un FEN completo o solo el campo de colocación."""
        placement = fen.strip().split(" ")[0]
        # Validar que el placement es sintácticamente correcto.
        chess.Board(placement + " w - - 0 1")
        return cls(placement=placement, **kwargs)

    # ---- Derivados ----------------------------------------------------- #

    def to_board(self) -> chess.Board:
        """Devuelve un ``chess.Board`` con la colocación y el turno.

        Si el turno es desconocido asume blancas para poder construir el objeto,
        pero el análisis exige turno explícito (steering: product #7).
        """
        side = "w" if self.turn in (Turn.WHITE, Turn.UNKNOWN) else "b"
        # Sin enroques ni al paso: un diagrama no los declara.
        return chess.Board(f"{self.placement} {side} - - 0 1")

    def full_fen(self) -> str:
        """FEN completo (con turno resuelto). Útil para python-chess/Stockfish."""
        return self.to_board().fen()

    def piece_map(self) -> dict[str, str]:
        """Mapa casilla → símbolo FEN de pieza, p. ej. {'e4': 'p', 'h3': 'K'}."""
        board = chess.Board(self.placement + " w - - 0 1")
        result: dict[str, str] = {}
        for sq, piece in board.piece_map().items():
            result[chess.square_name(sq)] = piece.symbol()
        return result

    # ---- Correcciones -------------------------------------------------- #

    def apply_correction(self, correction: Correction) -> "CanonicalPosition":
        """Aplica una corrección **confirmada** y devuelve un modelo nuevo.

        No muta el objeto original. Lanza ``ValueError`` si no está confirmada.
        """
        if not correction.confirmed:
            raise ValueError(
                "La corrección debe confirmarse antes de aplicarse (product #5)."
            )
        board = chess.Board(self.placement + " w - - 0 1")
        sq = chess.parse_square(correction.square)
        if correction.new is None:
            board.remove_piece_at(sq)
        else:
            board.set_piece_at(sq, chess.Piece.from_symbol(correction.new))

        new_low = [s for s in self.low_confidence if s != correction.square]
        return CanonicalPosition(
            placement=board.board_fen(),
            turn=self.turn,
            diagram_number=self.diagram_number,
            orientation=self.orientation,
            highlights=list(self.highlights),
            arrows=list(self.arrows),
            low_confidence=new_low,
            corrections=[*self.corrections, correction],
        )

    def needs_review(self) -> bool:
        """¿Quedan casillas de confianza baja sin resolver?"""
        return bool(self.low_confidence)

    # ---- Serialización ------------------------------------------------- #

    def to_dict(self) -> dict:
        d = asdict(self)
        d["turn"] = self.turn.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "CanonicalPosition":
        return cls(
            placement=data.get("placement", EMPTY_PLACEMENT),
            turn=Turn(data.get("turn", "unknown")),
            diagram_number=data.get("diagram_number"),
            orientation=data.get("orientation", "white"),
            highlights=[Highlight(**h) for h in data.get("highlights", [])],
            arrows=[Arrow(**a) for a in data.get("arrows", [])],
            low_confidence=list(data.get("low_confidence", [])),
            corrections=[Correction(**c) for c in data.get("corrections", [])],
        )
