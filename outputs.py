"""Módulo 5 — Salidas.

Funciones **puras** que generan, siempre desde el modelo canónico:

- ``transcribe_braille`` — transcripción braille según el Documento técnico B8.
- ``describe_text``      — descripción en texto («Blancas: Rey en H3…»).
- ``query`` (por pieza, color o casilla).

Reglas de orden (steering: notacion-once #10):
- En **braille**: R, D, T, C, A, P.
- En **texto**:   R, D, T, A, C, P.
- Dentro de cada tipo: por columna (a-h) y después por fila (1-8).

Resaltados y flechas son opcionales: si no existen, no se emite ninguna línea
sobre ellos (#11). Los resaltados solo provienen de imágenes en color (#12), lo
cual ya se decide en reconocimiento; aquí solo se transcriben si están.

Ninguna función accede a la imagen (product #2). La validación es responsabilidad
del backend antes de llamar aquí, pero estas funciones pueden invocarse tras
comprobar ``can_emit_outputs``.
"""

from __future__ import annotations

import chess

from chesscribe import braille_b8 as b8
from chesscribe.canonical import CanonicalPosition, Turn


# Orden de tipos de pieza.
_ORDER_BRAILLE = ["K", "Q", "R", "N", "B", "P"]  # R D T C A P
_ORDER_TEXT = ["K", "Q", "R", "B", "N", "P"]     # R D T A C P

# Nombre de pieza en texto (singular/plural), por símbolo FEN en mayúscula.
_PIECE_TEXT = {
    "K": ("Rey", "Reyes"),
    "Q": ("Dama", "Damas"),
    "R": ("Torre", "Torres"),
    "B": ("Alfil", "Alfiles"),
    "N": ("Caballo", "Caballos"),
    "P": ("Peón", "Peones"),
}

# Inicial de pieza (español) por símbolo FEN en mayúscula.
_PIECE_INITIAL = {"K": "R", "Q": "D", "R": "T", "B": "A", "N": "C", "P": "P"}


def _sorted_squares(squares: list[str]) -> list[str]:
    """Ordena casillas por columna (a-h) y luego por fila (1-8)."""
    return sorted(squares, key=lambda s: (s[0], s[1]))


def _pieces_by_color(position: CanonicalPosition):
    """Devuelve (blancas, negras): dict tipo_mayúscula → lista de casillas."""
    board = chess.Board(position.placement + " w - - 0 1")
    white: dict[str, list[str]] = {}
    black: dict[str, list[str]] = {}
    for sq, piece in board.piece_map().items():
        name = chess.square_name(sq)
        target = white if piece.color == chess.WHITE else black
        target.setdefault(piece.symbol().upper(), []).append(name)
    return white, black


# --------------------------------------------------------------------------- #
# Descripción en texto
# --------------------------------------------------------------------------- #

def _describe_side_text(label: str, pieces: dict[str, list[str]]) -> str:
    parts: list[str] = []
    for kind in _ORDER_TEXT:
        squares = _sorted_squares(pieces.get(kind, []))
        if not squares:
            continue
        singular, plural = _PIECE_TEXT[kind]
        name = singular if len(squares) == 1 else plural
        ubic = ", ".join(s.upper() for s in squares)
        parts.append(f"{name} en {ubic}")
    if not parts:
        return f"{label}: sin piezas."
    return f"{label}: " + "; ".join(parts) + "."


def describe_text(position: CanonicalPosition) -> str:
    """Descripción en texto de toda la posición.

    Incluye líneas de resaltados y flechas solo si existen (#11) y el turno solo
    si el usuario lo indicó.
    """
    white, black = _pieces_by_color(position)
    lines = [
        _describe_side_text("Blancas", white),
        _describe_side_text("Negras", black),
    ]

    if position.highlights:
        hl = "; ".join(
            f"{h.square.upper()} ({h.color})" for h in position.highlights
        )
        lines.append(f"Casillas resaltadas: {hl}.")

    if position.arrows:
        ar = "; ".join(
            f"{a.src.upper()}→{a.dst.upper()}" for a in position.arrows
        )
        lines.append(f"Flechas: {ar}.")

    if position.turn is Turn.WHITE:
        lines.append("Mueven las blancas.")
    elif position.turn is Turn.BLACK:
        lines.append("Mueven las negras.")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Transcripción braille (Documento técnico B8)
# --------------------------------------------------------------------------- #

def _braille_side(pieces: dict[str, list[str]]) -> str:
    """Transcribe un bando en orden R, D, T, C, A, P.

    Cada pieza se escribe como inicial en mayúscula braille seguida de la casilla
    (columna en mayúscula + fila con signo de número). Las piezas se separan con
    espacio.
    """
    tokens: list[str] = []
    for kind in _ORDER_BRAILLE:
        for sq in _sorted_squares(pieces.get(kind, [])):
            initial = _PIECE_INITIAL[kind]
            # Inicial de pieza en mayúscula braille + casilla.
            token = b8.CAPITAL + b8.PIECE_LETTER[initial] + b8.square(sq)
            tokens.append(token)
    return b8.SPACE.join(tokens)


def transcribe_braille(position: CanonicalPosition) -> str:
    """Transcripción braille B8 de la posición.

    Devuelve texto Unicode braille exportable como UTF-8. Blancas y negras en
    líneas separadas; resaltados y flechas solo si existen.
    """
    white, black = _pieces_by_color(position)
    lines = [
        _braille_side(white),
        _braille_side(black),
    ]

    if position.highlights:
        hl = b8.SPACE.join(b8.square(h.square) for h in position.highlights)
        lines.append(hl)

    if position.arrows:
        ar = b8.SPACE.join(
            b8.square(a.src) + b8.square(a.dst) for a in position.arrows
        )
        lines.append(ar)

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Consultas por pieza / color / casilla
# --------------------------------------------------------------------------- #

_COLOR_WORDS = {
    "blanca": chess.WHITE, "blancas": chess.WHITE, "blanco": chess.WHITE,
    "blancos": chess.WHITE, "white": chess.WHITE,
    "negra": chess.BLACK, "negras": chess.BLACK, "negro": chess.BLACK,
    "negros": chess.BLACK, "black": chess.BLACK,
}

_KIND_WORDS = {
    "rey": "K", "reyes": "K",
    "dama": "Q", "damas": "Q", "reina": "Q", "reinas": "Q",
    "torre": "R", "torres": "R",
    "alfil": "B", "alfiles": "B",
    "caballo": "N", "caballos": "N",
    "peon": "P", "peones": "P", "peón": "P",
}


def query_square(position: CanonicalPosition, square_name: str) -> str:
    """«¿Qué hay en E4?» → «Peón negro» o «Casilla vacía»."""
    board = chess.Board(position.placement + " w - - 0 1")
    sq = chess.parse_square(square_name.lower())
    piece = board.piece_at(sq)
    if piece is None:
        return "Casilla vacía."
    name = _PIECE_TEXT[piece.symbol().upper()][0]
    color = "blanco" if piece.color == chess.WHITE else "negro"
    # Concordancia simple: Dama/Torre son femeninas.
    if piece.symbol().upper() in ("Q", "R"):
        color = "blanca" if piece.color == chess.WHITE else "negra"
    return f"{name} {color}."


def query_pieces(
    position: CanonicalPosition, kind_word: str, color_word: str | None = None
) -> str:
    """«¿Dónde están los peones negros?» → «Peones negros en A7, C5…»."""
    kind = _KIND_WORDS.get(kind_word.lower().strip())
    if kind is None:
        return f"No reconozco la pieza «{kind_word}»."

    color = None
    if color_word is not None:
        color = _COLOR_WORDS.get(color_word.lower().strip())
        if color is None:
            return f"No reconozco el color «{color_word}»."

    board = chess.Board(position.placement + " w - - 0 1")
    squares: list[str] = []
    for sq, piece in board.piece_map().items():
        if piece.symbol().upper() != kind:
            continue
        if color is not None and piece.color != color:
            continue
        squares.append(chess.square_name(sq))

    squares = _sorted_squares(squares)
    singular, plural = _PIECE_TEXT[kind]

    if color is None:
        etiqueta = plural if len(squares) != 1 else singular
    else:
        col_adj = _color_adjective(kind, color, plural=len(squares) != 1)
        etiqueta = f"{plural if len(squares) != 1 else singular} {col_adj}"

    if not squares:
        return f"No hay {etiqueta.lower()}."
    ubic = ", ".join(s.upper() for s in squares)
    return f"{etiqueta} en {ubic}."


def _color_adjective(kind: str, color: bool, plural: bool) -> str:
    fem = kind in ("Q", "R")  # Dama, Torre
    if color == chess.WHITE:
        base = "blanca" if fem else "blanco"
    else:
        base = "negra" if fem else "negro"
    return base + ("s" if plural else "")
