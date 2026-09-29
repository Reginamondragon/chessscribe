"""Módulo 4 — Validación.

Reglas (python-chess + propias):

- Una pieza por casilla (garantizado por el propio FEN).
- Exactamente un rey por bando.
- Como máximo 16 piezas y 8 peones por bando.
- Ningún peón en las filas 1 u 8.
- Al indicar el turno, el bando que NO mueve no puede estar en jaque.

Las incidencias y las casillas de confianza baja se marcan para revisión. Si hay
cualquier incidencia bloqueante o casillas por revisar, **no se genera ninguna
salida** (steering: product #3). Cada corrección se revalida antes de aplicarse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import chess

from chesscribe.canonical import CanonicalPosition, Turn


class Severity(str, Enum):
    ERROR = "error"      # bloquea la generación de salidas
    REVIEW = "review"    # casilla dudosa: requiere confirmación del usuario


@dataclass
class Issue:
    """Incidencia de validación con mensaje accesible."""

    code: str
    severity: Severity
    message: str
    squares: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "squares": self.squares,
        }


@dataclass
class ValidationResult:
    ok: bool
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def reviews(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.REVIEW]

    def to_dict(self) -> dict:
        return {"ok": self.ok, "issues": [i.to_dict() for i in self.issues]}


_PIECE_NAMES = {
    "K": "rey blanco", "Q": "dama blanca", "R": "torre blanca",
    "B": "alfil blanco", "N": "caballo blanco", "P": "peón blanco",
    "k": "rey negro", "q": "dama negra", "r": "torre negra",
    "b": "alfil negro", "n": "caballo negro", "p": "peón negro",
}


def validate(position: CanonicalPosition) -> ValidationResult:
    """Valida el modelo canónico y devuelve todas las incidencias encontradas."""
    issues: list[Issue] = []

    # El FEN de colocación debe ser sintácticamente válido.
    try:
        board = chess.Board(position.placement + " w - - 0 1")
    except (ValueError, IndexError):
        issues.append(
            Issue(
                code="fen_invalido",
                severity=Severity.ERROR,
                message="La colocación de piezas no es un FEN válido.",
            )
        )
        return ValidationResult(ok=False, issues=issues)

    piece_map = board.piece_map()

    # --- Conteos por bando --------------------------------------------- #
    white = [p for p in piece_map.values() if p.color == chess.WHITE]
    black = [p for p in piece_map.values() if p.color == chess.BLACK]
    w_kings = sum(1 for p in white if p.piece_type == chess.KING)
    b_kings = sum(1 for p in black if p.piece_type == chess.KING)
    w_pawns = sum(1 for p in white if p.piece_type == chess.PAWN)
    b_pawns = sum(1 for p in black if p.piece_type == chess.PAWN)

    # Exactamente un rey por bando.
    if w_kings != 1:
        issues.append(Issue(
            "reyes_blancas", Severity.ERROR,
            f"Las blancas deben tener exactamente un rey; se han contado {w_kings}.",
        ))
    if b_kings != 1:
        issues.append(Issue(
            "reyes_negras", Severity.ERROR,
            f"Las negras deben tener exactamente un rey; se han contado {b_kings}.",
        ))

    # Máximo 16 piezas por bando.
    if len(white) > 16:
        issues.append(Issue(
            "max_piezas_blancas", Severity.ERROR,
            f"Las blancas no pueden tener más de 16 piezas; hay {len(white)}.",
        ))
    if len(black) > 16:
        issues.append(Issue(
            "max_piezas_negras", Severity.ERROR,
            f"Las negras no pueden tener más de 16 piezas; hay {len(black)}.",
        ))

    # Máximo 8 peones por bando.
    if w_pawns > 8:
        issues.append(Issue(
            "max_peones_blancas", Severity.ERROR,
            f"Las blancas no pueden tener más de 8 peones; hay {w_pawns}.",
        ))
    if b_pawns > 8:
        issues.append(Issue(
            "max_peones_negras", Severity.ERROR,
            f"Las negras no pueden tener más de 8 peones; hay {b_pawns}.",
        ))

    # Ningún peón en las filas 1 u 8.
    peones_ilegales: list[str] = []
    for sq, piece in piece_map.items():
        if piece.piece_type == chess.PAWN:
            rank = chess.square_rank(sq)  # 0 = fila 1, 7 = fila 8
            if rank in (0, 7):
                peones_ilegales.append(chess.square_name(sq))
    if peones_ilegales:
        issues.append(Issue(
            "peon_fila_borde", Severity.ERROR,
            "No puede haber peones en la primera ni en la octava fila: "
            + ", ".join(s.upper() for s in sorted(peones_ilegales)) + ".",
            squares=sorted(peones_ilegales),
        ))

    # --- Turno: el bando que NO mueve no puede estar en jaque ---------- #
    if position.turn in (Turn.WHITE, Turn.BLACK) and w_kings == 1 and b_kings == 1:
        side_to_move = chess.WHITE if position.turn is Turn.WHITE else chess.BLACK
        # Construimos un tablero con el turno del bando que mueve y comprobamos
        # si el rival (que no mueve) está en jaque, lo cual sería ilegal.
        probe = chess.Board(
            f"{position.placement} "
            f"{'w' if side_to_move == chess.WHITE else 'b'} - - 0 1"
        )
        # Cambiar el turno "virtualmente" para ver si el que no mueve da jaque.
        opponent = not side_to_move
        # ¿Está el rey del bando que NO mueve atacado por el que mueve?
        king_sq = probe.king(opponent)
        if king_sq is not None and probe.is_attacked_by(side_to_move, king_sq):
            mueve = "blancas" if side_to_move == chess.WHITE else "negras"
            no_mueve = "negras" if side_to_move == chess.WHITE else "blancas"
            issues.append(Issue(
                "jaque_al_que_no_mueve", Severity.ERROR,
                f"Mueven las {mueve}, pero las {no_mueve} están en jaque; "
                "la posición no es legal con ese turno.",
            ))

    # --- Casillas de confianza baja -> revisión ------------------------ #
    if position.low_confidence:
        squares = sorted(s.lower() for s in position.low_confidence)
        issues.append(Issue(
            "confianza_baja", Severity.REVIEW,
            "Casillas por revisar antes de generar la salida: "
            + ", ".join(s.upper() for s in squares) + ".",
            squares=squares,
        ))

    blocking = any(i.severity is Severity.ERROR for i in issues)
    has_review = any(i.severity is Severity.REVIEW for i in issues)
    return ValidationResult(ok=not (blocking or has_review), issues=issues)


def can_emit_outputs(position: CanonicalPosition) -> ValidationResult:
    """Puerta única antes de cualquier salida (product #3 y #6)."""
    return validate(position)
