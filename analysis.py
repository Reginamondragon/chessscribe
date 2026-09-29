"""Módulo 6 — Análisis (python-chess + Stockfish).

- ``legal_moves_of`` — movimientos legales de una pieza, distinguiendo casillas
  libres de capturas.
- ``recommend`` — mejor jugada para el bando que mueve (Stockfish, profundidad
  fija), expresada en notación española y en texto.
- ``verify_move`` — comprueba si la jugada del usuario es legal y cómo de buena
  es, según su pérdida de evaluación respecto a la mejor.

No ejecuta jugadas: la posición solo cambia por correcciones del reconocimiento
(product #1). Todo el análisis exige turno explícito (product #7): un diagrama no
indica quién mueve.

El motor Stockfish es opcional. Si no está disponible, ``recommend`` y la parte
comparativa de ``verify_move`` se degradan con un mensaje accesible en lugar de
fallar.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Optional

import chess
import chess.engine

from chesscribe.canonical import CanonicalPosition, Turn


class TurnRequiredError(ValueError):
    """El análisis requiere que el usuario indique el bando que mueve."""


# Nombres de pieza en español para notación (SAN español).
_SAN_LETTER = {
    chess.KING: "R", chess.QUEEN: "D", chess.ROOK: "T",
    chess.BISHOP: "A", chess.KNIGHT: "C", chess.PAWN: "",
}
_PIECE_TEXT = {
    chess.KING: "Rey", chess.QUEEN: "Dama", chess.ROOK: "Torre",
    chess.BISHOP: "Alfil", chess.KNIGHT: "Caballo", chess.PAWN: "Peón",
}


def _require_turn(position: CanonicalPosition) -> chess.Board:
    if position.turn not in (Turn.WHITE, Turn.BLACK):
        raise TurnRequiredError(
            "Indica qué bando mueve antes de pedir movimientos, "
            "recomendaciones o verificaciones."
        )
    return position.to_board()


# --------------------------------------------------------------------------- #
# Notación española (SAN traducido)
# --------------------------------------------------------------------------- #

def san_es(board: chess.Board, move: chess.Move) -> str:
    """Convierte un movimiento a notación algebraica española.

    Traduce las iniciales de pieza del SAN inglés (K,Q,R,B,N) al español
    (R,D,T,A,C) y conserva capturas (x), jaque (+), mate (#) y enroques.
    """
    san = board.san(move)
    # Enroques se mantienen igual (O-O, O-O-O).
    if san.startswith("O-O"):
        return san
    trans = {"K": "R", "Q": "D", "R": "T", "B": "A", "N": "C"}
    if san and san[0] in trans:
        return trans[san[0]] + san[1:]
    return san  # jugada de peón: sin inicial


# --------------------------------------------------------------------------- #
# Movimientos legales de una pieza
# --------------------------------------------------------------------------- #

@dataclass
class PieceMoves:
    square: str
    piece_text: str
    quiet: list[str]      # destinos sin captura
    captures: list[str]   # destinos con captura

    def to_text(self) -> str:
        if not self.quiet and not self.captures:
            return f"{self.piece_text} de {self.square.upper()} no tiene movimientos legales."
        destinos = sorted(self.quiet + self.captures)
        detalle = ", ".join(d.upper() for d in destinos)
        if not self.captures:
            capt = "ninguna con captura"
        elif not self.quiet:
            capt = "todas con captura"
        else:
            capt = "con captura en " + ", ".join(c.upper() for c in sorted(self.captures))
        return (
            f"{self.piece_text} de {self.square.upper()} puede ir a "
            f"{detalle}, {capt}."
        )

    def to_dict(self) -> dict:
        return {
            "square": self.square,
            "piece_text": self.piece_text,
            "quiet": sorted(self.quiet),
            "captures": sorted(self.captures),
            "text": self.to_text(),
        }


def legal_moves_of(position: CanonicalPosition, square_name: str) -> PieceMoves:
    """Movimientos legales de la pieza situada en ``square_name``.

    Solo se consideran movimientos si esa pieza pertenece al bando que mueve
    (según python-chess, que solo genera movimientos del turno actual).
    """
    board = _require_turn(position)
    sq = chess.parse_square(square_name.lower())
    piece = board.piece_at(sq)
    if piece is None:
        return PieceMoves(square_name.lower(), "Casilla vacía", [], [])

    piece_text = _PIECE_TEXT[piece.piece_type]
    quiet: list[str] = []
    captures: list[str] = []
    for move in board.legal_moves:
        if move.from_square != sq:
            continue
        dst = chess.square_name(move.to_square)
        if board.is_capture(move):
            captures.append(dst)
        else:
            quiet.append(dst)
    return PieceMoves(square_name.lower(), piece_text, quiet, captures)


# --------------------------------------------------------------------------- #
# Motor Stockfish
# --------------------------------------------------------------------------- #

def _stockfish_path() -> Optional[str]:
    return os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")


def _analysis_depth() -> int:
    try:
        return int(os.environ.get("ANALYSIS_DEPTH", "15"))
    except ValueError:
        return 15


@dataclass
class Recommendation:
    available: bool
    move_san_es: Optional[str] = None
    text: str = ""

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "move_san_es": self.move_san_es,
            "text": self.text,
        }


def recommend(position: CanonicalPosition) -> Recommendation:
    """Recomienda la mejor jugada para el bando que mueve."""
    board = _require_turn(position)
    path = _stockfish_path()
    if path is None:
        return Recommendation(
            available=False,
            text="El motor de análisis no está disponible en este servidor.",
        )
    try:
        with chess.engine.SimpleEngine.popen_uci(path) as engine:
            result = engine.play(
                board, chess.engine.Limit(depth=_analysis_depth())
            )
            move = result.move
            if move is None:
                return Recommendation(False, text="No hay jugada disponible.")
            san = san_es(board, move)
            mueve = "blancas" if board.turn == chess.WHITE else "negras"
            return Recommendation(
                available=True,
                move_san_es=san,
                text=f"La mejor jugada para las {mueve} es {san}.",
            )
    except (chess.engine.EngineError, OSError) as exc:
        return Recommendation(
            available=False,
            text=f"No se pudo consultar el motor de análisis: {exc}",
        )


# --------------------------------------------------------------------------- #
# Verificación de la jugada del usuario
# --------------------------------------------------------------------------- #

@dataclass
class MoveVerification:
    legal: bool
    text: str
    move_san_es: Optional[str] = None
    best_san_es: Optional[str] = None
    centipawn_loss: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "legal": self.legal,
            "text": self.text,
            "move_san_es": self.move_san_es,
            "best_san_es": self.best_san_es,
            "centipawn_loss": self.centipawn_loss,
        }


def _parse_user_move(board: chess.Board, text: str) -> Optional[chess.Move]:
    """Interpreta la jugada del usuario en varias notaciones.

    Acepta SAN inglés/español, UCI (e2e4) y coordenadas. Devuelve None si no se
    puede interpretar como jugada legal.
    """
    raw = text.strip()
    # Traducir iniciales españolas a inglesas para SAN.
    trans = str.maketrans({"R": "K", "D": "Q", "T": "R", "A": "B", "C": "N"})
    candidates = [raw, raw.translate(trans)]
    for cand in candidates:
        try:
            return board.parse_san(cand)
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError,
                chess.AmbiguousMoveError):
            pass
    # UCI / coordenadas.
    try:
        move = chess.Move.from_uci(raw.lower())
        if move in board.legal_moves:
            return move
    except (ValueError, chess.InvalidMoveError):
        pass
    return None


def _score_cp(info_score, pov_color: bool) -> Optional[int]:
    """Puntuación en centipeones desde el punto de vista de ``pov_color``."""
    score = info_score.pov(pov_color)
    if score.is_mate():
        # Mate a favor = valor muy alto; en contra = muy bajo.
        return 100000 if score.mate() > 0 else -100000
    return score.score()


def verify_move(position: CanonicalPosition, move_text: str) -> MoveVerification:
    """Indica si la jugada del usuario es legal y cómo de recomendable es."""
    board = _require_turn(position)
    move = _parse_user_move(board, move_text)
    if move is None:
        return MoveVerification(
            legal=False,
            text=f"La jugada «{move_text}» no es legal en esta posición.",
        )

    move_san = san_es(board, move)
    path = _stockfish_path()
    if path is None:
        return MoveVerification(
            legal=True,
            move_san_es=move_san,
            text=f"{move_san} es legal. El motor no está disponible para "
                 "valorar si es la mejor.",
        )

    try:
        depth = _analysis_depth()
        with chess.engine.SimpleEngine.popen_uci(path) as engine:
            pov = board.turn
            # Mejor jugada y su evaluación.
            best_info = engine.analyse(board, chess.engine.Limit(depth=depth))
            best_move = best_info["pv"][0]
            best_cp = _score_cp(best_info["score"], pov)
            best_san = san_es(board, best_move)

            # Evaluación tras la jugada del usuario.
            board.push(move)
            user_info = engine.analyse(board, chess.engine.Limit(depth=depth))
            # Tras empujar, el turno es del rival: puntuación desde 'pov'.
            user_cp = _score_cp(user_info["score"], pov)
            board.pop()

        loss = None
        if best_cp is not None and user_cp is not None:
            loss = max(0, best_cp - user_cp)

        if move == best_move:
            verdict = "Es la mejor jugada disponible."
        elif loss is not None and loss <= 30:
            verdict = f"Es una buena jugada. La mejor sería {best_san}."
        elif loss is not None and loss <= 100:
            verdict = f"Es jugable, pero pierde algo de ventaja. Mejor: {best_san}."
        else:
            verdict = f"No es recomendable. La mejor jugada es {best_san}."

        return MoveVerification(
            legal=True,
            move_san_es=move_san,
            best_san_es=best_san,
            centipawn_loss=loss,
            text=f"{move_san} es legal. {verdict}",
        )
    except (chess.engine.EngineError, OSError) as exc:
        return MoveVerification(
            legal=True,
            move_san_es=move_san,
            text=f"{move_san} es legal. No se pudo valorar con el motor: {exc}",
        )
