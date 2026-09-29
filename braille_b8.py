"""Tabla de signografía braille B8 (Comisión Braille Española).

Este módulo aísla el mapeo *carácter → celda braille Unicode* para que la
transcripción del módulo de salidas sea una función pura y **calibrable**: si un
ejemplo del enunciado exige un matiz concreto, se ajusta aquí en un único sitio
(steering: notacion-once #9 — «ante cualquier duda prevalecen sus ejemplos»).

Braille de 6 puntos. Numeración de puntos:

    1 4
    2 5
    3 6

Cada celda es un carácter Unicode del bloque Braille Patterns (U+2800..U+283F).
El valor de un punto p contribuye con 2**(p-1) al offset sobre U+2800.
"""

from __future__ import annotations


def cell(*dots: int) -> str:
    """Devuelve la celda braille Unicode con los puntos indicados.

    >>> cell(1)        # 'a'
    '⠁'
    >>> cell(1, 2, 5)  # 'd'... (no; d = 1,4,5)
    """
    offset = 0
    for d in dots:
        if not 1 <= d <= 6:
            raise ValueError(f"Punto fuera de rango: {d}")
        offset |= 1 << (d - 1)
    return chr(0x2800 + offset)


# --- Letras base (grado 1, alfabeto español sin signos añadidos) ----------- #
# Solo definimos las que usa la notación de ajedrez.
LETTERS = {
    "a": cell(1),
    "b": cell(1, 2),
    "c": cell(1, 4),
    "d": cell(1, 4, 5),
    "e": cell(1, 5),
    "f": cell(1, 2, 4),
    "g": cell(1, 2, 4, 5),
    "h": cell(1, 2, 5),
    "i": cell(2, 4),
    "j": cell(2, 4, 5),
    "k": cell(1, 3),
    "l": cell(1, 2, 3),
    "n": cell(1, 3, 4, 5),
    "p": cell(1, 2, 3, 4),
    "r": cell(1, 2, 3, 5),
    "t": cell(2, 3, 4, 5),
}

# Iniciales de pieza en español. En braille se escriben en mayúscula, por lo que
# van precedidas del prefijo de mayúscula.
#   R rey · D dama · T torre · C caballo · A alfil · P peón
PIECE_LETTER = {
    "R": LETTERS["r"],
    "D": LETTERS["d"],
    "T": LETTERS["t"],
    "C": LETTERS["c"],
    "A": LETTERS["a"],
    "P": LETTERS["p"],
}

# --- Signos de composición ------------------------------------------------- #
CAPITAL = cell(4, 6)        # signo de mayúscula (puntos 4-6)
NUMBER = cell(3, 4, 5, 6)   # signo de número (puntos 3-4-5-6)
SPACE = " "

# --- Dígitos ---------------------------------------------------------------- #
# En braille los dígitos 1..9,0 se representan con las letras a..i,j precedidas
# del signo de número.
_DIGIT_LETTER = {
    "1": "a", "2": "b", "3": "c", "4": "d", "5": "e",
    "6": "f", "7": "g", "8": "h", "9": "i", "0": "j",
}


def digit(d: str) -> str:
    """Celda braille de un único dígito (sin el signo de número)."""
    return LETTERS[_DIGIT_LETTER[d]]


def number(text: str) -> str:
    """Traduce una secuencia de dígitos con un único signo de número inicial."""
    return NUMBER + "".join(digit(c) for c in text)


def capital_letter(latin: str) -> str:
    """Letra latina minúscula → celda braille en mayúscula (con prefijo)."""
    return CAPITAL + LETTERS[latin.lower()]


def square(name: str) -> str:
    """Casilla algebraica 'e4' → braille: columna (letra mayúscula) + fila.

    La columna se escribe como letra mayúscula (a-h) y la fila como dígito con
    signo de número, siguiendo la convención del documento B8.
    """
    col, row = name[0].lower(), name[1]
    return capital_letter(col) + number(row)
