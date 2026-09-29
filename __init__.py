"""ChessScribe: transcripción braille B8 (ONCE) y análisis accesible de
diagramas de ajedrez.

La librería es independiente de la interfaz web. Módulos:

- ``preprocessing``  (1) OpenCV: escaneo, perspectiva, contraste, casillas.
- ``recognition``    (2) VLM + visión clásica.
- ``canonical``      (3) Modelo canónico: FEN + JSON de anotaciones.
- ``validation``     (4) python-chess + reglas propias.
- ``outputs``        (5) Salidas puras: braille B8, texto, consultas.
- ``analysis``       (6) python-chess + Stockfish.
"""

from chesscribe.canonical import CanonicalPosition, Highlight, Arrow

__all__ = ["CanonicalPosition", "Highlight", "Arrow"]
__version__ = "0.1.0"
