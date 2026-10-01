"""Mat de vision — détection de tags ArUco au-dessus d'une table de jeu.

Ce package regroupe les briques du projet :

* :mod:`matvision.camera`      — accès à la webcam USB (LattePanda Delta).
* :mod:`matvision.calibration` — calibration automatique des intrinsèques.
* :mod:`matvision.aruco`       — détection des tags ArUco.
* :mod:`matvision.table`       — repère de la table à partir des 4 tags de coin.
* :mod:`matvision.overlay`     — annotations de l'image d'aperçu.
* :mod:`matvision.vision`      — moteur (boucle) qui relève les positions.
* :mod:`matvision.api`         — API REST Flask pour le réseau local.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
