"""Suivi des objets marqués posés sur la table.

Le tracker conserve un historique par objet (plusieurs objets peuvent partager le
même identifiant de tag), lisse les positions et oublie les objets disparus.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .config import ObjectConfig
from .geometry import Position, normalize_angle_deg

log = logging.getLogger(__name__)

__all__ = ["ObjectDetection", "ObjectTracker", "TrackedObject"]


@dataclass
class ObjectDetection:
    """Objet détecté sur une image (coordonnées table, mm / degrés)."""

    marker_id: int
    label: str
    position: Position
    size: float = 0.0


@dataclass
class TrackedObject:
    """Objet suivi dans le temps."""

    marker_id: int
    label: str
    position: Position
    size: float = 0.0
    hits: int = 0
    misses: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def age_s(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def to_dict(self) -> dict:
        data = self.position.to_dict()
        data.update(
            {
                "id": self.marker_id,
                "label": self.label,
                "hits": self.hits,
                "missing": self.misses,
                "age_s": round(self.age_s, 3),
                "seen_ago_s": round(max(0.0, time.monotonic() - self.last_seen), 3),
            }
        )
        return data


class ObjectTracker:
    """Association détections <-> objets suivi + lissage exponentiel."""

    def __init__(
        self,
        objects: Sequence[ObjectConfig] | None = None,
        *,
        smoothing: float = 0.4,
        max_age_s: float = 1.0,
        match_distance_mm: float = 200.0,
    ) -> None:
        self.objects = {obj.id: obj for obj in (objects or [])}
        self.smoothing = float(min(max(smoothing, 0.0), 1.0))
        self.max_age_s = float(max_age_s)
        self.match_distance_mm = float(match_distance_mm)
        self._tracks: list[TrackedObject] = []

    # ------------------------------------------------------------------ #
    def configure(self, objects: Sequence[ObjectConfig], *, smoothing: float | None = None,
                  max_age_s: float | None = None, match_distance_mm: float | None = None) -> None:
        """Met à jour la configuration sans perdre les objets déjà suivis."""
        self.objects = {obj.id: obj for obj in objects}
        if smoothing is not None:
            self.smoothing = float(min(max(smoothing, 0.0), 1.0))
        if max_age_s is not None:
            self.max_age_s = float(max_age_s)
        if match_distance_mm is not None:
            self.match_distance_mm = float(match_distance_mm)

    # ------------------------------------------------------------------ #
    def update(self, detections: Iterable[ObjectDetection], now: float | None = None) -> None:
        """Intègre les détections de l'image courante."""
        now = time.monotonic() if now is None else float(now)
        detections = list(detections)
        used = [False] * len(self._tracks)

        for detection in detections:
            index = self._closest_track(detection, used)
            if index is None:
                self._tracks.append(
                    TrackedObject(
                        marker_id=detection.marker_id,
                        label=detection.label,
                        position=detection.position,
                        size=detection.size,
                        hits=1,
                        first_seen=now,
                        last_seen=now,
                    )
                )
                used.append(True)
            else:
                track = self._tracks[index]
                track.position = self._smooth(track.position, detection.position)
                track.label = detection.label or track.label
                track.size = detection.size or track.size
                track.hits += 1
                track.misses = 0
                track.last_seen = now
                used[index] = True

        for index, track in enumerate(self._tracks):
            if not used[index]:
                track.misses += 1

        self._purge(now)

    def _closest_track(self, detection: ObjectDetection, used: Sequence[bool]) -> int | None:
        best_index: int | None = None
        best_distance = self.match_distance_mm
        for index, track in enumerate(self._tracks):
            if used[index] or track.marker_id != detection.marker_id:
                continue
            distance = track.position.distance_to(detection.position)
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def _smooth(self, previous: Position, new: Position) -> Position:
        if self.smoothing <= 0.0:
            return previous
        if self.smoothing >= 1.0:
            return new
        return previous.lerp(new, self.smoothing)

    def _purge(self, now: float) -> None:
        self._tracks = [t for t in self._tracks if (now - t.last_seen) <= self.max_age_s]

    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        self._tracks.clear()

    @property
    def tracks(self) -> list[TrackedObject]:
        return list(self._tracks)

    @property
    def count(self) -> int:
        return len(self._tracks)

    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict[str, list[dict]]:
        """Vue sérialisable : ``{"<label>": [objets...]}``."""
        result: dict[str, list[dict]] = {}
        for track in self._tracks:
            key = track.label or str(track.marker_id)
            result.setdefault(key, []).append(track.to_dict())
        return result

    def flat(self) -> list[dict]:
        """Liste à plat, triée par identifiant puis distance au centre."""
        items = [track.to_dict() for track in self._tracks]
        items.sort(key=lambda item: (item["id"], np.hypot(item["x"], item["y"])))
        return items
