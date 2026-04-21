"""
Module : visualizer.py
But    : Dessins et graphiques pour analyse football CV
Auteur : Elkhalil DAHANI — INSEA PFE 2025-2026
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_bboxes(frame: np.ndarray, detections: List[Dict[str, object]], color: Tuple[int, int, int] = (0, 220, 0)) -> np.ndarray:
	"""Dessine des bounding boxes sur une frame BGR."""
	canvas = frame.copy()
	for det in detections:
		bbox = det.get("bbox_xyxy")
		if bbox is None:
			continue
		x1, y1, x2, y2 = [int(v) for v in bbox]
		label = f"c{int(det.get('class_id', -1))}:{float(det.get('confidence', 0.0)):.2f}"
		cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
		cv2.putText(canvas, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
	return canvas


def draw_tracks(frame: np.ndarray, tracks: List[Dict[str, object]]) -> np.ndarray:
	"""Dessine les tracks avec ID sur frame BGR."""
	canvas = frame.copy()
	for trk in tracks:
		bbox = trk.get("bbox_xyxy")
		tid = trk.get("track_id", -1)
		if bbox is None:
			continue
		x1, y1, x2, y2 = [int(v) for v in bbox]
		cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 180, 0), 2)
		cv2.putText(canvas, f"ID {tid}", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)
	return canvas


def plot_compactness_over_time(values: List[float], title: str = "Compacité défensive") -> plt.Figure:
	"""Construit une figure matplotlib de compacité."""
	fig, ax = plt.subplots(figsize=(8, 3.5))
	ax.plot(values, color="#1f77b4", linewidth=2)
	ax.set_title(title)
	ax.set_xlabel("Frame")
	ax.set_ylabel("Indice")
	ax.grid(alpha=0.3)
	fig.tight_layout()
	return fig


__all__ = ["draw_bboxes", "draw_tracks", "plot_compactness_over_time"]

