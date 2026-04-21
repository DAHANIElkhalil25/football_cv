"""Dashboard Streamlit pour détection + tracking football."""

from __future__ import annotations

from pathlib import Path

import cv2
import streamlit as st

from src.detection.predictor import YOLOPredictor
from src.tracking.tracker import ByteTrackerWrapper
from src.visualization.visualizer import draw_bboxes, draw_tracks


def _read_first_frame(video_path: Path):
	cap = cv2.VideoCapture(str(video_path))
	ok, frame = cap.read()
	cap.release()
	return ok, frame


def main() -> None:
	st.set_page_config(page_title="Football CV Tactical", layout="wide")
	st.title("⚽ Football CV Tactical Analysis")

	model_path = st.text_input("Modèle YOLO (.pt)", value="data/processed/runs/yolov8m_soccernet/weights/best.pt")
	video_path = st.text_input("Vidéo", value="")
	bytetrack_cfg = st.text_input("ByteTrack config", value="config/bytetrack.yaml")
	device = st.selectbox("Device", ["cpu", "cuda", "auto"], index=0)

	if st.button("Analyser la première frame"):
		model = Path(model_path).expanduser().resolve()
		video = Path(video_path).expanduser().resolve()
		cfg = Path(bytetrack_cfg).expanduser().resolve()

		if not model.exists() or not video.exists() or not cfg.exists():
			st.error("Vérifie les chemins du modèle, de la vidéo et du config tracker.")
			return

		ok, frame = _read_first_frame(video)
		if not ok:
			st.error("Impossible de lire la vidéo.")
			return

		predictor = YOLOPredictor(model_path=str(model), device=device)
		dets = predictor.predict_frame(frame)

		tracker = ByteTrackerWrapper(model_path=str(model), bytetrack_cfg=str(cfg), device=device)
		tracks_obj = tracker.update(frame)
		tracks = [
			{
				"track_id": t.track_id,
				"bbox_xyxy": t.bbox_xyxy,
				"class_id": t.class_id,
				"confidence": t.confidence,
			}
			for t in tracks_obj
		]

		col1, col2 = st.columns(2)
		with col1:
			st.subheader("Détections")
			img_det = draw_bboxes(frame, dets)
			st.image(cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB), use_container_width=True)
		with col2:
			st.subheader("Tracking")
			img_trk = draw_tracks(frame, tracks)
			st.image(cv2.cvtColor(img_trk, cv2.COLOR_BGR2RGB), use_container_width=True)

		st.write({"detections": len(dets), "tracks": len(tracks)})


if __name__ == "__main__":
	main()

