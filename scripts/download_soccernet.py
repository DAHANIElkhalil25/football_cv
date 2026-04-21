"""CLI de téléchargement SoccerNet-Tracking."""

from __future__ import annotations

import argparse
import getpass
import os
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader import (
	download_soccernet_tracking,
	import_soccernet_from_zip,
	initialize_soccernet_layout,
	normalize_requested_splits,
)


def parse_args() -> argparse.Namespace:
	"""Parse les arguments CLI de téléchargement."""
	parser = argparse.ArgumentParser(description="Télécharger SoccerNet-Tracking")
	parser.add_argument("--local_dir", type=str, default="./data/raw/soccernet", help="Dossier local cible")
	parser.add_argument("--password", type=str, default="", help="Mot de passe SoccerNet")
	parser.add_argument("--password_file", type=str, default="", help="Fichier texte contenant le mot de passe")
	parser.add_argument("--ask_password", action="store_true", help="Saisir le mot de passe de manière interactive (masqué)")
	parser.add_argument("--zip_path", type=str, default="", help="Chemin d'un ZIP SoccerNet local (sans compte)")
	parser.add_argument("--zip_url", type=str, default="", help="URL d'un ZIP SoccerNet public (sans compte)")
	parser.add_argument(
		"--splits",
		type=str,
		default="train,test,challenge",
		help="Splits à télécharger dans l'ordre (ex: train,val,challenge ; val=alias de test)",
	)
	parser.add_argument(
		"--init_only",
		action="store_true",
		help="Initialiser seulement l'arborescence locale (sans téléchargement)",
	)
	parser.add_argument("--download_videos", action="store_true", help="Télécharger aussi les images")
	parser.add_argument("--force_redownload", action="store_true", help="Forcer le re-téléchargement")
	return parser.parse_args()


def main() -> None:
	"""Point d'entrée CLI."""
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
	args = parse_args()
	logger = logging.getLogger("download_soccernet")

	if args.init_only:
		stats = initialize_soccernet_layout(local_dir=args.local_dir)
		logger.info("Mode init terminé: %s", stats)
		return

	if args.zip_path or args.zip_url:
		stats = import_soccernet_from_zip(
			local_dir=args.local_dir,
			zip_path=args.zip_path or None,
			zip_url=args.zip_url or None,
			force_redownload=args.force_redownload,
		)
		logger.info("Import ZIP terminé: %s", stats)
		return

	password = ""
	if args.password_file:
		password_path = Path(args.password_file).expanduser().resolve()
		if not password_path.exists():
			raise FileNotFoundError(f"Fichier mot de passe introuvable: {password_path}")
		password = password_path.read_text(encoding="utf-8").strip()
	elif args.ask_password:
		password = getpass.getpass("SoccerNet Tracking password: ").strip()
	else:
		password = (args.password or os.getenv("SOCCERNET_PASSWORD", "")).strip()

	if not password:
		password = "SoccerNet"
		logger.warning(
			"Aucun mot de passe fourni: utilisation du mot de passe par défaut SoccerNet ('SoccerNet')."
		)

	raw_splits = [item.strip() for item in args.splits.split(",") if item.strip()]
	splits = normalize_requested_splits(raw_splits)
	logger.info("Splits demandés (ordre): %s", splits)

	try:
		stats = download_soccernet_tracking(
			local_dir=args.local_dir,
			password=password,
			splits=splits,
			download_videos=args.download_videos,
			force_redownload=args.force_redownload,
		)
		logger.info("Résumé téléchargement: %s", stats["totals"])
	except PermissionError as exc:
		logger.error("%s", exc)
		logger.error(
			"Action: utilisez --ask_password ou --password_file pour éviter les erreurs de caractères spéciaux. "
			"Si 401 persiste, demandez l'accès/mot de passe spécifique SoccerNet Tracking-2023."
		)
		raise SystemExit(2)
	except (ConnectionError, RuntimeError, ValueError, FileNotFoundError) as exc:
		logger.error("%s", exc)
		raise SystemExit(2)


if __name__ == "__main__":
	main()
