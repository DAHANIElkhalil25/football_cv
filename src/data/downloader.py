"""
Module : downloader.py
But    : Télécharger le dataset SoccerNet-Tracking via l'API officielle
Auteur : Elkhalil DAHANI  INSEA PFE 2025-2026
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List
import inspect
from urllib.request import urlretrieve
from urllib.request import build_opener, HTTPBasicAuthHandler, HTTPPasswordMgrWithDefaultRealm
from urllib.request import Request
import zipfile
from urllib.error import URLError, HTTPError
import time

from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

TRACKING_2023_WEBDAV_BASE = "https://exrcsdrive.kaust.edu.sa/public.php/webdav"
TRACKING_2023_PUBLIC_USER = "OP7fl7h25NqGfcN"
VALID_SPLITS = {"train", "test", "challenge"}
SPLIT_ALIASES = {
    "val": "test",
    "validation": "test",
}


def normalize_requested_splits(splits: List[str]) -> List[str]:
    """Normalise et valide les splits demandés.

    - Accepte les alias (`val`, `validation`) et les mappe vers `test`.
    - Déduplique en conservant l'ordre.
    """
    normalized: List[str] = []
    seen = set()

    for split in splits:
        key = split.lower().strip()
        key = SPLIT_ALIASES.get(key, key)
        if key not in VALID_SPLITS:
            raise ValueError(f"Split invalide: '{split}'. Splits autorisés: {sorted(VALID_SPLITS)}")
        if key not in seen:
            normalized.append(key)
            seen.add(key)

    if not normalized:
        raise ValueError("Aucun split valide fourni.")

    return normalized


def _safe_extract_zip(zip_file: zipfile.ZipFile, extract_to: Path) -> None:
    """Extrait un ZIP de manière sûre (protection zip-slip)."""
    extract_to_resolved = extract_to.resolve()
    for member in zip_file.infolist():
        member_path = extract_to / member.filename
        member_resolved = member_path.resolve()
        if not str(member_resolved).startswith(str(extract_to_resolved)):
            raise ValueError(f"Archive ZIP invalide (chemin dangereux): {member.filename}")
    zip_file.extractall(extract_to)


def _count_soccernet_sequences(root: Path) -> int:
    """Compte les séquences SoccerNet valides (gt + img1)."""
    if not root.exists():
        return 0
    count = 0
    for split in ["train", "test", "challenge"]:
        split_root = root / split
        if not split_root.exists():
            continue
        for candidate in split_root.rglob("*"):
            if candidate.is_dir() and (candidate / "gt" / "gt.txt").exists() and (candidate / "img1").exists():
                count += 1
    return count


def initialize_soccernet_layout(local_dir: str) -> Dict[str, Any]:
    """Initialise une arborescence SoccerNet locale sans téléchargement.

    Cette méthode est utile lorsqu'aucun compte SoccerNet n'est disponible.
    Elle crée la structure attendue pour permettre l'import manuel ultérieur.
    """
    _configure_logging()

    output_root = Path(local_dir).expanduser().resolve()
    created = []

    for split in ["train", "test"]:
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        placeholder = split_dir / ".gitkeep"
        if not placeholder.exists():
            placeholder.write_text("", encoding="utf-8")
            created.append(str(placeholder))

    readme = output_root / "README_IMPORT.md"
    if not readme.exists():
        readme.write_text(
            "\n".join(
                [
                    "Structure attendue pour conversion SoccerNet -> YOLO:",
                    "",
                    "<local_dir>/train/<sequence>/img1/*.jpg",
                    "<local_dir>/train/<sequence>/gt/gt.txt",
                    "<local_dir>/test/<sequence>/img1/*.jpg",
                    "<local_dir>/test/<sequence>/gt/gt.txt",
                    "",
                    "Vous pouvez ensuite lancer:",
                    "python scripts/run_finetune.py --soccernet_dir <local_dir> --output_dir ./data/processed --device cpu",
                ]
            ),
            encoding="utf-8",
        )
        created.append(str(readme))

    stats = {
        "local_dir": str(output_root),
        "mode": "init",
        "created_files": created,
        "sequences_detected": _count_soccernet_sequences(output_root),
    }
    LOGGER.info("Arborescence locale initialisée dans %s", output_root)
    return stats


def import_soccernet_from_zip(
    local_dir: str,
    zip_path: str | None = None,
    zip_url: str | None = None,
    force_redownload: bool = False,
) -> Dict[str, Any]:
    """Importe SoccerNet-Tracking depuis une archive ZIP locale ou URL publique.

    Args:
        local_dir: Dossier local de destination.
        zip_path: Chemin local vers une archive `.zip`.
        zip_url: URL publique d'une archive `.zip`.
        force_redownload: Réimporte même si des fichiers sont présents.

    Returns:
        Statistiques d'import.

    Raises:
        ValueError: Si paramètres invalides.
        FileNotFoundError: Si le ZIP local est introuvable.
    """
    _configure_logging()

    if bool(zip_path) == bool(zip_url):
        raise ValueError("Fournissez exactement un seul paramètre: `zip_path` ou `zip_url`.")

    output_root = Path(local_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    existing_files = _safe_count_files(output_root)
    if existing_files > 0 and not force_redownload:
        LOGGER.info("Import ZIP ignoré: dossier déjà rempli (%d fichiers).", existing_files)
        return {
            "local_dir": str(output_root),
            "mode": "zip",
            "status": "skipped_already_exists",
            "files_before": existing_files,
            "files_after": existing_files,
            "files_added": 0,
            "sequences_detected": _count_soccernet_sequences(output_root),
        }

    if force_redownload and output_root.exists():
        for child in output_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

    archive_path: Path
    downloaded_tmp = False

    if zip_path:
        archive_path = Path(zip_path).expanduser().resolve()
        if not archive_path.exists():
            raise FileNotFoundError(f"ZIP introuvable: {archive_path}")
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="soccernet_zip_"))
        archive_path = temp_dir / "dataset.zip"
        LOGGER.info("Téléchargement de l'archive depuis %s", zip_url)
        urlretrieve(zip_url, str(archive_path))
        downloaded_tmp = True

    with zipfile.ZipFile(archive_path, "r") as zf:
        _safe_extract_zip(zf, output_root)

    if downloaded_tmp:
        try:
            shutil.rmtree(archive_path.parent, ignore_errors=True)
        except Exception:
            pass

    files_after = _safe_count_files(output_root)
    stats = {
        "local_dir": str(output_root),
        "mode": "zip",
        "status": "imported",
        "source": str(zip_path) if zip_path else str(zip_url),
        "files_before": 0 if force_redownload else existing_files,
        "files_after": files_after,
        "files_added": files_after - (0 if force_redownload else existing_files),
        "sequences_detected": _count_soccernet_sequences(output_root),
    }
    LOGGER.info("Import ZIP terminé: fichiers=%d, séquences détectées=%d", files_after, stats["sequences_detected"])
    return stats


def _configure_logging() -> None:
    """Configure un logger standard si aucun handler n'est présent."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def _safe_count_files(root: Path) -> int:
    """Compte les fichiers présents dans un dossier de manière robuste.

    Args:
        root: Dossier racine.

    Returns:
        Nombre total de fichiers trouvés récursivement.
    """
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file())


def _split_looks_downloaded(split_dir: Path, require_images: bool, split_name: str | None = None) -> bool:
    """Vérifie rapidement si un split semble déjà téléchargé.

    Args:
        split_dir: Répertoire du split (ex: train/test).
        require_images: Si True, les images doivent être présentes.

    Returns:
        True si les artefacts minimum sont présents, sinon False.
    """
    if not split_dir.exists():
        return False

    split_key = (split_name or split_dir.name).lower().strip()

    if split_key == "challenge":
        if require_images:
            return any(split_dir.rglob("img1")) or any(split_dir.rglob("*.jpg"))
        return any(p.is_file() for p in split_dir.rglob("*"))

    has_gt = any(split_dir.rglob("gt.txt"))
    if not has_gt:
        return False

    if require_images:
        has_images = any(split_dir.rglob("img1")) or any(split_dir.rglob("*.jpg"))
        return has_images

    return True


def _build_downloader(local_dir: str, password: str) -> Any:
    """Construit une instance de SoccerNetDownloader.

    Args:
        local_dir: Dossier local de stockage.
        password: Mot de passe SoccerNet.

    Returns:
        Instance configurée de SoccerNetDownloader.

    Raises:
        ImportError: Si la librairie SoccerNet est absente.
        RuntimeError: Si la construction de l'API échoue.
    """
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except Exception as exc:
        raise ImportError(
            "La librairie SoccerNet est introuvable. Installez-la via `pip install SoccerNet`."
        ) from exc

    try:
        downloader = SoccerNetDownloader(LocalDirectory=local_dir)
        downloader.password = password
        return downloader
    except Exception as exc:
        raise RuntimeError(f"Échec d'initialisation SoccerNetDownloader: {exc}") from exc


def _normalize_tracking_layout(output_root: Path, split: str) -> None:
    """Normalise la structure locale en `<root>/<split>/...`.

    Certaines versions de SoccerNet téléchargent sous `tracking-2023/<split>/...`.
    Cette fonction remonte les dossiers vers la racine attendue par le convertisseur.
    """
    target_split = output_root / split
    if target_split.exists():
        return

    for candidate_parent in ["tracking-2023", "tracking"]:
        candidate = output_root / candidate_parent / split
        if candidate.exists() and candidate.is_dir():
            target_split.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidate), str(target_split))
            break


def _extract_downloaded_split_archives(output_root: Path, split: str) -> List[str]:
    """Extrait les archives de split si SoccerNet les télécharge en `.zip`.

    Returns:
        Liste des archives extraites.
    """
    extracted: List[str] = []
    candidates = [
        output_root / "tracking" / f"{split}.zip",
        output_root / "tracking-2023" / f"{split}.zip",
        output_root / f"{split}.zip",
    ]
    if split == "challenge":
        candidates.extend(
            [
                output_root / "tracking" / "challenge2023.zip",
                output_root / "tracking-2023" / "challenge2023.zip",
                output_root / "challenge2023.zip",
            ]
        )

    for archive in candidates:
        if not archive.exists() or not archive.is_file():
            continue
        try:
            if archive.stat().st_size < 1024:
                archive.unlink(missing_ok=True)
                raise ValueError("Archive trop petite, probablement incomplète")
            with zipfile.ZipFile(archive, "r") as zf:
                if zf.testzip() is not None:
                    raise ValueError("Archive ZIP corrompue")
                _safe_extract_zip(zf, output_root)
            extracted.append(str(archive))
            _normalize_tracking_layout(output_root=output_root, split=split)
        except Exception as exc:
            LOGGER.warning("Extraction archive '%s' échouée: %s", archive, exc)
            try:
                archive.unlink(missing_ok=True)
            except Exception:
                pass

    return extracted


def _cleanup_stale_split_archives(output_root: Path, split: str) -> List[str]:
    """Supprime les archives de split incomplètes/corrompues avant relance download."""
    removed: List[str] = []
    candidates = [
        output_root / "tracking" / f"{split}.zip",
        output_root / "tracking-2023" / f"{split}.zip",
        output_root / f"{split}.zip",
    ]
    if split == "challenge":
        candidates.extend(
            [
                output_root / "tracking" / "challenge2023.zip",
                output_root / "tracking-2023" / "challenge2023.zip",
                output_root / "challenge2023.zip",
            ]
        )

    for archive in candidates:
        if not archive.exists() or not archive.is_file():
            continue
        should_remove = False
        try:
            if archive.stat().st_size < 1024:
                should_remove = True
            else:
                with zipfile.ZipFile(archive, "r") as zf:
                    should_remove = zf.testzip() is not None
        except Exception:
            should_remove = True

        if should_remove:
            try:
                archive.unlink(missing_ok=True)
                removed.append(str(archive))
            except Exception:
                pass

    return removed


def _call_download_data_task(downloader: Any, task: str, split: str, password: str) -> None:
    """Appelle `downloadDataTask` en s'adaptant à la signature disponible."""
    method = getattr(downloader, "downloadDataTask")
    sig = inspect.signature(method)
    kwargs: Dict[str, Any] = {}

    if "split" in sig.parameters:
        kwargs["split"] = [split]
    if "password" in sig.parameters:
        kwargs["password"] = password
    if "verbose" in sig.parameters:
        kwargs["verbose"] = True

    if "source" in sig.parameters:
        try:
            method(task=task, source="OwnCloud", **kwargs)
            return
        except Exception:
            pass

    method(task=task, **kwargs)


def _download_file_webdav(
    url: str,
    user: str,
    password: str,
    output_file: Path,
    retries: int = 3,
    timeout_sec: int = 60,
) -> None:
    """Télécharge un fichier via WebDAV avec Basic Auth et retries."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    pwd_mgr = HTTPPasswordMgrWithDefaultRealm()
    pwd_mgr.add_password(None, TRACKING_2023_WEBDAV_BASE, user, password)
    opener = build_opener(HTTPBasicAuthHandler(pwd_mgr))

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with opener.open(url, timeout=timeout_sec) as response, output_file.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            return
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(min(2 * attempt, 5))

    if last_exc is not None:
        raise last_exc


def _download_split_direct_webdav(output_root: Path, split: str, password: str) -> Dict[str, Any]:
    """Fallback direct pour `tracking-2023` sans wrapper SoccerNet."""
    archive_names = [f"{split}.zip"]
    if split == "challenge":
        archive_names = ["challenge2023.zip", "challenge.zip"]

    last_exc: Exception | None = None
    for archive_name in archive_names:
        archive_path = output_root / "tracking-2023" / archive_name
        remote_url = f"{TRACKING_2023_WEBDAV_BASE}/{archive_name}"
        try:
            _download_file_webdav(
                url=remote_url,
                user=TRACKING_2023_PUBLIC_USER,
                password=password,
                output_file=archive_path,
                retries=3,
                timeout_sec=90,
            )

            with zipfile.ZipFile(archive_path, "r") as zf:
                _safe_extract_zip(zf, output_root)

            try:
                archive_path.unlink(missing_ok=True)
            except Exception:
                pass

            _normalize_tracking_layout(output_root=output_root, split=split)
            return {"strategy": f"direct-webdav:tracking-2023:{archive_name}", "status": "ok"}
        except Exception as exc:
            last_exc = exc

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Aucune archive challenge/train/test disponible via WebDAV.")


def _verify_tracking_credentials(password: str, timeout_sec: int = 20) -> None:
    """Vérifie rapidement les credentials WebDAV pour Tracking-2023.

    Lève une erreur explicite en cas de 401 avant la boucle de téléchargement.
    """
    pwd_mgr = HTTPPasswordMgrWithDefaultRealm()
    pwd_mgr.add_password(None, TRACKING_2023_WEBDAV_BASE, TRACKING_2023_PUBLIC_USER, password)
    opener = build_opener(HTTPBasicAuthHandler(pwd_mgr))

    probe_url = f"{TRACKING_2023_WEBDAV_BASE}/train.zip"
    request = Request(probe_url, method="HEAD")
    try:
        with opener.open(request, timeout=timeout_sec):
            return
    except HTTPError as exc:
        if exc.code == 401:
            raise PermissionError(
                "Accès refusé (401) au dataset SoccerNet Tracking-2023. "
                "Vérifiez le mot de passe tracking (différent possible du compte SoccerNet) "
                "et attention aux caractères spéciaux PowerShell."
            ) from exc
        raise


def _download_split_with_fallback(
    downloader: Any,
    split: str,
    include_images: bool,
    output_root: Path,
    password: str,
) -> Dict[str, Any]:
    """Télécharge un split via plusieurs stratégies compatibles versions SoccerNet.

    Args:
        downloader: Instance SoccerNetDownloader.
        split: Nom du split (train/test).
        include_images: True pour récupérer images + labels, False labels uniquement.

    Returns:
        Dictionnaire d'état de téléchargement.

    Raises:
        RuntimeError: Si toutes les stratégies échouent.
    """
    files_labels = ["gameinfo.ini", "gt/gt.txt"]
    files_images = ["img1"]

    errors: List[str] = []

    if hasattr(downloader, "downloadDataTask"):
        for task_name in ["tracking", "tracking-2023"]:
            try:
                _call_download_data_task(
                    downloader=downloader,
                    task=task_name,
                    split=split,
                    password=password,
                )
                _normalize_tracking_layout(output_root=output_root, split=split)
                return {"strategy": f"downloadDataTask:{task_name}", "status": "ok"}
            except Exception as exc:
                errors.append(f"downloadDataTask({task_name}): {exc}")

    try:
        state = _download_split_direct_webdav(output_root=output_root, split=split, password=password)
        return state
    except Exception as exc:
        errors.append(f"direct-webdav({split}): {exc}")

    if hasattr(downloader, "downloadGames"):
        try:
            files = list(files_labels)
            if include_images:
                files.extend(files_images)
            for task_name in ["tracking", "tracking-2023"]:
                try:
                    downloader.downloadGames(files=files, split=[split], task=task_name)
                    _normalize_tracking_layout(output_root=output_root, split=split)
                    return {"strategy": f"downloadGames-task:{task_name}", "status": "ok"}
                except Exception as inner_exc:
                    errors.append(f"downloadGames-task({task_name}): {inner_exc}")
            raise RuntimeError("Aucun appel downloadGames(task=...) n'a réussi.")
        except TypeError:
            try:
                files = list(files_labels)
                if include_images:
                    files.extend(files_images)
                downloader.downloadGames(files=files, split=[split])
                _normalize_tracking_layout(output_root=output_root, split=split)
                return {"strategy": "downloadGames-generic", "status": "ok"}
            except Exception as exc:
                errors.append(f"downloadGames-generic: {exc}")
        except Exception as exc:
            errors.append(f"downloadGames-task: {exc}")

    message = " | ".join(errors) if errors else "Aucune méthode de téléchargement compatible trouvée."
    raise RuntimeError(message)


def download_soccernet_tracking(
    local_dir: str,
    password: str,
    splits: List[str] = ["train", "test", "challenge"],
    download_videos: bool = False,
    force_redownload: bool = False,
) -> Dict[str, Any]:
    """Télécharge SoccerNet-Tracking (labels et optionnellement images).

    Args:
        local_dir: Dossier local où stocker le dataset.
        password: Mot de passe SoccerNet pour accès aux données.
        splits: Splits à télécharger (`train`, `test`).
        download_videos: Si True, télécharge labels + images; sinon labels seulement.
        force_redownload: Si True, retélécharge même si les fichiers semblent présents.

    Returns:
        Dictionnaire contenant les chemins et statistiques de téléchargement.

    Raises:
        ValueError: Si les paramètres sont invalides.
        ConnectionError: Si erreur réseau/credentials probable.
        RuntimeError: Si le téléchargement échoue complètement.
    """
    _configure_logging()

    requested = normalize_requested_splits(splits)

    output_root = Path(local_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        _verify_tracking_credentials(password=password)
    except PermissionError as exc:
        LOGGER.warning(
            "Pré-vérification credentials non concluante (%s). Tentative de téléchargement réel via SoccerNet...",
            exc,
        )
    except Exception as exc:
        LOGGER.warning(
            "Pré-vérification credentials ignorée (%s). Tentative de téléchargement réel via SoccerNet...",
            exc,
        )

    downloader = _build_downloader(str(output_root), password=password)

    stats: Dict[str, Any] = {
        "local_dir": str(output_root),
        "splits_requested": requested,
        "download_videos": download_videos,
        "force_redownload": force_redownload,
        "splits": {},
        "totals": {
            "downloaded_splits": 0,
            "skipped_splits": 0,
            "failed_splits": 0,
            "files_before": _safe_count_files(output_root),
            "files_after": 0,
            "files_added": 0,
        },
    }

    LOGGER.info("Démarrage du téléchargement SoccerNet-Tracking dans %s", output_root)

    for split in tqdm(requested, desc="Téléchargement SoccerNet", unit="split"):
        split_dir = output_root / split
        split_result: Dict[str, Any] = {
            "path": str(split_dir),
            "status": "pending",
            "strategy": None,
            "error": None,
        }

        try:
            if not force_redownload and _split_looks_downloaded(
                split_dir,
                require_images=download_videos,
                split_name=split,
            ):
                split_result["status"] = "skipped_already_exists"
                stats["totals"]["skipped_splits"] += 1
                LOGGER.info("Split '%s' ignoré (déjà présent).", split)
            else:
                removed_stale = _cleanup_stale_split_archives(output_root=output_root, split=split)
                if removed_stale:
                    LOGGER.warning("Archives corrompues supprimées pour split '%s': %s", split, removed_stale)
                state = _download_split_with_fallback(
                    downloader=downloader,
                    split=split,
                    include_images=download_videos,
                    output_root=output_root,
                    password=password,
                )
                extracted_archives = _extract_downloaded_split_archives(output_root=output_root, split=split)
                if extracted_archives:
                    LOGGER.info("Archives extraites pour split '%s': %s", split, extracted_archives)
                if not _split_looks_downloaded(
                    split_dir,
                    require_images=download_videos,
                    split_name=split,
                ):
                    raise RuntimeError(
                        "Téléchargement signalé mais artefacts absents (gt/img1). "
                        "Vérifiez mot de passe, droits d'accès SoccerNet et disponibilité du split."
                    )
                split_result["status"] = "downloaded"
                split_result["strategy"] = state.get("strategy")
                stats["totals"]["downloaded_splits"] += 1
                LOGGER.info("Split '%s' téléchargé via %s.", split, split_result["strategy"])
        except Exception as exc:
            split_result["status"] = "failed"
            split_result["error"] = str(exc)
            stats["totals"]["failed_splits"] += 1
            LOGGER.error("Échec split '%s': %s", split, exc)

        stats["splits"][split] = split_result

    stats["totals"]["files_after"] = _safe_count_files(output_root)
    stats["totals"]["files_added"] = stats["totals"]["files_after"] - stats["totals"]["files_before"]

    if stats["totals"]["failed_splits"] == len(requested):
        if stats["totals"]["files_added"] <= 0:
            raise ConnectionError(
                "Tous les téléchargements ont échoué et aucun fichier n'a été ajouté. "
                "Cause probable: mot de passe/droits invalides pour SoccerNet Tracking-2023 (OwnCloud). "
                "Le mot de passe de compte SoccerNet peut être différent du mot de passe du dataset tracking. "
                "Alternative: utilisez --zip_path / --zip_url pour importer une archive locale/publique."
            )
        raise ConnectionError(
            "Tous les téléchargements ont échoué. Vérifiez la connexion internet et les credentials SoccerNet."
        )

    LOGGER.info(
        "Téléchargement terminé: téléchargés=%s, ignorés=%s, échoués=%s, fichiers ajoutés=%s",
        stats["totals"]["downloaded_splits"],
        stats["totals"]["skipped_splits"],
        stats["totals"]["failed_splits"],
        stats["totals"]["files_added"],
    )

    return stats


__all__ = [
    "download_soccernet_tracking",
    "import_soccernet_from_zip",
    "initialize_soccernet_layout",
]

