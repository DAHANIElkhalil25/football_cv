"""
Module : pressing.py
But    : Estimer un indicateur de pressing type PPDA
Auteur : Elkhalil DAHANI — INSEA PFE 2025-2026

Améliorations v2
----------------
La version initiale comptait une « action défensive » pour CHAQUE frame où un
défenseur était à moins de 4 m du ballon. À 25 fps, une seconde de proximité
comptait pour 25 actions : le chiffre n'était donc pas comparable aux comptes
événementiels de StatsBomb, indépendamment du rayon choisi.

La v2 corrige cela sans aucune donnée ni matériel supplémentaire :
- comptage PAR ÉVÉNEMENT (debounce / fusion des frames consécutives) ;
- attribution du porteur de balle puis filtre « adversaires uniquement » ;
- restriction de zone (tiers/60 % offensifs, définition standard du PPDA) ;
- score de pression CONTINU (décroissance exponentielle avec la distance) ;
- prise en compte de la VITESSE DE FERMETURE (un défenseur qui ferme vite
  presse, même un peu plus loin) ;
- calibration du rayon contre une cible (ex: événements StatsBomb).

Les fonctions de la v1 sont conservées telles quelles pour compatibilité.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ================================================================
# v1 — conservé pour compatibilité
# ================================================================
def estimate_ppda(
    opponent_passes: int,
    defensive_actions: int,
    epsilon: float = 1e-6,
) -> float:
    """Calcule le PPDA simple: passes adverses / actions défensives.

    Plus la valeur est faible, plus le pressing est intense.
    """
    if opponent_passes < 0 or defensive_actions < 0:
        raise ValueError("Les compteurs doivent être positifs.")
    return float(opponent_passes) / float(defensive_actions + epsilon)


def estimate_defensive_actions_from_tracking(
    defender_xy: np.ndarray,
    ball_xy: np.ndarray,
    pressure_radius_m: float = 4.0,
) -> int:
    """[v1] Approxime les actions défensives par proximité balle-défenseur.

    ATTENTION : comptage PAR FRAME (surévalue le pressing). Conservé pour
    rétro-compatibilité. Préférer `count_pressing_events` en v2.
    """
    if len(defender_xy) == 0 or len(ball_xy) == 0:
        return 0
    actions = 0
    for b in ball_xy:
        distances = np.linalg.norm(defender_xy - b[None, :], axis=1)
        if float(distances.min()) <= pressure_radius_m:
            actions += 1
    return actions


def pressing_summary(opponent_passes: int, defensive_actions: int) -> Dict[str, float | str]:
    """Retourne un résumé interprétable du pressing."""
    ppda = estimate_ppda(opponent_passes=opponent_passes, defensive_actions=defensive_actions)
    if ppda < 8.0:
        level = "high"
    elif ppda < 12.0:
        level = "medium"
    else:
        level = "low"
    return {"ppda": ppda, "pressing_level": level}


# ================================================================
# v2 — pressing contextuel par événement
# ================================================================
def assign_ball_carrier(
    players_xy: np.ndarray,
    player_teams: Sequence[int],
    ball_xy: Tuple[float, float],
    max_carrier_dist_m: float = 3.0,
) -> Tuple[Optional[int], Optional[int]]:
    """Attribue le porteur de balle (joueur le plus proche du ballon).

    Args:
        players_xy: (N,2) positions joueurs en mètres (BEV).
        player_teams: équipe de chaque joueur (même ordre).
        ball_xy: position du ballon (x,y) en mètres.
        max_carrier_dist_m: au-delà, on considère qu'aucun joueur ne porte.

    Returns:
        (indice_du_porteur, equipe_du_porteur) ou (None, None).
    """
    if players_xy is None or len(players_xy) == 0:
        return None, None
    d = np.linalg.norm(players_xy - np.asarray(ball_xy, dtype=float)[None, :], axis=1)
    idx = int(np.argmin(d))
    if float(d[idx]) > max_carrier_dist_m:
        return None, None
    return idx, int(player_teams[idx])


def pressure_intensity(
    defenders_xy: np.ndarray,
    carrier_xy: Tuple[float, float],
    decay_m: float = 4.0,
) -> float:
    """Score de pression CONTINU sur le porteur (0 = aucune pression).

    Remplace le seuil binaire à 4 m par une somme de contributions à
    décroissance exponentielle : exp(-d / decay_m). Un défenseur collé pèse ~1,
    un défenseur à `decay_m` pèse ~0.37, au-delà l'effet s'estompe en douceur.
    """
    if defenders_xy is None or len(defenders_xy) == 0:
        return 0.0
    d = np.linalg.norm(defenders_xy - np.asarray(carrier_xy, dtype=float)[None, :], axis=1)
    return float(np.exp(-d / max(1e-6, decay_m)).sum())


def closing_speed(
    defender_xy_prev: Tuple[float, float],
    defender_xy_now: Tuple[float, float],
    carrier_xy_now: Tuple[float, float],
    dt: float,
) -> float:
    """Vitesse de fermeture (m/s) d'un défenseur vers le porteur.

    Positive = le défenseur se rapproche du porteur. Projetée sur l'axe
    défenseur→porteur pour ne garder que la composante « pression ».
    """
    if dt <= 0:
        return 0.0
    p0 = np.asarray(defender_xy_prev, dtype=float)
    p1 = np.asarray(defender_xy_now, dtype=float)
    c = np.asarray(carrier_xy_now, dtype=float)
    direction = c - p1
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return 0.0
    direction = direction / norm
    velocity = (p1 - p0) / dt
    return float(np.dot(velocity, direction))


def _in_pressing_zone(
    carrier_xy: Tuple[float, float],
    attack_to_right: bool,
    pitch_length: float,
    offensive_fraction: float,
) -> bool:
    """Le porteur est-il dans la zone où le pressing « compte » (PPDA standard) ?

    PPDA = actions hors des 40 % défensifs de l'équipe qui presse,
    soit les `offensive_fraction` (≈ 0.6) du terrain côté adverse.
    """
    x = float(carrier_xy[0])
    if attack_to_right:
        return x >= pitch_length * (1.0 - offensive_fraction)
    return x <= pitch_length * offensive_fraction


def count_pressing_events(
    frames: List[Dict],
    pressure_radius_m: float = 4.0,
    cooldown_frames: int = 5,
    restrict_zone: bool = True,
    pressing_team_attacks_right: bool = True,
    pitch_length: float = 105.0,
    offensive_fraction: float = 0.6,
) -> Dict[str, int]:
    """Compte les ÉVÉNEMENTS de pression (et non les frames).

    Une frame est « sous pression » si un adversaire du porteur est à <= rayon
    (et, si activé, si le porteur est en zone offensive). Des frames sous
    pression consécutives (ou séparées d'un trou <= cooldown) forment UN seul
    événement — ce qui rend le compte comparable aux événements StatsBomb.

    Args:
        frames: liste ordonnée de dicts par frame, chacun :
            {
              'players_xy': (N,2) en mètres,
              'teams':      (N,) équipe de chaque joueur,
              'ball_xy':    (x,y) en mètres ou None,
            }
        pressure_radius_m: rayon de pression.
        cooldown_frames: trou max (frames) fusionné dans le même événement.
        restrict_zone: appliquer la restriction de zone PPDA.
        pressing_team_attacks_right: sens d'attaque de l'équipe qui presse.
        pitch_length: longueur terrain (m).
        offensive_fraction: fraction offensive prise en compte (0.6 standard).

    Returns:
        Dict: nb d'événements, nb de frames sous pression, durée moyenne (frames).
    """
    under_pressure: List[bool] = []
    for fr in frames:
        ball = fr.get("ball_xy")
        players = fr.get("players_xy")
        teams = fr.get("teams")
        if ball is None or players is None or len(players) == 0:
            under_pressure.append(False)
            continue
        players = np.asarray(players, dtype=float)
        teams = np.asarray(teams)
        carrier_idx, carrier_team = assign_ball_carrier(players, teams, ball)
        if carrier_idx is None:
            under_pressure.append(False)
            continue
        if restrict_zone and not _in_pressing_zone(
            tuple(players[carrier_idx]), pressing_team_attacks_right, pitch_length, offensive_fraction
        ):
            under_pressure.append(False)
            continue
        opponents = players[teams != carrier_team]
        if len(opponents) == 0:
            under_pressure.append(False)
            continue
        dmin = float(np.linalg.norm(opponents - players[carrier_idx][None, :], axis=1).min())
        under_pressure.append(dmin <= pressure_radius_m)

    # Fusion en événements (debounce)
    events = 0
    frames_under = sum(under_pressure)
    i = 0
    n = len(under_pressure)
    while i < n:
        if under_pressure[i]:
            events += 1
            j = i
            miss = 0
            while j < n and miss <= cooldown_frames:
                if under_pressure[j]:
                    miss = 0
                else:
                    miss += 1
                j += 1
            i = j
        else:
            i += 1

    return {
        "pressing_events": int(events),
        "frames_under_pressure": int(frames_under),
        "mean_event_len_frames": float(frames_under / events) if events else 0.0,
    }


def ppda_v2(
    opponent_passes: int,
    frames: List[Dict],
    **kwargs,
) -> Dict[str, float]:
    """PPDA v2 : passes adverses / ÉVÉNEMENTS de pression (et non frames)."""
    counts = count_pressing_events(frames, **kwargs)
    actions = counts["pressing_events"]
    ppda = estimate_ppda(opponent_passes, actions)
    summary = pressing_summary(opponent_passes, actions)
    return {**counts, "ppda": ppda, "pressing_level": summary["pressing_level"]}


def calibrate_pressure_radius(
    frames: List[Dict],
    target_actions: int,
    radii: Sequence[float] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0),
    **kwargs,
) -> Dict[str, float]:
    """Ajuste le rayon de pression pour coller à une cible (ex: StatsBomb).

    Au lieu d'un seuil arbitraire de 4 m, on choisit le rayon qui minimise
    l'écart entre nos événements de pression et un nombre d'actions de référence
    mesuré sur le MÊME match (données ouvertes StatsBomb).

    Returns:
        Dict avec best_radius, best_events, abs_error et la courbe complète.
    """
    curve = {}
    best_r, best_err, best_ev = None, float("inf"), None
    for r in radii:
        ev = count_pressing_events(frames, pressure_radius_m=float(r), **kwargs)["pressing_events"]
        err = abs(ev - target_actions)
        curve[float(r)] = ev
        if err < best_err:
            best_err, best_r, best_ev = err, float(r), ev
    return {
        "best_radius": best_r,
        "best_events": best_ev,
        "target_actions": int(target_actions),
        "abs_error": float(best_err),
        "curve": curve,
    }


__all__ = [
    "estimate_ppda",
    "estimate_defensive_actions_from_tracking",
    "pressing_summary",
    "assign_ball_carrier",
    "pressure_intensity",
    "closing_speed",
    "count_pressing_events",
    "ppda_v2",
    "calibrate_pressure_radius",
]
