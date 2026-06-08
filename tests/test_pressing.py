import numpy as np

from src.tactics.pressing import (
    assign_ball_carrier,
    pressure_intensity,
    closing_speed,
    count_pressing_events,
    ppda_v2,
    calibrate_pressure_radius,
)


def _frame(players, teams, ball):
    return {"players_xy": np.array(players, dtype=float), "teams": np.array(teams), "ball_xy": ball}


def test_assign_ball_carrier_picks_nearest_in_range():
    players = np.array([[10.0, 10.0], [60.0, 34.0]])
    idx, team = assign_ball_carrier(players, [0, 1], (61.0, 34.0))
    assert idx == 1 and team == 1


def test_assign_ball_carrier_none_when_far():
    players = np.array([[10.0, 10.0]])
    idx, team = assign_ball_carrier(players, [0], (90.0, 60.0), max_carrier_dist_m=3.0)
    assert idx is None and team is None


def test_pressure_intensity_decreases_with_distance():
    near = pressure_intensity(np.array([[60.0, 34.0]]), (61.0, 34.0))
    far = pressure_intensity(np.array([[60.0, 34.0]]), (70.0, 34.0))
    assert near > far


def test_closing_speed_positive_when_approaching():
    v = closing_speed((50.0, 34.0), (52.0, 34.0), (60.0, 34.0), dt=1.0)
    assert v > 0


def test_count_events_collapses_consecutive_frames():
    # carrier (team 1) at x=80 (offensive zone for team attacking right),
    # an opponent (team 0) glued for 10 consecutive frames -> 1 event, not 10
    frames = []
    for _ in range(10):
        frames.append(_frame([[80.0, 34.0], [81.0, 34.0]], [1, 0], (80.0, 34.0)))
    out = count_pressing_events(frames, pressure_radius_m=4.0, restrict_zone=True,
                                pressing_team_attacks_right=True)
    assert out["pressing_events"] == 1
    assert out["frames_under_pressure"] == 10


def test_zone_restriction_filters_defensive_third():
    # carrier in own defensive zone (x small) -> not counted when attacking right
    frames = [_frame([[10.0, 34.0], [11.0, 34.0]], [1, 0], (10.0, 34.0)) for _ in range(5)]
    out = count_pressing_events(frames, restrict_zone=True, pressing_team_attacks_right=True)
    assert out["pressing_events"] == 0


def test_ppda_v2_runs():
    frames = [_frame([[80.0, 34.0], [81.0, 34.0]], [1, 0], (80.0, 34.0)) for _ in range(6)]
    res = ppda_v2(opponent_passes=20, frames=frames)
    assert res["pressing_events"] >= 1 and res["ppda"] > 0


def test_calibrate_radius_returns_best():
    frames = [_frame([[80.0, 34.0], [83.0, 34.0]], [1, 0], (80.0, 34.0)) for _ in range(4)]
    out = calibrate_pressure_radius(frames, target_actions=1, radii=(1.0, 4.0))
    assert out["best_radius"] in (1.0, 4.0)
    assert "curve" in out
