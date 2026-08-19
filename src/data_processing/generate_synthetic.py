"""
Synthetic season generator for Football Performance Intelligence.

Why synthetic data?
-------------------
Real club data is proprietary and, for medical/availability information,
sensitive. To demonstrate the full pipeline we simulate one professional
season for a squad using a *latent hazard* process:

    latent state (fatigue, acute:chronic load, neuromuscular fatigue,
                  sleep debt, susceptibility)
        -> observable measurements (GPS load, sRPE, wellness, CMJ)  [+ noise]
        -> daily hazard of entering a reduced-availability state
        -> availability episodes (modified training / unavailable)

Two design choices make this credible rather than a toy:

1. The model only ever *predicts* from the OBSERVABLE measurements (which are
   noisy proxies of the latent state), never from the latent state itself. So
   the signal is genuinely learnable but imperfect, exactly like real life.

2. Relationships are intentionally *not* deterministic: there is a base event
   rate, heavy noise on every channel, and events that fire for unlucky reasons.
   A model that scored AUC 0.99 here would be a red flag, not a success.

The generator is fully seeded and reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

RNG_SEED = config.RANDOM_SEED

FIRST_NAMES = [
    "Lucas", "Mateo", "Adam", "Noah", "Enzo", "Gabriel", "Louis", "Raphael",
    "Arthur", "Jules", "Hugo", "Leo", "Ethan", "Nathan", "Aaron", "Sacha",
    "Tom", "Marius", "Nolan", "Ilan", "Diego", "Kylian", "Yanis", "Rayan",
    "Theo", "Malik", "Oscar", "Victor", "Amine", "Bilal",
]
LAST_NAMES = [
    "Martin", "Bernard", "Dubois", "Moreau", "Laurent", "Simon", "Michel",
    "Garcia", "Roux", "Fournier", "Girard", "Bonnet", "Dupont", "Lambert",
    "Fontaine", "Rousseau", "Vincent", "Muller", "Faure", "Blanc",
    "Guerin", "Da Silva", "Mercier", "Chevalier", "Robin", "Clement",
    "Gauthier", "Perrin", "Morel", "Andre",
]


# --------------------------------------------------------------------------- #
# Squad
# --------------------------------------------------------------------------- #
def _build_players(rng: np.random.Generator) -> pd.DataFrame:
    positions = []
    for pos, n in config.POSITION_COUNTS.items():
        positions.extend([pos] * n)
    positions = positions[: config.N_PLAYERS]
    while len(positions) < config.N_PLAYERS:
        positions.append("CM")

    names = [f"{FIRST_NAMES[i]} {LAST_NAMES[i]}" for i in range(config.N_PLAYERS)]

    rows = []
    for i in range(config.N_PLAYERS):
        age = int(np.clip(rng.normal(25, 4), 17, 36))
        # Older players and certain positions carry slightly higher baseline
        # susceptibility to reduced-availability episodes.
        pos = positions[i]
        pos_risk = {"ST": 0.15, "W": 0.15, "FB": 0.1, "CM": 0.05}.get(pos, 0.0)
        # Susceptibility is driven mostly by age (an observable feature) plus a
        # position effect and a smaller unobservable random component. This keeps
        # a genuine, learnable static signal without making age deterministic.
        susceptibility = float(
            np.clip((age - 24) * 0.06 + pos_risk + rng.normal(0.0, 1.0) * 0.22, -0.8, 1.6)
        )
        rows.append(
            {
                "player_id": f"P{i + 1:02d}",
                "player_name": names[i],
                "age": age,
                "position": pos,
                "dominant_leg": rng.choice(["Right", "Left"], p=[0.78, 0.22]),
                "height_cm": int(np.clip(rng.normal(183 if pos in ("GK", "CB") else 179, 6), 168, 200)),
                "weight_kg": int(np.clip(rng.normal(78, 6), 62, 95)),
                # latent, not exported to the "measured" tables but kept for the
                # simulation and for an honest data dictionary.
                "_susceptibility": susceptibility,
                "_robustness": float(np.clip(rng.normal(1.0, 0.15), 0.6, 1.4)),
                "_load_capacity": float(np.clip(rng.normal(1.0, 0.12), 0.7, 1.3)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Match & training calendar
# --------------------------------------------------------------------------- #
def _build_calendar(rng: np.random.Generator) -> pd.DataFrame:
    """One row per calendar day for the whole squad timeline.

    Assigns a team-level 'day_type' (match / training / recovery / off) and a
    matchday code (MD-x). Congested periods have two matches per week.
    """
    dates = pd.date_range(config.SEASON_START, config.SEASON_END, freq="D")
    df = pd.DataFrame({"date": dates})
    df["dow"] = df["date"].dt.dayofweek  # 0 = Monday

    season_day = (df["date"] - df["date"].min()).dt.days
    total_days = season_day.max()
    df["preseason"] = season_day < 28  # first 4 weeks are pre-season

    # Congested windows (e.g. cup + league midweek fixtures): a few blocks.
    frac = season_day / total_days
    congested = ((frac > 0.25) & (frac < 0.38)) | ((frac > 0.60) & (frac < 0.72))
    df["congested"] = congested & ~df["preseason"]

    # Match assignment: Saturdays always a match (in-season). Congested weeks add
    # a Tuesday/Wednesday midweek match.
    is_match = pd.Series(False, index=df.index)
    is_match |= (~df["preseason"]) & (df["dow"] == 5)                    # Saturday
    is_match |= df["congested"] & (df["dow"] == 2)                       # midweek
    # Pre-season friendlies once a week
    is_match |= df["preseason"] & (df["dow"] == 6) & (season_day > 7)
    df["is_match"] = is_match

    # Days since / until nearest match, for MD coding.
    match_days = df.loc[df["is_match"], "date"].to_numpy()
    day_type = []
    md_code = []
    for _, row in df.iterrows():
        d = row["date"].to_datetime64()
        if row["is_match"]:
            day_type.append("match")
            md_code.append("MD")
            continue
        future = match_days[match_days > d]
        past = match_days[match_days < d]
        days_to_next = (future.min() - d) / np.timedelta64(1, "D") if len(future) else 99
        days_from_prev = (d - past.max()) / np.timedelta64(1, "D") if len(past) else 99
        if days_from_prev == 1:
            day_type.append("recovery")
            md_code.append("MD+1")
        elif days_to_next <= 4 and not row["preseason"]:
            md_code.append(f"MD-{int(days_to_next)}")
            day_type.append("training")
        elif days_to_next == 99:
            day_type.append("off")
            md_code.append("OFF")
        else:
            day_type.append("training")
            md_code.append("MD-x")
    df["day_type"] = day_type
    df["md_code"] = md_code
    # Rare full rest days (Sundays after Saturday match in non-congested weeks).
    df.loc[(df["day_type"] == "recovery") & (df["dow"] == 6), "day_type"] = "off"
    return df


# --------------------------------------------------------------------------- #
# Per-day load templates
# --------------------------------------------------------------------------- #
# Relative session intensity by matchday code (fraction of a match). Classic
# tactical-periodisation shape: hard in the middle of the week, taper toward MD.
_MD_INTENSITY = {
    "match": 1.00,
    "MD+1": 0.25,   # recovery / regen
    "MD-4": 0.75,
    "MD-3": 0.90,   # heaviest training day
    "MD-2": 0.60,
    "MD-1": 0.35,   # activation / taper
    "MD-x": 0.70,
    "OFF": 0.0,
    "preseason": 0.85,
}


def _position_load_profile(pos: str) -> dict:
    """Position-specific external-load scaling (mean multipliers)."""
    base = {"dist": 1.0, "hsr": 1.0, "sprint": 1.0, "accel": 1.0}
    profiles = {
        "GK": {"dist": 0.45, "hsr": 0.2, "sprint": 0.15, "accel": 0.5},
        "CB": {"dist": 0.9, "hsr": 0.8, "sprint": 0.75, "accel": 0.95},
        "FB": {"dist": 1.1, "hsr": 1.25, "sprint": 1.2, "accel": 1.15},
        "DM": {"dist": 1.05, "hsr": 0.9, "sprint": 0.8, "accel": 1.0},
        "CM": {"dist": 1.15, "hsr": 1.05, "sprint": 0.95, "accel": 1.05},
        "AM": {"dist": 1.05, "hsr": 1.1, "sprint": 1.1, "accel": 1.1},
        "W": {"dist": 1.05, "hsr": 1.35, "sprint": 1.4, "accel": 1.2},
        "ST": {"dist": 0.95, "hsr": 1.2, "sprint": 1.25, "accel": 1.15},
    }
    return profiles.get(pos, base)


def generate_season(seed: int = RNG_SEED) -> dict[str, pd.DataFrame]:
    """Run the full simulation and return normalized tables.

    Returns a dict with keys: players, sessions, matches, gps_data, wellness,
    physical_tests, availability (episodes), daily (denormalized master).
    """
    rng = np.random.default_rng(seed)
    players = _build_players(rng)
    cal = _build_calendar(rng)

    records = []           # denormalized daily rows
    availability_eps = []  # episode-level availability log

    for _, p in players.iterrows():
        state = _simulate_player(rng, p, cal)
        records.extend(state["daily"])
        availability_eps.extend(state["episodes"])

    daily = pd.DataFrame(records)
    daily = daily.sort_values(["player_id", "date"]).reset_index(drop=True)

    # Inject a small amount of realistic missingness AFTER simulation so the
    # data-quality page has something to detect (e.g. a wellness form not filled,
    # a GPS unit not worn). Availability itself is never missing.
    daily = _inject_missingness(daily, rng)

    tables = _normalize(players, cal, daily, availability_eps)
    return tables


# --------------------------------------------------------------------------- #
# Player-level simulation
# --------------------------------------------------------------------------- #
def _simulate_player(rng, p, cal) -> dict:
    prof = _position_load_profile(p["position"])
    cap = p["_load_capacity"]

    # Latent running states
    acute = 400.0          # EWMA of daily player_load (~7d)
    chronic = 400.0        # EWMA of daily player_load (~28d)
    fatigue_state = 0.0    # accumulates, decays
    nm_fatigue = 0.0       # neuromuscular fatigue -> depresses CMJ
    sleep_debt = 0.0

    # Individual measurement baselines (for wellness & CMJ), used to make each
    # player their own reference.
    cmj_baseline = float(np.clip(rng.normal(36, 3.5), 26, 46))  # cm
    sleep_baseline = float(np.clip(rng.normal(7.8, 0.5), 6.5, 9.0))
    hr_baseline = int(np.clip(rng.normal(52, 4), 44, 62))

    availability = "available"  # available / modified / unavailable
    episode_days_left = 0
    episode_type = None
    episode_start = None
    minutes_reservoir = rng.uniform(0.55, 1.0)  # rotation tendency (starter vs squad)

    daily_rows = []
    episodes = []
    days_since_match = 7

    for _, day in cal.iterrows():
        date = day["date"]
        md = day["md_code"] if not day["preseason"] else "preseason"
        day_type = day["day_type"]

        # ----- availability episode bookkeeping -----
        modified_today = False
        unavailable_today = False
        if episode_days_left > 0:
            if episode_type == "unavailable":
                unavailable_today = True
            else:
                modified_today = True
            episode_days_left -= 1
            if episode_days_left == 0:
                episodes.append(
                    {
                        "player_id": p["player_id"],
                        "start_date": episode_start,
                        "end_date": date,
                        "type": episode_type,
                        "duration_days": (date - episode_start).days + 1,
                    }
                )
                episode_type = None

        # ----- decide today's realized load -----
        intensity = _MD_INTENSITY.get(md, 0.7)
        played_match = False
        minutes = 0
        match_intensity = np.nan

        if unavailable_today:
            intensity = 0.0
        elif modified_today:
            intensity *= rng.uniform(0.35, 0.6)  # reduced/modified session

        if day_type == "match" and not unavailable_today:
            # Selection: reservoir + noise decides if he plays and how long.
            if rng.random() < minutes_reservoir and not modified_today:
                played_match = True
                minutes = int(np.clip(rng.normal(80, 20), 15, 96))
                if p["position"] == "GK" and minutes_reservoir > 0.8:
                    minutes = 90 if rng.random() < 0.9 else minutes
                match_intensity = float(np.clip(rng.normal(0.85, 0.08), 0.5, 1.0))
                days_since_match = 0
            else:
                intensity *= 0.4  # non-selected -> compensatory training

        # External load construction from intensity
        if intensity <= 0:
            gps = _zero_gps()
            srpe = 0.0
            hr_mean = hr_baseline + int(rng.normal(0, 2))
        else:
            noise = lambda s=0.12: max(0.0, rng.normal(1.0, s))
            base_dist = 9500 if played_match else 6200
            total_distance = base_dist * intensity * prof["dist"] * cap * noise()
            hsr = (620 if played_match else 380) * intensity * prof["hsr"] * cap * noise(0.18)
            sprint = (210 if played_match else 120) * intensity * prof["sprint"] * cap * noise(0.22)
            accel = (42 if played_match else 30) * intensity * prof["accel"] * noise(0.18)
            decel = (40 if played_match else 28) * intensity * prof["accel"] * noise(0.18)
            player_load = total_distance * 0.055 * noise(0.08) + hsr * 0.15
            metabolic = player_load * rng.uniform(0.9, 1.1)
            # Internal load
            rpe = np.clip(intensity * 8.2 * (1 + 0.15 * fatigue_state) * noise(0.1), 1, 10)
            duration = minutes if played_match else int(np.clip(rng.normal(80, 15), 45, 110))
            srpe = float(rpe * duration)
            hr_mean = int(hr_baseline + intensity * 95 * noise(0.05))
            gps = {
                "total_distance": round(total_distance, 1),
                "high_speed_running": round(hsr, 1),
                "sprint_distance": round(sprint, 1),
                "accelerations": int(round(accel)),
                "decelerations": int(round(decel)),
                "player_load": round(player_load, 1),
                "metabolic_load": round(metabolic, 1),
            }

        pl_today = gps["player_load"]

        # ----- update latent load states (EWMA) -----
        acute = 0.80 * acute + 0.20 * pl_today
        chronic = 0.965 * chronic + 0.035 * pl_today
        acwr = acute / chronic if chronic > 1 else 1.0

        # ----- fatigue dynamics -----
        # accumulate with today's load relative to capacity; decay with rest.
        fatigue_state = np.clip(
            fatigue_state * 0.80 + (pl_today / 650.0) * 0.75 - 0.05, 0, 5
        )
        nm_fatigue = np.clip(
            nm_fatigue * 0.84 + (hsr_component := gps["high_speed_running"] / 460.0) * 0.6 - 0.04,
            0, 5,
        )

        # ----- wellness (morning report), individual baseline + fatigue signal -----
        sleep_dur = float(np.clip(sleep_baseline + rng.normal(0, 0.6) - 0.15 * (md in ("MD+1",)) , 4.5, 10))
        sleep_debt = np.clip(sleep_debt * 0.7 + (sleep_baseline - sleep_dur) * 0.5, -2, 4)
        sleep_quality = int(np.clip(round(5 - 0.6 * sleep_debt + rng.normal(0, 0.7)), 1, 7))
        fatigue_rep = int(np.clip(round(3.6 + 0.9 * fatigue_state + 0.4 * sleep_debt + rng.normal(0, 0.65)), 1, 7))
        soreness = int(np.clip(round(3.3 + 0.95 * nm_fatigue + 0.5 * (days_since_match <= 2) + rng.normal(0, 0.65)), 1, 7))
        stress = int(np.clip(round(3.5 + 0.4 * sleep_debt + rng.normal(0, 1.0)), 1, 7))
        mood = int(np.clip(round(5 - 0.3 * fatigue_state - 0.3 * stress / 3 + rng.normal(0, 0.8)), 1, 7))

        # Wellness convention here: higher fatigue/soreness/stress = worse;
        # higher sleep_quality/mood = better. (Documented in the data dictionary.)

        # ----- periodic physical tests -----
        cmj = eccentric = concentric = peak = ct = np.nan
        sprint10 = sprint30 = np.nan
        # CMJ twice per week (MD-4 and MD+1), depressed by neuromuscular fatigue.
        if (md in ("MD-4", "MD+1") or (day["preseason"] and day["dow"] in (0, 3))) and not unavailable_today:
            cmj = round(cmj_baseline * (1 - 0.03 * nm_fatigue) * max(0.0, rng.normal(1.0, 0.02)), 1)
            ct = round(np.clip(rng.normal(0.25, 0.02) + 0.01 * nm_fatigue, 0.18, 0.35), 3)
            peak = round(np.clip(rng.normal(2600, 220) * (1 - 0.02 * nm_fatigue), 1800, 3400), 0)
            eccentric = round(peak * rng.uniform(0.55, 0.65), 0)
            concentric = round(peak * rng.uniform(0.62, 0.72), 0)
        # Sprint tests ~ every 3-4 weeks
        if md == "MD-3" and rng.random() < 0.18 and not unavailable_today:
            sprint10 = round(np.clip(rng.normal(1.72, 0.05) + 0.01 * nm_fatigue, 1.55, 2.0), 3)
            sprint30 = round(np.clip(rng.normal(4.05, 0.12) + 0.02 * nm_fatigue, 3.7, 4.6), 3)

        if not played_match:
            days_since_match += 1

        # ----- hazard: onset of a NEW availability episode -----
        # Only fire when currently available. Uses latent+observable drivers.
        new_event = False
        if availability == "available" and episode_days_left == 0 and not day["preseason"]:
            acwr_excess = max(0.0, acwr - 1.25)
            logit = (
                -5.85
                + 3.0 * acwr_excess
                + 0.72 * fatigue_state
                + 0.66 * nm_fatigue
                + 0.46 * max(0.0, sleep_debt)
                + 0.26 * (soreness - 3.5)
                + 0.022 * max(0, _minutes_last_days(daily_rows, 7) - 160)
                + 0.55 * p["_susceptibility"]
                - 1.6 * (p["_robustness"] - 1.0)
                + rng.normal(0, 0.38)  # irreducible noise (kept, but no longer dominant)
            )
            prob = 1 / (1 + np.exp(-logit))
            if rng.random() < prob:
                new_event = True

        if new_event:
            # Severity split: most events are short modified-training; a minority
            # are longer unavailability (soft-tissue injuries etc.).
            if rng.random() < 0.62:
                episode_type = "modified"
                dur = int(np.clip(rng.normal(2, 1), 1, 5))
            else:
                episode_type = "unavailable"
                dur = int(np.clip(rng.lognormal(1.8, 0.55), 3, 24))
            episode_days_left = dur
            episode_start = date
            # Reset acute load if going unavailable
            if episode_type == "unavailable":
                acute *= 0.6

        # availability label for TODAY (already-decremented above)
        if unavailable_today:
            avail_label = "unavailable"
        elif modified_today or (new_event and episode_type == "modified"):
            avail_label = "modified_training"
        elif new_event and episode_type == "unavailable":
            avail_label = "unavailable"
        else:
            avail_label = "available"

        if avail_label != "available":
            availability = avail_label
        else:
            availability = "available"

        row = {
            "player_id": p["player_id"],
            "player_name": p["player_name"],
            "age": int(p["age"]),
            "position": p["position"],
            "date": date,
            "day_type": day_type,
            "md_code": day["md_code"],
            "preseason": bool(day["preseason"]),
            "congested": bool(day["congested"]),
            # external load
            **gps,
            # internal load
            "session_rpe": round(srpe, 0),
            "heart_rate_mean": int(hr_mean),
            "hr_zone_high_min": round(max(0.0, (intensity - 0.5) * 40) * (rng.normal(1, .1)), 1) if intensity > 0 else 0.0,
            # wellness
            "sleep_duration": round(sleep_dur, 2),
            "sleep_quality": sleep_quality,
            "fatigue": fatigue_rep,
            "muscle_soreness": soreness,
            "stress": stress,
            "mood": mood,
            # physical tests
            "cmj_height": cmj,
            "cmj_contraction_time": ct,
            "eccentric_force": eccentric,
            "concentric_force": concentric,
            "peak_force": peak,
            "sprint_10m": sprint10,
            "sprint_30m": sprint30,
            # football
            "minutes_played": minutes,
            "played_match": bool(played_match),
            "match_intensity": round(match_intensity, 3) if not np.isnan(match_intensity) else np.nan,
            "days_since_last_match": int(days_since_match),
            # availability (label / target source) — never fed to the model as a feature
            "availability_status": avail_label,
            "available": int(avail_label == "available"),
            "modified_training": int(avail_label == "modified_training"),
            "unavailable": int(avail_label == "unavailable"),
            # latent columns (diagnostic only, dropped from measured tables)
            "_acwr_true": round(acwr, 3),
            "_fatigue_state": round(float(fatigue_state), 3),
        }
        daily_rows.append(row)

    return {"daily": daily_rows, "episodes": episodes}


def _minutes_last_days(rows, n):
    if not rows:
        return 0
    return sum(r["minutes_played"] for r in rows[-n:])


def _zero_gps():
    return {
        "total_distance": 0.0, "high_speed_running": 0.0, "sprint_distance": 0.0,
        "accelerations": 0, "decelerations": 0, "player_load": 0.0, "metabolic_load": 0.0,
    }


# --------------------------------------------------------------------------- #
# Missingness & normalization
# --------------------------------------------------------------------------- #
def _inject_missingness(daily: pd.DataFrame, rng) -> pd.DataFrame:
    daily = daily.copy()
    n = len(daily)
    # ~3% of wellness forms not completed
    miss_well = rng.random(n) < 0.03
    for col in ["sleep_duration", "sleep_quality", "fatigue", "muscle_soreness", "stress", "mood"]:
        daily.loc[miss_well & (rng.random(n) < 0.7), col] = np.nan
    # ~2% GPS unit not worn on training days
    miss_gps = (rng.random(n) < 0.02) & (daily["day_type"].isin(["training", "recovery"]))
    for col in ["total_distance", "high_speed_running", "sprint_distance", "player_load"]:
        daily.loc[miss_gps, col] = np.nan
    # a few duplicate rows (double import) to be caught by the pipeline
    dups = daily.sample(6, random_state=RNG_SEED)
    daily = pd.concat([daily, dups], ignore_index=True)
    # a couple of physically impossible outliers (sensor glitch)
    idx = daily.sample(4, random_state=RNG_SEED + 1).index
    daily.loc[idx, "total_distance"] = daily.loc[idx, "total_distance"] * 5 + 40000
    return daily


def _normalize(players, cal, daily, episodes) -> dict[str, pd.DataFrame]:
    # players table (drop latent underscore cols from the public schema but keep
    # a documented copy for the data dictionary)
    players_public = players.drop(columns=[c for c in players.columns if c.startswith("_")])

    # matches table (team-level fixtures)
    matches = cal.loc[cal["is_match"], ["date", "md_code", "congested", "preseason"]].copy()
    matches = matches.reset_index(drop=True)
    matches.insert(0, "match_id", [f"M{i + 1:03d}" for i in range(len(matches))])
    matches["opponent"] = [f"Opponent {i + 1}" for i in range(len(matches))]
    matches["competition"] = np.where(matches["preseason"], "Friendly",
                              np.where(matches["congested"], "Cup/Midweek", "League"))
    matches["is_home"] = (np.arange(len(matches)) % 2 == 0)

    # sessions table (session-level metadata)
    sessions = daily[["player_id", "date", "day_type", "md_code", "session_rpe",
                      "minutes_played", "played_match"]].copy()
    sessions.insert(0, "session_id", range(1, len(sessions) + 1))

    gps_cols = ["player_id", "date", "total_distance", "high_speed_running",
                "sprint_distance", "accelerations", "decelerations",
                "player_load", "metabolic_load"]
    gps_data = daily[gps_cols].copy()

    wellness = daily[["player_id", "date", "sleep_duration", "sleep_quality",
                      "fatigue", "muscle_soreness", "stress", "mood"]].copy()

    physical = daily[["player_id", "date", "cmj_height", "cmj_contraction_time",
                      "eccentric_force", "concentric_force", "peak_force",
                      "sprint_10m", "sprint_30m"]].copy()
    physical = physical.dropna(subset=["cmj_height", "sprint_10m"], how="all")

    availability = daily[["player_id", "date", "availability_status", "available",
                          "modified_training", "unavailable", "minutes_played",
                          "days_since_last_match"]].copy()

    episodes_df = pd.DataFrame(episodes) if episodes else pd.DataFrame(
        columns=["player_id", "start_date", "end_date", "type", "duration_days"])

    return {
        "players": players_public,
        "players_full": players,
        "matches": matches,
        "sessions": sessions,
        "gps_data": gps_data,
        "wellness": wellness,
        "physical_tests": physical,
        "availability": availability,
        "availability_episodes": episodes_df,
        "daily": daily,   # denormalized master used by the pipeline
    }


if __name__ == "__main__":
    t = generate_season()
    for k, v in t.items():
        print(f"{k:22s} {v.shape}")
    d = t["daily"]
    print("\nAvailability distribution:")
    print(d["availability_status"].value_counts(normalize=True).round(4))
    print(f"\nUnavailable episodes: {len(t['availability_episodes'])}")
