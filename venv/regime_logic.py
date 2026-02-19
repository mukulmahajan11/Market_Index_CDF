#!/usr/bin/env python3
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional

@dataclass
class MoodConfig:
    # Convert probabilities -> mood
    mood_floor: float = 0.0
    mood_ceiling: float = 100.0

    # EWMA smoothing for mood to reduce noise
    ewma_alpha: float = 0.25

    # Consecutive-day persistence before switching regimes
    enter_persistence_days: int = 2   # require N days below/above enter threshold to ENTER new regime
    exit_persistence_days: int = 3    # require M days to EXIT back to better regime

    # Hysteresis thresholds (enter thresholds are stricter than exit thresholds)
    # Mood is 0–100, higher is healthier.
    # Enter Warning if mood <= 65 for N days; Exit Warning back to Normal if mood >= 72 for M days.
    warning_enter: float = 65.0
    warning_exit: float = 72.0

    # Enter Critical if mood <= 40 for N days; Exit Critical to Warning if mood >= 48 for M days.
    critical_enter: float = 40.0
    critical_exit: float = 48.0

    # Optional hard safety rules based on stress probability and confidence
    # If p_critical >= 0.55 AND confidence >= 0.65 -> immediate Critical.
    immediate_critical_p: float = 0.55
    immediate_critical_conf: float = 0.65

    # If p_stress >= 0.60 AND confidence >= 0.70 -> at least Warning (fast escalation)
    immediate_warning_stress_p: float = 0.60
    immediate_warning_conf: float = 0.70

def prob_confidence(proba: np.ndarray) -> float:
    """
    Convert a probability vector into confidence in [0,1] using normalized entropy.
    - entropy = -sum(p log p)
    - normalized by log(K)
    - confidence = 1 - normalized_entropy
    """
    p = np.clip(proba, 1e-12, 1.0)
    entropy = -np.sum(p * np.log(p))
    k = len(p)
    max_entropy = np.log(k)
    norm_entropy = entropy / max_entropy
    return float(np.clip(1.0 - norm_entropy, 0.0, 1.0))

def mood_from_proba(proba: np.ndarray, cfg: MoodConfig) -> tuple[float, float, float, float, float]:
    """
    Returns:
      mood_raw, mood, conf, p_stress, p_critical
    """
    p_normal, p_warning, p_critical = proba
    p_stress = p_warning + p_critical
    conf = prob_confidence(proba)

    mood_raw = 100.0 * (1.0 - p_stress)  # 0..100
    # Penalize uncertainty by scaling in [0.5, 1.0] (tunable)
    mood = mood_raw * (0.5 + 0.5 * conf)

    mood = float(np.clip(mood, cfg.mood_floor, cfg.mood_ceiling))
    return float(mood_raw), mood, float(conf), float(p_stress), float(p_critical)

class MarketMoodIndex:
    """
    Stateful regime engine with smoothing + hysteresis + persistence.
    Usage:
      engine = MarketMoodIndex(cfg)
      out_df = engine.run(df)  # df must include columns with probabilities
    """

    def __init__(self, cfg: Optional[MoodConfig] = None):
        self.cfg = cfg or MoodConfig()

    def _apply_regime_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        regimes = []
        current = "Normal"
        # counters for persistence
        enter_count = 0
        exit_count = 0

        for _, r in df.iterrows():
            mood = float(r["mood_ewma"])
            conf = float(r["confidence"])
            p_stress = float(r["p_stress"])
            p_critical = float(r["p_critical"])

            # Immediate escalation rules (business overrides)
            if (p_critical >= cfg.immediate_critical_p) and (conf >= cfg.immediate_critical_conf):
                current = "Critical"
                enter_count = 0
                exit_count = 0
                regimes.append(current)
                continue

            if (p_stress >= cfg.immediate_warning_stress_p) and (conf >= cfg.immediate_warning_conf):
                if current == "Normal":
                    current = "Warning"
                # If already Critical keep it unless mood/hysteresis later exits
                regimes.append(current)
                continue

            # Hysteresis + persistence switching
            if current == "Normal":
                # Check entering Warning
                if mood <= cfg.warning_enter:
                    enter_count += 1
                    if enter_count >= cfg.enter_persistence_days:
                        current = "Warning"
                        enter_count = 0
                        exit_count = 0
                else:
                    enter_count = 0

            elif current == "Warning":
                # Escalate to Critical
                if mood <= cfg.critical_enter:
                    enter_count += 1
                    if enter_count >= cfg.enter_persistence_days:
                        current = "Critical"
                        enter_count = 0
                        exit_count = 0
                # Exit back to Normal
                elif mood >= cfg.warning_exit:
                    exit_count += 1
                    if exit_count >= cfg.exit_persistence_days:
                        current = "Normal"
                        exit_count = 0
                        enter_count = 0
                else:
                    enter_count = 0
                    exit_count = 0

            elif current == "Critical":
                # Exit Critical to Warning (requires higher exit threshold)
                if mood >= cfg.critical_exit:
                    exit_count += 1
                    if exit_count >= cfg.exit_persistence_days:
                        current = "Warning"
                        exit_count = 0
                        enter_count = 0
                else:
                    exit_count = 0

            regimes.append(current)

        df["regime"] = regimes
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects columns: date, asset_id, p_normal, p_warning, p_critical
        Returns df with mood_raw, mood, confidence, mood_ewma, regime.
        """
        cfg = self.cfg
        out = df.copy().sort_values(["asset_id","date"]).reset_index(drop=True)

        mood_raw_list = []
        mood_list = []
        conf_list = []
        p_stress_list = []
        p_critical_list = []

        for _, r in out.iterrows():
            proba = np.array([r["p_normal"], r["p_warning"], r["p_critical"]], dtype=float)
            mood_raw, mood, conf, p_stress, p_critical = mood_from_proba(proba, cfg)
            mood_raw_list.append(mood_raw)
            mood_list.append(mood)
            conf_list.append(conf)
            p_stress_list.append(p_stress)
            p_critical_list.append(p_critical)

        out["mood_raw"] = mood_raw_list
        out["mood"] = mood_list
        out["confidence"] = conf_list
        out["p_stress"] = p_stress_list
        out["p_critical"] = p_critical_list

        # EWMA smoothing per asset
        out["mood_ewma"] = (
            out.groupby("asset_id")["mood"]
               .transform(lambda s: s.ewm(alpha=cfg.ewma_alpha, adjust=False).mean())
        )

        # Apply regime rules per asset
        out = out.groupby("asset_id", group_keys=False).apply(self._apply_regime_rules)
        return out
