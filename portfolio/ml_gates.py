"""ML drift gates for the daily agent — block new risk when models drift."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from portfolio.decisions import Action, TickerDecision

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BASELINE = _ROOT / "projection" / "ml_model" / "saved_models" / "feature_baseline.json"
_DEFAULT_CALIB_LOG = _ROOT / "projection" / "ml_model" / "saved_models" / "calibration_log.jsonl"

# Cross-sectional features collected from each daily scan (one value per ticker).
_GATE_FEATURES = ("ml_score", "p_up_20d", "atr_pct", "vol_60d_annual")


@dataclass
class MlGateResult:
    block_new_entries: bool
    reasons: list[str] = field(default_factory=list)
    feature_drift: dict[str, Any] | None = None
    calibration_alerts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_new_entries": self.block_new_entries,
            "reasons": list(self.reasons),
            "feature_drift": self.feature_drift,
            "calibration_alerts": self.calibration_alerts,
        }


def _gate_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("ml_gates") or {})


def _collect_features(analyses: list[dict[str, Any]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {k: [] for k in _GATE_FEATURES}
    for a in analyses:
        if not a.get("ok"):
            continue
        for key in _GATE_FEATURES:
            val = a.get(key)
            if isinstance(val, (int, float)) and val == val:
                out[key].append(float(val))
    return {k: v for k, v in out.items() if len(v) >= 5}


def evaluate_ml_gates(
    cfg: dict[str, Any],
    analyses: list[dict[str, Any]],
) -> MlGateResult:
    """Evaluate feature + calibration drift; may recommend blocking new entries."""
    gates = _gate_cfg(cfg)
    if not gates.get("enabled", False):
        return MlGateResult(block_new_entries=False)

    reasons: list[str] = []
    feature_report: dict[str, Any] | None = None
    cal_alerts: list[dict[str, Any]] = []

    severity_threshold = str(gates.get("drift_severity_threshold", "high")).lower()
    block_feature = bool(gates.get("block_entries_on_high_feature_drift", True))
    block_cal = bool(gates.get("block_entries_on_calibration_drift", True))

    baseline_path = Path(gates.get("feature_baseline_path", _DEFAULT_BASELINE))
    if block_feature:
        if baseline_path.is_file():
            from projection.ml_model.drift_monitor import DriftMonitor

            monitor = DriftMonitor.load(baseline_path)
            features = _collect_features(analyses)
            if features:
                report = monitor.compute(features)
                feature_report = report.to_json()
                sev = report.severity
                breach = sev == "high" or (
                    severity_threshold == "moderate" and sev in ("moderate", "high")
                )
                if breach:
                    reasons.append(
                        f"Feature drift {sev} (threshold={severity_threshold})"
                    )
        elif gates.get("warn_if_missing_baseline", True):
            reasons.append(f"Feature baseline missing ({baseline_path}); drift not checked.")

    cal_path = Path(gates.get("calibration_log_path", _DEFAULT_CALIB_LOG))
    if block_cal and cal_path.is_file():
        from projection.ml_model.walk_forward import detect_calibration_drift

        rolling_n = int(gates.get("calibration_rolling_n", 5))
        z_threshold = float(gates.get("calibration_z_threshold", 2.0))
        alerts = detect_calibration_drift(
            cal_path, rolling_n=rolling_n, z_threshold=z_threshold,
        )
        if alerts:
            latest = alerts[-1]
            cal_alerts.append(latest.to_json() if hasattr(latest, "to_json") else dict(latest))
            if latest.breach:
                reasons.append(
                    f"Calibration drift on {latest.metric}: z={latest.z_score:.2f}"
                )

    hard_reasons = [r for r in reasons if "missing" not in r.lower()]
    block = len(hard_reasons) > 0

    return MlGateResult(
        block_new_entries=block,
        reasons=reasons,
        feature_drift=feature_report,
        calibration_alerts=cal_alerts,
    )


def apply_ml_entry_gates(
    decisions: list[TickerDecision],
    gate: MlGateResult,
) -> tuple[list[TickerDecision], list[dict[str, Any]]]:
    """Downgrade ENTER_* to NO_TRADE when gates are active (exits unchanged)."""
    if not gate.block_new_entries:
        return decisions, []

    blocked: list[dict[str, Any]] = []
    reason = "; ".join(gate.reasons) or "ML gate active"
    out: list[TickerDecision] = []
    for d in decisions:
        if d.action in (Action.ENTER_LONG, Action.ENTER_SHORT):
            blocked.append({"ticker": d.ticker, "action": d.action.value, "reason": reason})
            out.append(
                TickerDecision(
                    ticker=d.ticker,
                    action=Action.NO_TRADE,
                    reason=f"ML gate: {reason}",
                    ml_score=d.ml_score,
                    quintile=d.quintile,
                    p_up_20d=d.p_up_20d,
                    price=d.price,
                )
            )
        else:
            out.append(d)
    return out, blocked
