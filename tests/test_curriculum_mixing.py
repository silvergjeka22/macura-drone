"""Fast correctness tests for the real-data-first curriculum + batch-level real/imagined mixing
(AGENT_PROMPT_curriculum.md §8). Correctness, not performance — minutes, CPU.

Covers:
  1. RealRatioSchedule: constant + default knots (start/mid/tail-clamp).
  2. build_schedule from config (default = constant paper real_ratio).
  3. sac_update_mixed: rr=1.0 → 0 model batches; rr=0.0 → 0 real; rr=0.5 → ≈50/50.
  4. Default config → realised real_pct ≈ 5% (the paper-faithful structural fix landed).
  5. MACURA rollout still κ-truncated (NOT hand-scheduled); Eq. 22 UTD scaler unchanged.
  6. Best-checkpoint save writes BOTH policy and ensemble for a model-based agent; ensemble reloads.
  7. (env-gated) one tiny smoke run per model-based algo with the curriculum ON; realised real_pct
     falls over steps as scheduled. Skipped automatically when no MuJoCo/PyBullet backend is present.

Run directly (`python3 tests/test_curriculum_mixing.py`) for a PASS/FAIL + wall-time summary, or
under pytest (`python3 -m pytest tests/test_curriculum_mixing.py`).
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from training.curriculum import RealRatioSchedule, build_schedule  # noqa: E402
from algorithms import sac as sac_mod  # noqa: E402
from models import ensemble as ens  # noqa: E402

CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "configs", "macura_drone.yaml")
OBS_DIM, ACT_DIM = 13, 4


def _load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def _tiny_sac_cfg():
    return {"gamma": 0.99, "tau": 0.005, "alpha": "auto", "actor_lr": 3e-4, "critic_lr": 3e-4,
            "hidden_size": 32, "batch_size": 32, "target_update_interval": 1,
            "gradient_steps_max": 8, "real_ratio": 0.05, "buffer_size": 2000}


def _tiny_ens_cfg():
    return {"num_members": 3, "hidden_size": 32, "num_layers": 3, "activation": "silu",
            "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 64,
            "train_epochs_per_round": 2, "logvar_bounds": [-10.0, 0.5],
            "deterministic": False, "propagation": "random_member"}


def _fill(buf, n, batch_dim=ACT_DIM, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        o = rng.normal(size=OBS_DIM).astype(np.float32)
        a = rng.uniform(-1, 1, size=batch_dim).astype(np.float32)
        no = o + 0.01 * rng.normal(size=OBS_DIM).astype(np.float32)
        sac_mod.add_to_buffer(buf, o, a, no, float(rng.normal()), False)


# ── 1. RealRatioSchedule ──────────────────────────────────────────────────────────
def test_schedule_constant():
    s = RealRatioSchedule.constant(0.05)
    for x in (-100, 0, 7, 12345, 10 ** 9):
        assert abs(s(x) - 0.05) < 1e-12


def test_schedule_default_knots():
    s = RealRatioSchedule([(0, 0.90), (8000, 0.90), (16000, 0.20), (24000, 0.05)])
    assert abs(s(0) - 0.90) < 1e-9           # start = ~90% real
    assert abs(s(8000) - 0.90) < 1e-9        # plateau
    assert abs(s(16000) - 0.20) < 1e-9       # mid knot
    assert abs(s(12000) - 0.55) < 1e-9       # interpolated halfway
    assert abs(s(24000) - 0.05) < 1e-9       # steady state
    assert abs(s(50000) - 0.05) < 1e-9       # clamped beyond range
    assert abs(s(-10) - 0.90) < 1e-9         # clamped before range


# ── 2. build_schedule from config ──────────────────────────────────────────────────
def test_build_schedule_default_is_constant():
    cfg = _load_cfg()
    assert cfg["curriculum"]["enabled"] is False, "curriculum must default OFF"
    s = build_schedule(cfg)
    rr = float(cfg["sac"]["real_ratio"])
    for x in (0, 5000, 30000):
        assert abs(s(x) - rr) < 1e-12, "disabled curriculum must be constant sac.real_ratio"


def test_build_schedule_enabled_ramps():
    cfg = _load_cfg()
    cfg["curriculum"]["enabled"] = True
    s = build_schedule(cfg)
    assert s(0) > s(30000), "enabled curriculum must ramp DOWN from mostly-real to steady state"
    assert abs(s(0) - 0.90) < 1e-9


# ── 3 & 4. batch-level mixing split ────────────────────────────────────────────────
def _mix_split(rr, num_updates, seed=0):
    agent = sac_mod.build_sac(OBS_DIM, ACT_DIM, _tiny_sac_cfg(), "cpu", seed)
    model_rb = agent.replay_buffer
    real_rb = sac_mod.build_replay_buffer(agent, 2000)
    _fill(real_rb, 100, seed=seed)
    _fill(model_rb, 100, seed=seed + 1)
    rng = np.random.default_rng(seed)
    info = sac_mod.sac_update_mixed(agent, real_rb, model_rb, num_updates, 32, rr, rng)
    return info


def test_mixing_all_real():
    info = _mix_split(1.0, 40)
    assert info["n_model"] == 0 and info["n_real"] == 40


def test_mixing_all_model():
    info = _mix_split(0.0, 40)
    assert info["n_real"] == 0 and info["n_model"] == 40


def test_mixing_half():
    info = _mix_split(0.5, 600, seed=3)
    frac = info["n_real"] / info["updates"]
    assert 0.42 < frac < 0.58, f"expected ≈50/50, got {frac:.3f}"


def test_mixing_paper_default_5pct():
    cfg = _load_cfg()
    rr = float(cfg["sac"]["real_ratio"])           # 0.05 paper default
    info = _mix_split(rr, 1000, seed=7)
    assert abs(info["real_pct"] - 100.0 * rr) < 3.0, \
        f"realised real_pct {info['real_pct']:.1f}% should be ≈ {100 * rr:.0f}%"


# ── 5. MACURA rollout unchanged (κ-truncated, not hand-scheduled) ───────────────────
def test_macura_utd_scaler_unchanged():
    from algorithms import macura as macura_mod
    # Eq. 22 scales with model-buffer fullness, NOT env-step → not a length schedule.
    assert macura_mod.gradient_steps(100, 100, 20, True) == 20
    assert macura_mod.gradient_steps(50, 100, 20, True) == 10
    assert macura_mod.gradient_steps(0, 100, 20, False) == 20    # adaptive off → g_max


def test_macura_rollout_is_kappa_truncated():
    from algorithms import macura as macura_mod
    from envs import drone_env
    cfg = _load_cfg()
    assert cfg["rollout"]["macura"]["t_max"] == 10, "MACURA t_max must stay at the paper default"
    agent = sac_mod.build_sac(OBS_DIM, ACT_DIM, _tiny_sac_cfg(), "cpu", 0)
    dyn = ens.build_ensemble(_tiny_ens_cfg(), OBS_DIM, ACT_DIM, "cpu")
    data = {"obs": np.random.randn(200, OBS_DIM).astype(np.float32),
            "act": np.random.uniform(-1, 1, (200, ACT_DIM)).astype(np.float32),
            "next_obs": np.random.randn(200, OBS_DIM).astype(np.float32)}
    ens.train_ensemble(dyn, data, _tiny_ens_cfg())
    reward_fn = drone_env.known_reward_fn(cfg["env"])
    done_fn = drone_env.termination_fn(cfg["env"])
    start = np.random.randn(64, OBS_DIM).astype(np.float32)
    _trans, diag = macura_mod.macura_rollout(dyn, agent, start, reward_fn, done_fn, {}, cfg)
    assert "kappa" in diag and "mean_rollout_length" in diag
    assert diag["mean_rollout_length"] <= cfg["rollout"]["macura"]["t_max"]


# ── 6. best-checkpoint save persists policy + ensemble ──────────────────────────────
def test_save_best_writes_policy_and_ensemble():
    from training import train as train_mod
    agent = sac_mod.build_sac(OBS_DIM, ACT_DIM, _tiny_sac_cfg(), "cpu", 0)
    dyn = ens.build_ensemble(_tiny_ens_cfg(), OBS_DIM, ACT_DIM, "cpu")
    with tempfile.TemporaryDirectory() as d:
        path = train_mod._save_best(agent, dyn, d, "macura", 0)
        assert path and os.path.exists(path), "policy .zip missing"
        ens_path = os.path.join(d, "checkpoints", "macura_seed0_best_ensemble.pt")
        assert os.path.exists(ens_path), "ensemble .pt missing"
        ens.load_ensemble(dyn, ens_path)         # reload must succeed
    # SAC baseline: dynamics_model None → no ensemble file, still saves policy
    with tempfile.TemporaryDirectory() as d:
        path = train_mod._save_best(agent, None, d, "sac", 0)
        assert path and os.path.exists(path)
        assert not os.path.exists(os.path.join(d, "checkpoints", "sac_seed0_best_ensemble.pt"))


# ── 7. (env-gated) tiny smoke run per model-based algo, curriculum ON ────────────────
def _env_available(cfg):
    try:
        from envs import drone_env
        env, _, _ = drone_env.make_env(cfg["env"], seed=0)
        env.close()
        return True
    except Exception:
        return False


def test_smoke_curriculum_real_pct_falls():
    from training import train as train_mod
    cfg = _load_cfg()
    if not _env_available(cfg):
        print("  [skip] no MuJoCo/PyBullet backend — smoke run is Colab/GPU only")
        return
    # tiny, fast config with the curriculum ON and knots scaled to the smoke horizon
    cfg["experiment"]["total_env_steps"] = 500
    cfg["experiment"]["warmup_random_steps"] = 150
    cfg["experiment"]["eval_every_steps"] = 100
    cfg["experiment"]["eval_episodes"] = 1
    cfg["selection"] = {"start_step": 0, "eval_every": 100, "final_eval_episodes": 1}
    cfg["sac"]["gradient_steps_max"] = 4
    cfg["ensemble"]["train_epochs_per_round"] = 2
    cfg["rollout"]["freq_steps"] = 100
    cfg["rollout"]["num_rollouts"] = 50
    cfg["curriculum"] = {"enabled": True, "real_ratio_knots": [
        {"step": 150, "real_ratio": 0.95}, {"step": 500, "real_ratio": 0.05}]}
    with tempfile.TemporaryDirectory() as d:
        for algo in ("macura", "mbpo", "m2ac"):
            run = train_mod.train_one(algo, cfg, d, seed=0)
            tgt = [v for _, v in run["real_ratio_target"] if not np.isnan(v)]
            assert tgt, f"{algo}: no real_ratio telemetry logged"
            assert tgt[0] > tgt[-1] + 0.1, f"{algo}: real_ratio target should fall over training"


# ── runner ──────────────────────────────────────────────────────────────────────────
def _all_tests():
    g = globals()
    return [(n, g[n]) for n in sorted(g) if n.startswith("test_") and callable(g[n])]


if __name__ == "__main__":
    t_start = time.time()
    passed = failed = 0
    for name, fn in _all_tests():
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t0:.2f}s)")
            passed += 1
        except Exception as e:
            print(f"FAIL  {name}  ({time.time() - t0:.2f}s)  -> {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed in {time.time() - t_start:.1f}s")
    sys.exit(1 if failed else 0)
