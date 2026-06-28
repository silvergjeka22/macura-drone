"""Fast correctness tests for the two-buffer + fixed batch-level real/imagined mixing.

The four algorithms share everything (SAC backbone, ensemble, pink-noise exploration, eval seeds,
fixed `real_ratio`) EXCEPT their rollout strategy. These tests check the shared mixing machinery and
that the algorithm rules are untouched. Correctness, not performance — seconds on CPU.

Covers:
  1. sac_update_mixed: rr=1.0 → 0 model batches; rr=0.0 → 0 real; rr=0.5 → ≈50/50 (seeded RNG).
  2. Paper-faithful default: config real_ratio → realised real_pct ≈ that value.
  3. Config sanity: sac.real_ratio present, selection block present, no curriculum left over.
  4. MACURA unchanged: Eq. 22 UTD scaler + κ-truncated rollout (not hand-scheduled).
  5. Best-checkpoint save writes BOTH policy and ensemble for a model-based agent; ensemble reloads.
  6. (env-gated) tiny smoke run per model-based algo; realised real_pct ≈ the fixed target. Skipped
     automatically when no MuJoCo/PyBullet backend is present.

Run directly (`python3 tests/test_mixing.py`) for a PASS/FAIL + wall-time summary, or under pytest.
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

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


def _fill(buf, n, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        o = rng.normal(size=OBS_DIM).astype(np.float32)
        a = rng.uniform(-1, 1, size=ACT_DIM).astype(np.float32)
        no = o + 0.01 * rng.normal(size=OBS_DIM).astype(np.float32)
        sac_mod.add_to_buffer(buf, o, a, no, float(rng.normal()), False)


def _mix_split(rr, num_updates, seed=0):
    agent = sac_mod.build_sac(OBS_DIM, ACT_DIM, _tiny_sac_cfg(), "cpu", seed)
    model_rb = agent.replay_buffer
    real_rb = sac_mod.build_replay_buffer(agent, 2000)
    _fill(real_rb, 100, seed=seed)
    _fill(model_rb, 100, seed=seed + 1)
    rng = np.random.default_rng(seed)
    return sac_mod.sac_update_mixed(agent, real_rb, model_rb, num_updates, 32, rr, rng)


# ── 1. batch-level mixing split ─────────────────────────────────────────────────────
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


# ── 2. paper-faithful fixed default ─────────────────────────────────────────────────
def test_mixing_paper_default():
    rr = float(_load_cfg()["sac"]["real_ratio"])
    info = _mix_split(rr, 1000, seed=7)
    assert abs(info["real_pct"] - 100.0 * rr) < 3.0, \
        f"realised real_pct {info['real_pct']:.1f}% should be ≈ {100 * rr:.0f}%"


# ── 3. config sanity ────────────────────────────────────────────────────────────────
def test_config_is_fixed_no_curriculum():
    cfg = _load_cfg()
    assert "real_ratio" in cfg["sac"], "sac.real_ratio must be set (fixed mixing)"
    assert 0.0 <= cfg["sac"]["real_ratio"] <= 1.0
    assert "selection" in cfg, "selection block (start_step/eval_every/final_eval_episodes) required"
    assert "curriculum" not in cfg, "curriculum was removed — config must not reference it"


# ── 4. MACURA rule unchanged (κ-truncated, not hand-scheduled) ───────────────────────
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


# ── 5. best-checkpoint save persists policy + ensemble ──────────────────────────────
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


# ── 6. (env-gated) tiny smoke run per model-based algo ───────────────────────────────
def _env_available(cfg):
    try:
        from envs import drone_env
        env, _, _ = drone_env.make_env(cfg["env"], seed=0)
        env.close()
        return True
    except Exception:
        return False


def test_smoke_fixed_real_pct_near_target():
    from training import train as train_mod
    cfg = _load_cfg()
    if not _env_available(cfg):
        print("  [skip] no MuJoCo/PyBullet backend — smoke run is Colab/GPU only")
        return
    cfg["experiment"]["total_env_steps"] = 600
    cfg["experiment"]["warmup_random_steps"] = 150
    cfg["experiment"]["eval_episodes"] = 1
    cfg["selection"] = {"start_step": 0, "eval_every": 150, "final_eval_episodes": 1}
    cfg["sac"]["gradient_steps_max"] = 6
    cfg["sac"]["real_ratio"] = 0.05
    cfg["ensemble"]["train_epochs_per_round"] = 2
    cfg["rollout"]["freq_steps"] = 100
    cfg["rollout"]["num_rollouts"] = 50
    with tempfile.TemporaryDirectory() as d:
        for algo in ("macura", "mbpo", "m2ac"):
            run = train_mod.train_one(algo, cfg, d, seed=0)
            rp = [v for _, v in run["real_pct"] if not np.isnan(v)]
            assert rp, f"{algo}: no real_pct telemetry logged"
            # realised share should sit near the fixed 5% target (loose bound for a tiny run)
            assert np.mean(rp) < 25.0, f"{algo}: realised real% {np.mean(rp):.0f} far above target"


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
