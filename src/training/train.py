"""Training orchestration — Dyna loop for all four algorithms, plus evaluation.

Backbone: Stable-Baselines3 SAC (shared learner) + our PyTorch probabilistic ensemble +
our pure-function rollout strategies. The four algorithms share EVERYTHING except which
rollout function is called (SAC calls none).

Data flow (fixed real_ratio — Janner/MBPO; MACURA inherits):
  * SAC baseline: one buffer (the agent's own), 100% REAL, no mixing;
  * model-based (MACURA/MBPO/M2AC): TWO buffers — the agent's own SB3 buffer holds MODEL
    (imagined) rollouts, a second SB3 `real_rb` holds REAL transitions. Each SAC batch is
    `real_ratio` REAL + the rest MODEL (within-batch mixing). A numpy `_RealBuffer` trains
    the ensemble and supplies rollout start states. Each algorithm keeps its own rollout
    strategy (MACURA adaptive-kappa, MBPO truncated-linear, M2AC mask).

Everything is shared/fixed across the four algorithms (SAC backbone, ensemble, pink-noise
exploration, eval seeds, real_ratio) EXCEPT the rollout strategy — so any MACURA win is
attributable to its adaptive rollout alone.

Selection: best checkpoint = highest periodic greedy-eval return on fixed shared seeds,
gated by `selection.start_step`; each new best saves policy + ensemble + meta. One larger
final greedy eval on the reloaded best is the run summary.

Pure-function library. The notebook calls these.

Public functions:
    train_one(algo_name, cfg, output_dir, seed)    -> run_dict (saves best ckpt only)
    train_algo(algo_name, cfg, output_dir, seeds)  -> [run_dict] over all seeds (reloads if logged)
    load_runs(*output_dirs)                        -> [run_dict] merged from several Kaggle outputs
    evaluate(agent, env, eval_episodes)            -> metrics
    evaluate_best(cfg, output_dir, device, ...)    -> [{algo, seed, eval_*}]  (loads best)
"""

from __future__ import annotations

import os
import json
import time
import numpy as np

from src.envs import drone_env as env_mod
from src.models import ensemble as ens
from src.algorithms import sac as sac_mod
from src.algorithms import macura as macura_mod
from src.algorithms import mbpo as mbpo_mod
from src.algorithms import m2ac as m2ac_mod

try:
    import torch
except ImportError:
    torch = None


# ── a minimal numpy replay buffer for REAL data (ensemble training + rollout starts) ──
class _RealBuffer:
    """REAL (obs, act, next_obs) — all the ensemble needs (it predicts next_obs from obs+act;
    reward and termination are analytic). The SAC learner reads real transitions from the parallel
    SB3 `real_rb` instead, so this stays a small, decoupled numpy store."""

    def __init__(self, capacity, obs_dim, act_dim):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.cap, self.size, self.ptr = capacity, 0, 0

    def add(self, obs, act, next_obs):
        i = self.ptr
        self.obs[i], self.act[i], self.next_obs[i] = obs, act, next_obs
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def all(self):
        s = self.size
        return {"obs": self.obs[:s], "act": self.act[:s], "next_obs": self.next_obs[:s]}

    def sample_obs(self, n):
        """n rollout start states, WITH replacement (as MBPO/M2AC/MACURA sample them), so a large rollout
        batch (paper protocol: 25,000) is not cut down to the number of real transitions early on."""
        idx = np.random.randint(0, self.size, size=n)
        return self.obs[idx]


# ── evaluation (identical protocol for all algorithms) ────────────────────────
def evaluate(agent, env, eval_episodes: int, eval_seeds=None) -> dict:
    # FIXED per-episode scenarios: reset each eval episode from a deterministic seed so every
    # evaluation (and every algorithm) faces the IDENTICAL set of wind/obstacle/init scenarios.
    # Without this, env.reset() drew fresh scenarios each time, so a 5-episode eval had ~±0.22
    # crash-rate / ~±44 return sampling noise - which is what made the learning curve look wildly
    # unstable even when the policy was steady. Same base -> comparable across algos (fairness).
    base = int(eval_seeds[0]) if eval_seeds else 100
    returns, lengths, failures, successes, heavy = [], [], [], [], []
    max_sink, lift_steps = [], 0                        # measurement only (read from the sim state)
    payload_max = float(getattr(env, "payload_max", 0.0))
    raw = getattr(env, "unwrapped", env)
    for i in range(eval_episodes):
        obs, _ = env.reset(seed=base + i)
        done = False
        ep_ret, ep_len, failed, reached, ep_sink = 0.0, 0, False, False, 0.0
        while not done:
            act = sac_mod.select_action(agent, obs, evaluate=True)
            obs, rew, terminated, truncated, info = env.step(act)
            ep_sink = max(ep_sink, -float(raw.data.qvel[2]))
            lift_steps += float(getattr(raw, "thrust_eff", 1.0)) < 0.99
            ep_ret += rew
            ep_len += 1
            failed = failed or info.get("failure", False)     # crashed
            reached = reached or info.get("reached", False)    # reached & held the target
            done = terminated or truncated
        returns.append(ep_ret)
        lengths.append(ep_len)
        failures.append(float(failed))
        successes.append(float(reached))
        heavy.append(payload_max > 0.0 and env.payload >= 0.5 * payload_max)
        max_sink.append(ep_sink)
    out = {
        "eval_return": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length": float(np.mean(lengths)),
        "eval_failure_rate": float(np.mean(failures)),   # crash rate
        "eval_success_rate": float(np.mean(successes)),  # reached-and-held rate
        "eval_max_sink": float(np.median(max_sink)),     # fastest descent (m/s), median over episodes
        "eval_liftloss": float(lift_steps / max(1, sum(lengths))),   # share of steps losing lift
    }
    if payload_max > 0.0:                                # delivery task: crash rate by package weight
        f, h = np.array(failures), np.array(heavy, dtype=bool)
        out["eval_failure_light"] = float(f[~h].mean()) if (~h).any() else float("nan")
        out["eval_failure_heavy"] = float(f[h].mean()) if h.any() else float("nan")
    return out


# ── a single training run ─────────────────────────────────────────────────────
def train_one(algo_name: str, cfg: dict, output_dir: str, seed: int = 0) -> dict:
    """Train one algorithm in {'macura','mbpo','m2ac','sac'} and log curves.
    Saves only the best checkpoint (highest periodic greedy-eval return)."""
    if torch is None:
        raise ImportError("torch required for training")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = f"{algo_name}_seed{seed}"
    _seed_everything(seed)

    env, obs_dim, act_dim = env_mod.make_env(cfg["env"], seed=seed)
    env.action_space.seed(seed)       # warm-up random actions: reproducible (was seeded from the OS -> every
                                      # run started from different data, so a seed did not reproduce a run)
    eval_env, _, _ = env_mod.make_env(cfg["env"], seed=cfg["experiment"]["eval_seeds"][0])
    reward_fn = env_mod.known_reward_fn(cfg["env"])
    done_fn = env_mod.termination_fn(cfg["env"])

    model_based = algo_name in ("macura", "mbpo", "m2ac")
    buf_cap = cfg["rollout"]["model_buffer_capacity"] if model_based else 1_000_000
    sac_cfg = {**cfg["sac"], "buffer_size": buf_cap}
    agent = sac_mod.build_sac(obs_dim, act_dim, sac_cfg, device, seed)
    dynamics_model = ens.build_ensemble(cfg["ensemble"], obs_dim, act_dim, device) if model_based else None

    total_steps = cfg["experiment"]["total_env_steps"]
    real_buffer = _RealBuffer(total_steps, obs_dim, act_dim)

    # Two-buffer setup for the model-based agents: the agent's own SB3 buffer holds MODEL (imagined)
    # transitions; a second SB3 buffer holds REAL transitions, so each SAC batch mixes the two
    # (within-batch). The SAC baseline keeps one buffer (100% real).
    model_rb = agent.replay_buffer if model_based else None
    real_rb = sac_mod.build_replay_buffer(agent, total_steps) if model_based else None

    # FIXED real_ratio (Janner/MBPO; MACURA inherits) — IDENTICAL for MACURA/MBPO/M2AC so it
    # cannot bias the comparison; only their rollout strategy differs.
    real_ratio = float(cfg["sac"]["real_ratio"])
    mix_rng = np.random.default_rng(seed)

    warmup = cfg["experiment"]["warmup_random_steps"]
    sel = cfg.get("selection", {})
    eval_every = int(sel.get("eval_every", cfg["experiment"]["eval_every_steps"]))
    start_step = int(sel.get("start_step", 0))
    final_eval_episodes = int(sel.get("final_eval_episodes", cfg["experiment"]["eval_episodes"]))
    rollout_freq = cfg["rollout"]["freq_steps"]
    num_rollouts = cfg["rollout"]["num_rollouts"]
    g_max = cfg["sac"]["gradient_steps_max"]
    batch = cfg["sac"]["batch_size"]
    # UTD: the model-free SAC baseline runs a STANDARD ~1 update per real step
    # (`baseline_gradient_steps`), not the model-based ceiling g_max. Model-based agents
    # carry their per-round UTD (MACURA Eq. 22 adaptive, MBPO/M2AC fixed) between rounds.
    baseline_g = int(cfg["sac"].get("baseline_gradient_steps", 1))
    num_updates = g_max if model_based else baseline_g

    kappa_state: dict = {}
    trust_diag_state: dict = {}   # MACURA's trust rule applied to MBPO/M2AC rollouts (measurement only)
    log = _empty_log()
    best_return = -np.inf
    best_step = None
    best_ckpt = None
    model_trained = False
    expl_cfg = algo_exploration(cfg["exploration"], algo_name)
    noise = _make_noise(expl_cfg, act_dim, cfg["env"]["max_episode_steps"], seed)
    # MACURA's own UTD ceiling (paper: Gmax = 2 x MBPO's G, since Eq. 22 uses about half of it)
    g_max_macura = int(cfg["rollout"]["macura"].get("gradient_steps_max", g_max))
    # REAL drones flown / broken while learning (measurement only: counts episodes that already happen)
    train_episodes, train_crashes = 0, 0
    # status lines (measurement only): the last model round's diagnostics + where the time goes
    last_diag, tm = None, {"fit": 0.0, "imagine": 0.0, "sac": 0.0}
    t_run = t_eval = time.perf_counter()

    obs, _ = env.reset(seed=seed)
    for step in range(total_steps):
        # --- act in the real environment ---
        act = env.action_space.sample() if step < warmup else _explore(agent, obs, noise)
        next_obs, rew, terminated, truncated, info = env.step(act)
        real_buffer.add(obs, act, next_obs)                       # numpy → ensemble + rollout starts
        # REAL transitions: into real_rb for model-based (mixing), the agent's buffer for SAC
        sac_mod.add_to_buffer(real_rb if model_based else agent.replay_buffer,
                              obs, act, next_obs, rew, terminated)
        if terminated or truncated:
            train_episodes += 1
            train_crashes += bool(info.get("failure", False))
            obs = env.reset()[0]
            noise.reset()                       # fresh pink sequence per episode
        else:
            obs = next_obs
        if step < warmup:
            continue

        # --- model-based: retrain ensemble + generate fresh rollouts ---
        # (needs one batch of real data; start states are sampled with replacement, so a large rollout batch
        # does not have to wait for that many real steps - it used to, which would have delayed the paper
        # protocol's first model round to step 25,000)
        if model_based and step % rollout_freq == 0 and real_buffer.size >= batch:
            t0 = time.perf_counter()
            ens.train_ensemble(dynamics_model, real_buffer.all(), cfg["ensemble"])
            t1 = time.perf_counter()
            tm["fit"] += t1 - t0
            start = real_buffer.sample_obs(num_rollouts)
            if algo_name == "macura":
                trans, diag = macura_mod.macura_rollout(
                    dynamics_model, agent, start, reward_fn, done_fn, kappa_state, cfg)
                num_updates = macura_mod.gradient_steps(
                    agent.replay_buffer.size(), buf_cap, g_max_macura,
                    cfg["rollout"]["macura"]["adaptive_gradient_steps"])
                log["kappa"].append((step, diag["kappa"]))
                log["rollout_length"].append((step, diag["mean_rollout_length"]))
                log["base_uncertainty"].append((step, diag["base_uncertainty"]))
                log["rollout_len_hist"] = diag["lengths"]        # latest round's truncation lengths
                for k in ("unc_near", "unc_far", "trust_near", "trust_far",   # landing zone vs transit
                          "unc_fast", "unc_slow", "trust_fast", "trust_slow",  # fast vs slow descent
                          "untrusted_frac", "discarded_frac", "fast_frac"):
                    log[k].append((step, diag[k]))
            elif algo_name == "mbpo":
                trans, diag = mbpo_mod.mbpo_rollout(
                    dynamics_model, agent, start, reward_fn, done_fn, step, cfg,
                    diag_state=trust_diag_state)
                num_updates = cfg["rollout"]["mbpo"]["fixed_gradient_steps"]
                log["rollout_length"].append((step, diag["rollout_length"]))
                log["untrusted_frac"].append((step, diag.get("untrusted_frac", float("nan"))))
                log["fast_frac"].append((step, diag["fast_frac"]))
            else:  # m2ac
                trans, diag = m2ac_mod.m2ac_rollout(
                    dynamics_model, agent, start, reward_fn, done_fn, cfg,
                    diag_state=trust_diag_state)
                num_updates = cfg["rollout"]["m2ac"]["fixed_gradient_steps"]
                log["rollout_length"].append((step, cfg["rollout"]["m2ac"]["t_max"]))  # fixed horizon
                log["untrusted_frac"].append((step, diag.get("untrusted_frac", float("nan"))))
                log["fast_frac"].append((step, diag["fast_frac"]))
            _store_model_transitions(agent, trans)
            model_trained = True
            last_diag = diag
            tm["imagine"] += time.perf_counter() - t1

        # --- SAC updates ---
        t_sac = time.perf_counter()
        if not model_based:
            # model-free baseline: one buffer, 100% real
            if agent.replay_buffer.size() >= batch:
                sac_mod.sac_update(agent, num_updates, batch)
        elif model_trained:
            # model-based: within-batch real/imagined mixing at the FIXED real_ratio. The SAME
            # path runs for MACURA/MBPO/M2AC; only `num_updates` (UTD) and which rollout produced
            # the model data differ. The SAC math is untouched.
            sac_mod.sac_update_mixed(agent, real_rb, model_rb, num_updates, batch, real_ratio, mix_rng)
        tm["sac"] += time.perf_counter() - t_sac

        # --- periodic GREEDY evaluation on FIXED shared seeds → headline curve + selection ---
        if step % eval_every == 0:
            m = evaluate(agent, eval_env, cfg["experiment"]["eval_episodes"],
                         cfg["experiment"]["eval_seeds"])
            log["steps"].append(step)
            log["eval_return"].append(m["eval_return"])
            log["eval_return_std"].append(m["eval_return_std"])
            log["eval_failure_rate"].append(m["eval_failure_rate"])
            log["eval_success_rate"].append(m["eval_success_rate"])
            if "eval_failure_heavy" in m:                      # delivery task: crash by package weight
                log["eval_failure_light"].append(m["eval_failure_light"])
                log["eval_failure_heavy"].append(m["eval_failure_heavy"])
            log["train_crashes"].append(train_crashes)         # cumulative, real training flights
            log["train_episodes"].append(train_episodes)
            log["eval_max_sink"].append(m["eval_max_sink"])
            log["eval_liftloss"].append(m["eval_liftloss"])
            improved = step >= start_step and m["eval_return"] > best_return
            if improved:                       # overwrite best checkpoint (policy + ensemble + meta)
                best_return = m["eval_return"]
                best_step = step
                best_ckpt = _save_best(agent, dynamics_model, output_dir, run_name)
            print(f"[{algo_name} seed{seed}] step {step:>6}  return {m['eval_return']:7.1f}"
                  f"  reach {m['eval_success_rate']:.2f}  crash {m['eval_failure_rate']:.2f}"
                  f"  broken {train_crashes}/{train_episodes}"
                  f"{'  <- best' if improved else ''}")
            now = time.perf_counter()
            _print_status(algo_name, cfg, last_diag, m, tm, now - t_eval, now - t_run, step, total_steps, eval_every)
            t_eval, tm = now, {k: 0.0 for k in tm}

    if best_ckpt is None:                       # never improved (e.g. no eval / before start_step)
        best_ckpt = _save_best(agent, dynamics_model, output_dir, run_name)

    # --- one larger final GREEDY eval on the reloaded best checkpoint → run summary ---
    # on FRESH scenarios (final_eval_seed_base..), not the ones used to pick the best checkpoint:
    # re-testing on the selection scenarios would overrate lucky, high-variance checkpoints.
    final_eval = _final_eval(agent, eval_env, best_ckpt, final_eval_episodes, _final_seeds(cfg))
    setup = {"exploration": expl_cfg["type"], "task": cfg["env"].get("task", ""),     # what this run used
             "xi": cfg["rollout"]["macura"]["xi"], "mbpo_horizon": cfg["rollout"]["mbpo"]["rollout_schedule"][1]}
    meta = {"algo": algo_name, "seed": seed, "best_step": best_step,
            "best_return": float(best_return), "final_eval": final_eval, "setup": setup,
            "train_crashes_total": train_crashes, "train_episodes_total": train_episodes}
    _save_meta(meta, output_dir, run_name)
    print(f"[{algo_name} seed{seed}] FINAL  return {final_eval['eval_return']:.1f}"
          f"±{final_eval['eval_return_std']:.1f}  crash {final_eval['eval_failure_rate']:.2f}"
          f"  (best @ step {best_step})  real crashes while learning {train_crashes}/{train_episodes}")

    run = {"algo": algo_name, "seed": seed, "checkpoint": best_ckpt,
           "best_return": float(best_return), "best_step": best_step,
           "final_eval": final_eval, "setup": setup,
           "train_crashes_total": train_crashes, "train_episodes_total": train_episodes, **log}
    _save_run(run, output_dir, run_name)
    env.close()
    eval_env.close()
    return run


def train_algo(algo_name: str, cfg: dict, output_dir: str, seeds=None) -> list:
    """Train one algorithm across all seeds and return the list of run dicts. A seed whose
    log already exists is reloaded instead of retrained, so re-running the notebook resumes."""
    seeds = seeds if seeds is not None else cfg["experiment"]["seeds"]
    runs = []
    for seed in seeds:
        log_path = os.path.join(output_dir, "logs", f"{algo_name}_seed{seed}.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                runs.append(json.load(f))
        else:
            runs.append(train_one(algo_name, cfg, output_dir, seed))
    return runs


def best_run(runs: list) -> dict:
    """Return the run with the highest best_return (the strongest seed of an algorithm)."""
    return max(runs, key=lambda r: r.get("best_return", -1e18))


def load_runs(*output_dirs, algorithms=None) -> list:
    """Merge the run logs of several (downloaded) Kaggle outputs, e.g. load_runs("out1/runs", "out2/runs")
    when the seeds were split across commits. A (algo, seed) pair found twice is kept once (first dir
    wins). Feed the result to the plots (IQM + bootstrap CIs over all seeds)."""
    import glob
    runs, seen = [], set()
    for d in output_dirs:
        for path in sorted(glob.glob(os.path.join(d, "logs", "*_seed*.json"))):
            with open(path) as f:
                r = json.load(f)
            key = (r.get("algo"), r.get("seed"))
            if key in seen or (algorithms and r.get("algo") not in algorithms):
                continue
            seen.add(key)
            runs.append(r)
    return runs


# small internals
def _empty_log():
    return {"steps": [], "eval_return": [], "eval_return_std": [],
            "eval_failure_rate": [], "eval_success_rate": [],
            "kappa": [], "rollout_length": [],
            # diagnostics (do not affect training): MACURA first-step GJS uncertainty over
            # training, and the last round's per-rollout truncation lengths (distribution plot).
            "base_uncertainty": [], "rollout_len_hist": [],
            # MACURA landing zone vs transit: mean model uncertainty + fraction of imagined steps trusted
            "unc_near": [], "unc_far": [], "trust_near": [], "trust_far": [],
            # WHY-diagnostic: share of the imagined data each method TRAINS on that is above MACURA's
            # trust threshold (MBPO/M2AC measured with the same rule; MACURA = 0 by construction),
            # and the share MACURA generated but threw away.
            "untrusted_frac": [], "discarded_frac": [],
            # delivery task: MACURA uncertainty/trust in FAST vs SLOW descents, the share of each method's
            # imagined training data that is a fast descent, and eval crash rate by package weight
            "unc_fast": [], "unc_slow": [], "trust_fast": [], "trust_slow": [], "fast_frac": [],
            "eval_failure_light": [], "eval_failure_heavy": [],
            # real training flights so far and how many of them crashed, at each eval point (cumulative):
            # the drones each method breaks WHILE learning - believing a wrong model costs real crashes
            "train_crashes": [], "train_episodes": [],
            # the greedy policy in the evals: fastest descent (median m/s) and share of steps losing lift
            "eval_max_sink": [], "eval_liftloss": []}


def _pct(x):
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def _print_status(algo, cfg, d, m, tm, dt, elapsed, step, total_steps, every):
    """Two short lines under each eval line (measurement only): is the mechanism doing what it should,
    how does the policy fly, and where does the time go. `d` = the latest model round's diagnostics.
      MACURA: imagined trip length (of t_max) and how much it trusts FAST vs SLOW descents (want: short
              trips / low trust near fast descents, high trust elsewhere).
      MBPO / M2AC: share of the imagined data they train on that MACURA's rule would reject, and the
              share of it that is a fast descent (the lured-into-diving signal)."""
    if d is not None:
        if algo == "macura":
            tmax = cfg["rollout"]["macura"]["t_max"]
            line = (f"trip {d['mean_rollout_length']:.1f}/{tmax} steps | trusts fast descents "
                    f"{_pct(d.get('trust_fast'))} vs slow {_pct(d.get('trust_slow'))}")
        elif algo == "mbpo":
            line = (f"trip {d['rollout_length']}/{d['rollout_length']} steps (fixed) | data MACURA would "
                    f"reject {_pct(d.get('untrusted_frac'))}")
        else:
            line = (f"keeps {_pct(d.get('kept_fraction'))} of imagined steps | data MACURA would reject "
                    f"{_pct(d.get('untrusted_frac'))}")
        print(f"    model: {line} | fast-descent share of its imagined data {_pct(d.get('fast_frac'))}")
    rate = dt / max(1, every)
    eta = (total_steps - step) * rate / 3600
    print(f"    policy: fastest descent {m['eval_max_sink']:.1f} m/s | losing lift {_pct(m['eval_liftloss'])}"
          f" of the time || time: {dt:.0f} s since last eval (model fit {tm['fit']:.0f} s, imagine "
          f"{tm['imagine']:.0f} s, SAC {tm['sac']:.0f} s) | run so far {elapsed / 60:.0f} min, ~{eta:.1f} h left",
          flush=True)


def _final_seeds(cfg):
    """Base seed of the final-eval scenarios - separate from the selection scenarios (eval_seeds)."""
    return [int(cfg.get("selection", {}).get("final_eval_seed_base", 1000))]


def _final_eval(agent, eval_env, best_ckpt, eval_episodes, eval_seeds=None):
    """Reload the best policy and run ONE larger greedy eval as the run summary.
    Best-effort: if reload fails (or there is no checkpoint) evaluate the in-memory agent."""
    if best_ckpt:
        try:
            agent.set_parameters(best_ckpt, device=agent.device)
        except Exception:
            pass
    return evaluate(agent, eval_env, eval_episodes, eval_seeds)


# ── exploration noise (pink/white, applied identically to all algorithms) ──────
class _WhiteNoise:
    """Uncorrelated Gaussian action noise."""
    def __init__(self, act_dim, scale, seed=0):
        self.act_dim, self.scale = act_dim, float(scale)
        self.rng = np.random.default_rng(seed)
    def reset(self):
        pass
    def sample(self):
        return self.scale * self.rng.normal(size=self.act_dim)


class _PinkNoise:
    """Temporally-correlated 1/f (pink) action noise, per action dim. A pink sequence of
    length `horizon` is precomputed per episode via FFT (Eberhard et al. 2023, used by
    MACURA), then replayed one step at a time."""
    def __init__(self, act_dim, horizon, scale, seed=0):
        self.act_dim = act_dim
        self.horizon = max(int(horizon), 2)
        self.scale = float(scale)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.seq = self._gen()

    def _gen(self):
        n = self.horizon
        f = np.fft.rfftfreq(n)
        f[0] = f[1]
        cols = []
        for _ in range(self.act_dim):
            spec = (self.rng.normal(size=f.shape) + 1j * self.rng.normal(size=f.shape)) / np.sqrt(f)
            x = np.fft.irfft(spec, n=n)
            cols.append(x / (x.std() + 1e-9))
        return np.stack(cols, axis=1)        # (horizon, act_dim), unit std per col

    def reset(self):
        self.t = 0
        self.seq = self._gen()

    def sample(self):
        v = self.seq[self.t % self.horizon]
        self.t += 1
        return self.scale * v


class _NoNoise:
    def reset(self):
        pass
    def sample(self):
        return 0.0


class _Stochastic(_NoNoise):
    """Standard SAC exploration: sample from the policy's own squashed Gaussian (no added noise)."""
    stochastic = True


def algo_exploration(expl_cfg, algo_name):
    """The exploration scheme `algo_name` uses. Default: the same `type` for every algorithm. The paper
    protocol sets `per_algo` (MACURA pink noise, MBPO/M2AC deterministic, SAC stochastic), which is what
    the MACURA paper ran (Sec. 7.3 / App. D.3) - a known confound that must be reported with the results."""
    t = (expl_cfg.get("per_algo") or {}).get(algo_name, expl_cfg.get("type", "white_noise"))
    return {**expl_cfg, "type": t}


def _make_noise(expl_cfg, act_dim, horizon, seed):
    t = expl_cfg.get("type", "white_noise")
    s = float(expl_cfg.get("scale", 0.1))
    if t == "deterministic":
        return _NoNoise()
    if t == "stochastic":
        return _Stochastic()
    if t == "pink_noise":
        return _PinkNoise(act_dim, horizon, s, seed)
    return _WhiteNoise(act_dim, s, seed)


def _explore(agent, obs, noise_proc):
    """Policy mean + exploration noise, or a sample from the SAC policy for `stochastic`."""
    if getattr(noise_proc, "stochastic", False):
        return sac_mod.select_action(agent, obs, evaluate=False)
    act = sac_mod.select_action(agent, obs, evaluate=True)
    return np.clip(act + noise_proc.sample(), -1.0, 1.0)


def _store_model_transitions(agent, transitions):
    """Add all imagined transitions to the SB3 replay buffer in ONE vectorized bulk write.
    Falls back to per-row add if the buffer layout differs on the installed SB3 version."""
    if not transitions:
        return
    obs = np.concatenate([t[0] for t in transitions]).astype(np.float32)
    act = np.concatenate([t[1] for t in transitions]).astype(np.float32)
    rew = np.concatenate([t[2] for t in transitions]).astype(np.float32)
    nxt = np.concatenate([t[3] for t in transitions]).astype(np.float32)
    done = np.concatenate([t[4] for t in transitions]).astype(np.float32)
    try:
        _bulk_add(agent, obs, act, nxt, rew, done)
    except Exception:
        for i in range(len(obs)):
            sac_mod.add_transition(agent, obs[i], act[i], nxt[i], float(rew[i]), bool(done[i]))


def _bulk_add(agent, obs, act, next_obs, rew, done):
    """Vectorized circular-buffer insert into the SB3 ReplayBuffer (n_envs=1)."""
    rb = agent.replay_buffer
    n = len(obs)
    cap = rb.buffer_size
    idx = (rb.pos + np.arange(n)) % cap
    rb.observations[idx, 0] = obs
    if getattr(rb, "optimize_memory_usage", False):
        rb.observations[(idx + 1) % cap, 0] = next_obs
    else:
        rb.next_observations[idx, 0] = next_obs
    rb.actions[idx, 0] = act
    rb.rewards[idx, 0] = rew
    rb.dones[idx, 0] = done
    if hasattr(rb, "timeouts"):
        rb.timeouts[idx, 0] = 0.0
    if rb.pos + n >= cap:
        rb.full = True
    rb.pos = (rb.pos + n) % cap


def _seed_everything(seed):
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _save_best(agent, dynamics_model, output_dir, name):
    """Overwrite the single best checkpoint on Drive: <name>_best.zip (SB3 policy) and,
    for model-based agents, <name>_best_ensemble.pt (world-model weights + normalizer).
    Returns the policy .zip path (or None on failure)."""
    path = os.path.join(output_dir, "checkpoints", f"{name}_best")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        agent.save(path)
    except Exception:
        return None
    if dynamics_model is not None:                  # model-based: also persist the ensemble
        try:
            ens.save_ensemble(dynamics_model, path + "_ensemble.pt")
        except Exception:
            pass
    return path + ".zip"


def _save_meta(meta, output_dir, name):
    path = os.path.join(output_dir, "checkpoints")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{name}_best_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _save_run(run, output_dir, name):
    path = os.path.join(output_dir, "logs")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{name}.json"), "w") as f:
        json.dump(run, f)


# ── final-evaluation helpers (load best models from Drive) ────────────────────
def evaluate_best(cfg: dict, output_dir: str, device: str = "cuda",
                  seeds=None, eval_episodes=None) -> list:
    """Load every <algo>_seed<seed>_best.zip from Drive and evaluate it.
    Returns a list of {algo, seed, eval_return, ...} dicts."""
    from stable_baselines3 import SAC
    seeds = seeds or cfg["experiment"]["seeds"]
    n = eval_episodes or cfg["experiment"]["eval_episodes"]
    eval_env, _, _ = env_mod.make_env(cfg["env"], seed=cfg["experiment"]["eval_seeds"][0])
    out = []
    for algo in cfg["experiment"]["algorithms"]:
        for seed in seeds:
            ck = os.path.join(output_dir, "checkpoints", f"{algo}_seed{seed}_best.zip")
            if not os.path.exists(ck):
                continue
            agent = SAC.load(ck, device=device)
            out.append({"algo": algo, "seed": seed, **evaluate(agent, eval_env, n, _final_seeds(cfg))})
    eval_env.close()
    return out
