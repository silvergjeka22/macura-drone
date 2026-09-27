"""Dyna training loop shared by MACURA, MBPO, M2AC and SAC, plus the evaluation protocol.

All four use the same SAC learner, ensemble, evaluation scenarios and real/imagined batch mix; only the
rollout function differs (SAC has none). Every `eval_every` steps the greedy policy flies the fixed
selection scenarios (seeds 100..119); a new best return saves policy + ensemble. At the end the best
checkpoint flies fresh test scenarios (seeds 1000..1029) for the run summary.

    train_algo(algo, cfg, output_dir, seeds)  -> [run]   (reloads a seed whose log already exists)
    train_one(algo, cfg, output_dir, seed)    -> run     (logs/<algo>_seed<seed>.json, checkpoints/...)
    evaluate(agent, env, episodes, seeds)     -> metrics
"""

from __future__ import annotations

import collections
import json
import os
import time

import numpy as np

from src.envs import drone_env as env_mod
from src.models import ensemble as ens
from src.algorithms import sac as sac_mod
from src.algorithms import macura as macura_mod
from src.algorithms import mbpo as mbpo_mod
from src.algorithms import m2ac as m2ac_mod
from src.training.exploration import algo_exploration, make_noise, explore

try:
    import torch
except ImportError:
    torch = None


class _RealBuffer:
    """Real (obs, act, next_obs) for ensemble training and rollout start states."""

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
        idx = np.random.randint(0, self.size, size=n)
        return self.obs[idx]


def evaluate(agent, env, eval_episodes: int, eval_seeds=None) -> dict:
    """Greedy flights on the fixed scenarios base, base+1, ... (base = eval_seeds[0])."""
    base = int(eval_seeds[0]) if eval_seeds else 100
    returns, lengths, failures, successes, heavy = [], [], [], [], []
    max_sink, lift_steps, laps = [], 0, []
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
            failed = failed or info.get("failure", False)
            reached = reached or info.get("reached", False)
            done = terminated or truncated
        returns.append(ep_ret)
        lengths.append(ep_len)
        failures.append(float(failed))
        successes.append(float(reached))
        heavy.append(payload_max > 0.0 and env.payload >= 0.5 * payload_max)
        max_sink.append(ep_sink)
        if "laps" in info:
            laps.append(float(info["laps"]))
    out = {
        "eval_return": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length": float(np.mean(lengths)),
        "eval_failure_rate": float(np.mean(failures)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_max_sink": float(np.median(max_sink)),
        "eval_liftloss": float(lift_steps / max(1, sum(lengths))),
    }
    if laps:
        out["eval_laps"] = float(np.mean(laps))
    if payload_max > 0.0:
        f, h = np.array(failures), np.array(heavy, dtype=bool)
        out["eval_failure_light"] = float(f[~h].mean()) if (~h).any() else float("nan")
        out["eval_failure_heavy"] = float(f[h].mean()) if h.any() else float("nan")
    return out


def train_one(algo_name: str, cfg: dict, output_dir: str, seed: int = 0) -> dict:
    if torch is None:
        raise ImportError("torch required for training")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = f"{algo_name}_seed{seed}"
    _seed_everything(seed)

    env, obs_dim, act_dim = env_mod.make_env(cfg["env"], seed=seed)
    env.action_space.seed(seed)
    eval_env, _, _ = env_mod.make_env(cfg["env"], seed=cfg["experiment"]["eval_seeds"][0])
    reward_fn = env_mod.known_reward_fn(cfg["env"])
    done_fn = env_mod.termination_fn(cfg["env"])

    model_based = algo_name in ("macura", "mbpo", "m2ac")
    buf_cap = cfg["rollout"]["model_buffer_capacity"] if model_based else 1_000_000
    agent = sac_mod.build_sac(obs_dim, act_dim, {**cfg["sac"], "buffer_size": buf_cap}, device, seed)
    dynamics_model = ens.build_ensemble(cfg["ensemble"], obs_dim, act_dim, device) if model_based else None

    total_steps = cfg["experiment"]["total_env_steps"]
    real_buffer = _RealBuffer(total_steps, obs_dim, act_dim)
    # model-based: agent.replay_buffer holds imagined data, real_rb the real transitions
    model_rb = agent.replay_buffer if model_based else None
    real_rb = sac_mod.build_replay_buffer(agent, total_steps) if model_based else None
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
    num_updates = g_max if model_based else int(cfg["sac"].get("baseline_gradient_steps", 1))
    g_max_macura = int(cfg["rollout"]["macura"].get("gradient_steps_max", g_max))

    kappa_state: dict = {}
    trust_diag_state: dict = {}
    log = _empty_log()
    # imagined data expires after `lifetime` model rounds (MACURA reference code); 0 = one FIFO buffer
    lifetime = int(cfg["rollout"].get("model_lifetime_rounds") or 0) if model_based else 0
    round_sizes = collections.deque(maxlen=lifetime) if lifetime else None
    model_window = None
    if lifetime:
        log["utd"] = []
    best_return, best_step, best_ckpt = -np.inf, None, None
    model_trained = False
    expl_cfg = algo_exploration(cfg["exploration"], algo_name)
    noise = make_noise(expl_cfg, act_dim, cfg["env"]["max_episode_steps"], seed)
    train_episodes, train_crashes = 0, 0
    last_diag, tm = None, {"fit": 0.0, "imagine": 0.0, "sac": 0.0}
    t_run = t_eval = time.perf_counter()

    obs, _ = env.reset(seed=seed)
    for step in range(total_steps):
        act = env.action_space.sample() if step < warmup else explore(agent, obs, noise)
        next_obs, rew, terminated, truncated, info = env.step(act)
        real_buffer.add(obs, act, next_obs)
        sac_mod.add_to_buffer(real_rb if model_based else agent.replay_buffer,
                              obs, act, next_obs, rew, terminated)
        if terminated or truncated:
            train_episodes += 1
            train_crashes += bool(info.get("failure", False))
            obs = env.reset()[0]
            noise.reset()
        else:
            obs = next_obs
        if step < warmup:
            continue

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
                log["rollout_len_hist"] = diag["lengths"]
                for k in ("unc_near", "unc_far", "trust_near", "trust_far",
                          "unc_fast", "unc_slow", "trust_fast", "trust_slow",
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
            else:
                trans, diag = m2ac_mod.m2ac_rollout(
                    dynamics_model, agent, start, reward_fn, done_fn, cfg,
                    diag_state=trust_diag_state)
                num_updates = cfg["rollout"]["m2ac"]["fixed_gradient_steps"]
                log["rollout_length"].append((step, cfg["rollout"]["m2ac"]["t_max"]))
                log["untrusted_frac"].append((step, diag.get("untrusted_frac", float("nan"))))
                log["fast_frac"].append((step, diag["fast_frac"]))
            _store_model_transitions(agent, trans)
            if lifetime:
                round_sizes.append(sum(len(t[0]) for t in trans))
                model_window = min(sum(round_sizes), buf_cap)
                if algo_name == "macura":            # Eq. 22 on the live window
                    num_updates = macura_mod.gradient_steps(
                        model_window, buf_cap, g_max_macura,
                        cfg["rollout"]["macura"]["adaptive_gradient_steps"])
            diag["utd"] = num_updates
            model_trained = True
            last_diag = diag
            tm["imagine"] += time.perf_counter() - t1

        t_sac = time.perf_counter()
        if not model_based:
            if agent.replay_buffer.size() >= batch:
                sac_mod.sac_update(agent, num_updates, batch)
        elif model_trained:
            sac_mod.sac_update_mixed(agent, real_rb, model_rb, num_updates, batch, real_ratio, mix_rng,
                                     model_window=model_window)
        tm["sac"] += time.perf_counter() - t_sac

        if step % eval_every == 0:
            m = evaluate(agent, eval_env, cfg["experiment"]["eval_episodes"],
                         cfg["experiment"]["eval_seeds"])
            log["steps"].append(step)
            for k in ("eval_return", "eval_return_std", "eval_failure_rate", "eval_success_rate"):
                log[k].append(m[k])
            if "eval_failure_heavy" in m:
                log["eval_failure_light"].append(m["eval_failure_light"])
                log["eval_failure_heavy"].append(m["eval_failure_heavy"])
            log["train_crashes"].append(train_crashes)
            log["train_episodes"].append(train_episodes)
            log["eval_max_sink"].append(m["eval_max_sink"])
            log["eval_liftloss"].append(m["eval_liftloss"])
            if "eval_laps" in m:
                log["eval_laps"].append(m["eval_laps"])
            if "utd" in log:
                log["utd"].append(num_updates)
            improved = step >= start_step and m["eval_return"] > best_return
            if improved:
                best_return, best_step = m["eval_return"], step
                best_ckpt = _save_best(agent, dynamics_model, output_dir, run_name)
            print(f"[{algo_name} seed{seed}] step {step:>6}  return {m['eval_return']:7.1f}"
                  f"  reach {m['eval_success_rate']:.2f}  crash {m['eval_failure_rate']:.2f}"
                  f"  broken {train_crashes}/{train_episodes}"
                  f"{'  <- best' if improved else ''}")
            now = time.perf_counter()
            _print_status(algo_name, cfg, last_diag, m, tm, now - t_eval, now - t_run, step, total_steps, eval_every)
            t_eval, tm = now, {k: 0.0 for k in tm}

    if best_ckpt is None:
        best_ckpt = _save_best(agent, dynamics_model, output_dir, run_name)

    final_eval = _final_eval(agent, eval_env, best_ckpt, final_eval_episodes, _final_seeds(cfg))
    setup = {"exploration": expl_cfg["type"], "task": cfg["env"].get("task", ""),
             "xi": cfg["rollout"]["macura"]["xi"], "mbpo_horizon": cfg["rollout"]["mbpo"]["rollout_schedule"][1]}
    if lifetime:
        setup.update({"model_lifetime_rounds": lifetime, "macura_gmax": g_max_macura,
                      "mbpo_utd": cfg["rollout"]["mbpo"]["fixed_gradient_steps"]})
    meta = {"algo": algo_name, "seed": seed, "best_step": best_step,
            "best_return": float(best_return), "final_eval": final_eval, "setup": setup,
            "train_crashes_total": train_crashes, "train_episodes_total": train_episodes}
    _save_json(meta, output_dir, "checkpoints", f"{run_name}_best_meta.json", indent=2)
    print(f"[{algo_name} seed{seed}] FINAL  return {final_eval['eval_return']:.1f}"
          f"±{final_eval['eval_return_std']:.1f}  crash {final_eval['eval_failure_rate']:.2f}"
          f"  (best @ step {best_step})  real crashes while learning {train_crashes}/{train_episodes}")

    run = {"algo": algo_name, "seed": seed, "checkpoint": best_ckpt,
           "best_return": float(best_return), "best_step": best_step,
           "final_eval": final_eval, "setup": setup,
           "train_crashes_total": train_crashes, "train_episodes_total": train_episodes, **log}
    _save_json(run, output_dir, "logs", f"{run_name}.json")
    env.close()
    eval_env.close()
    return run


def train_algo(algo_name: str, cfg: dict, output_dir: str, seeds=None) -> list:
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


def _empty_log():
    """Per-eval series (steps, eval_*, train_*) and per-model-round (step, value) diagnostics."""
    return {"steps": [], "eval_return": [], "eval_return_std": [],
            "eval_failure_rate": [], "eval_success_rate": [],
            "kappa": [], "rollout_length": [],
            "base_uncertainty": [], "rollout_len_hist": [],
            "unc_near": [], "unc_far": [], "trust_near": [], "trust_far": [],
            "untrusted_frac": [], "discarded_frac": [],
            "unc_fast": [], "unc_slow": [], "trust_fast": [], "trust_slow": [], "fast_frac": [],
            "eval_failure_light": [], "eval_failure_heavy": [],
            "train_crashes": [], "train_episodes": [],
            "eval_max_sink": [], "eval_liftloss": [],
            "eval_laps": []}


def _pct(x):
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def _print_status(algo, cfg, d, m, tm, dt, elapsed, step, total_steps, every):
    """Two lines under each eval line: what the model round did and how the policy flies."""
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
        utd = f" | SAC updates/step {d['utd']}" if "utd" in d else ""
        print(f"    model: {line} | fast-descent share of its imagined data {_pct(d.get('fast_frac'))}{utd}")
    eta = (total_steps - step) * dt / max(1, every) / 3600
    laps = f"laps {m['eval_laps']:.2f} per flight | " if "eval_laps" in m else ""
    print(f"    policy: {laps}fastest descent {m['eval_max_sink']:.1f} m/s | losing lift {_pct(m['eval_liftloss'])}"
          f" of the time || time: {dt:.0f} s since last eval (model fit {tm['fit']:.0f} s, imagine "
          f"{tm['imagine']:.0f} s, SAC {tm['sac']:.0f} s) | run so far {elapsed / 60:.0f} min, ~{eta:.1f} h left",
          flush=True)


def _final_seeds(cfg):
    return [int(cfg.get("selection", {}).get("final_eval_seed_base", 1000))]


def _final_eval(agent, eval_env, best_ckpt, eval_episodes, eval_seeds=None):
    if best_ckpt:
        try:
            agent.set_parameters(best_ckpt, device=agent.device)
        except Exception:
            pass
    return evaluate(agent, eval_env, eval_episodes, eval_seeds)


def _store_model_transitions(agent, transitions):
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
    """Vectorised circular insert into the SB3 ReplayBuffer (n_envs=1)."""
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
    """checkpoints/<name>_best.zip (policy) and <name>_best_ensemble.pt (world model)."""
    path = os.path.join(output_dir, "checkpoints", f"{name}_best")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        agent.save(path)
    except Exception:
        return None
    if dynamics_model is not None:
        try:
            ens.save_ensemble(dynamics_model, path + "_ensemble.pt")
        except Exception:
            pass
    return path + ".zip"


def _save_json(obj, output_dir, sub, filename, indent=None):
    path = os.path.join(output_dir, sub)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), "w") as f:
        json.dump(obj, f, indent=indent)
