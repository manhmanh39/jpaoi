# -*- coding: utf-8 -*-
"""
uav_hap_core.py (reconstructed from provided document, unmodified)
"""

from __future__ import annotations
import numpy as np
from scipy.special import gammaincc, gamma as gamma_func
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import itertools
import copy

RNG_SEED = 202601
np.random.seed(RNG_SEED)

@dataclass
class Config:
    u_weight: Dict[int, float] = field(default_factory=lambda: {1: 10.0, 2: 5.0, 3: 1.0})
    w_delay: Dict[int, float] = field(default_factory=lambda: {1: 3.0, 2: 2.0, 3: 1.0})
    xi_age: Dict[int, float] = field(default_factory=lambda: {1: 5.0, 2: 3.0})
    mu_D: float = 0.35
    mu_A: float = 0.4

    classes: Tuple[int, int, int] = (1, 2, 3)

    L: Dict[int, float] = field(default_factory=lambda: {1: 100000,
                                                           2: 200000,
                                                           3: 400000})

    lam0: Dict[int, float] = field(default_factory=lambda: {1: 0.4,
                                                              2: 0.6,
                                                              3: 0.8})

    B_tot: float = 2.0e6
    B_UH: float = 1.0e6
    B_HG: float = 1.0e6

    D_max: Dict[int, float] = field(default_factory=lambda: {1: 0.6, 2: 0.8, 3: 2.5})
    A_max: Dict[int, float] = field(default_factory=lambda: {1: 3.0, 2: 2.5, 3: 3.5})

    K_protected: Tuple[int, ...] = (1, 2)
    K_fresh: Tuple[int, ...] = (1, 2)
    K_crit: Tuple[int, ...] = (1,)
    omega: Dict[int, float] = field(default_factory=lambda: {1: 3.0, 2: 2.0, 3: 1.0})

    gamma_min: float = 0.1
    Q_quad: int = 64

    bar_gamma_UH: float = 10.0
    m_UH: float = 2.0

    p_LoS: float = 0.7
    bar_gamma_HG_LoS: float = 15.0
    bar_gamma_HG_NLoS: float = 5.0
    m_HG_LoS: float = 3.0
    m_HG_NLoS: float = 1.5

    alpha: float = 0.5
    a1: float = 1.0
    a2: float = 1.0
    a3: float = 1.0

    def lam(self, kappa: float = 1.0) -> Dict[int, float]:
        return {k: kappa * self.lam0[k] for k in self.classes}


CFG = Config()


def truncated_service_moments(L_k, eta_ij, B_ij, bar_gamma, m, gamma_min, Q=64):
    c = L_k * np.log(2.0) / (eta_ij * B_ij)
    a = m * gamma_min / bar_gamma
    p_suc = gammaincc(m, a)
    p_suc = max(p_suc, 1e-12)

    nodes, weights = np.polynomial.laguerre.laggauss(Q)
    t = a + nodes
    log_term = np.log1p((bar_gamma / m) * t)
    log_term = np.clip(log_term, 1e-12, None)

    pref = np.exp(-a) / (p_suc * gamma_func(m))
    base = t ** (m - 1.0)

    I1 = pref * np.sum(weights * base / log_term)
    I2 = pref * np.sum(weights * base / (log_term ** 2))

    E_S = c * I1
    E_S2 = (c ** 2) * I2
    return E_S, E_S2


def outage_probability(bar_gamma, m, gamma_min):
    a = m * gamma_min / bar_gamma
    return 1.0 - gammaincc(m, a)


def service_moments_UH(cfg, k, alpha=None, B_UH=None, bar_gamma_UH=None, m_UH=None):
    alpha = cfg.alpha if alpha is None else alpha
    B_UH = cfg.B_UH if B_UH is None else B_UH
    bar_gamma_UH = cfg.bar_gamma_UH if bar_gamma_UH is None else bar_gamma_UH
    m_UH = cfg.m_UH if m_UH is None else m_UH
    eta_UH = 1.0 - alpha
    return truncated_service_moments(cfg.L[k], eta_UH, B_UH, bar_gamma_UH,
                                      m_UH, cfg.gamma_min, cfg.Q_quad)


def service_moments_HG(cfg, k, B_HG=None, p_LoS=None,
                        bar_gamma_HG_LoS=None, bar_gamma_HG_NLoS=None):
    B_HG = cfg.B_HG if B_HG is None else B_HG
    p_LoS = cfg.p_LoS if p_LoS is None else p_LoS
    bar_gamma_HG_LoS = cfg.bar_gamma_HG_LoS if bar_gamma_HG_LoS is None else bar_gamma_HG_LoS
    bar_gamma_HG_NLoS = cfg.bar_gamma_HG_NLoS if bar_gamma_HG_NLoS is None else bar_gamma_HG_NLoS
    eta_HG = 1.0

    E_S_los, E_S2_los = truncated_service_moments(cfg.L[k], eta_HG, B_HG,
                                                    bar_gamma_HG_LoS, cfg.m_HG_LoS,
                                                    cfg.gamma_min, cfg.Q_quad)
    E_S_nlos, E_S2_nlos = truncated_service_moments(cfg.L[k], eta_HG, B_HG,
                                                      bar_gamma_HG_NLoS, cfg.m_HG_NLoS,
                                                      cfg.gamma_min, cfg.Q_quad)
    E_S = p_LoS * E_S_los + (1 - p_LoS) * E_S_nlos
    E_S2 = p_LoS * E_S2_los + (1 - p_LoS) * E_S2_nlos
    return E_S, E_S2


def priority_stage_delays(lam_tilde, moments, classes_order=(1, 2, 3)):
    rho = {k: lam_tilde[k] * moments[k][0] for k in classes_order}
    rho_total = sum(rho.values())

    W0 = sum(lam_tilde[k] * moments[k][1] for k in classes_order) / 2.0

    rho_cum = {}
    cum = 0.0
    for k in classes_order:
        rho_cum[k] = cum
        cum += rho[k]
    rho_cum_incl = {}
    cum = 0.0
    for k in classes_order:
        cum += rho[k]
        rho_cum_incl[k] = cum

    W = {}
    T = {}
    for k in classes_order:
        denom = (1.0 - rho_cum_incl[k]) * (1.0 - rho_cum[k])
        if denom <= 1e-9:
            W[k] = np.inf
        else:
            W[k] = W0 / denom
        T[k] = W[k] + moments[k][0]
    return T, rho, rho_total, W0


def evaluate_analytical(cfg, alpha, B_UH, B_HG, a, kappa=1.0,
                         bar_gamma_UH=None, p_LoS=None,
                         bar_gamma_HG_LoS=None, bar_gamma_HG_NLoS=None):
    classes = cfg.classes
    lam_k = cfg.lam(kappa)
    lam_tilde = {k: a[k] * lam_k[k] for k in classes}

    mom_U = {k: service_moments_UH(cfg, k, alpha=alpha, B_UH=B_UH,
                                    bar_gamma_UH=bar_gamma_UH) for k in classes}
    mom_H = {k: service_moments_HG(cfg, k, B_HG=B_HG, p_LoS=p_LoS,
                                    bar_gamma_HG_LoS=bar_gamma_HG_LoS,
                                    bar_gamma_HG_NLoS=bar_gamma_HG_NLoS) for k in classes}

    T_U, rho_U, rho_U_tot, W0_U = priority_stage_delays(lam_tilde, mom_U, classes)
    T_H, rho_H, rho_H_tot, W0_H = priority_stage_delays(lam_tilde, mom_H, classes)

    T_E2E = {k: T_U[k] + T_H[k] for k in classes}

    PAoI = {}
    for k in classes:
        if lam_tilde[k] > 1e-9:
            PAoI[k] = 1.0 / lam_tilde[k] + T_E2E[k]
        else:
            PAoI[k] = np.inf

    return dict(lam_k=lam_k, lam_tilde=lam_tilde, mom_U=mom_U, mom_H=mom_H,
                T_U=T_U, T_H=T_H, T_E2E=T_E2E, rho_U=rho_U, rho_H=rho_H,
                rho_U_tot=rho_U_tot, rho_H_tot=rho_H_tot, PAoI=PAoI,
                W0_U=W0_U, W0_H=W0_H,
                alpha=alpha, B_UH=B_UH, B_HG=B_HG, a=dict(a))


def cspr(cfg, res, kappa=1.0):
    lam_k = cfg.lam(kappa)
    num = sum(res['a'][k] * lam_k[k] for k in cfg.K_crit)
    den = sum(lam_k[k] for k in cfg.K_crit)
    return num / den if den > 0 else 1.0


def wss_soft(cfg, res):
    total = 0.0
    for k in cfg.classes:
        Dk = res['T_E2E'][k]
        Dmax = cfg.D_max.get(k, np.inf)
        term = max(0.0, 1.0 - Dk / Dmax) if np.isfinite(Dmax) else 1.0
        total += cfg.omega[k] * res['a'][k] * term
    return total


def gdi(cfg, res_kappa, res_nominal):
    denom = wss_soft(cfg, res_nominal)
    return wss_soft(cfg, res_kappa) / denom if denom > 0 else 0.0


def awss_soft(cfg, res):
    total = 0.0
    for k in cfg.classes:
        Dk = res['T_E2E'][k]
        Dmax = cfg.D_max.get(k, np.inf)
        term_delay = max(0.0, 1.0 - Dk / Dmax) if np.isfinite(Dmax) else 1.0
        Ak = res['PAoI'][k]
        Amax = cfg.A_max.get(k, np.inf)
        term_age = max(0.0, 1.0 - Ak / Amax) if np.isfinite(Amax) and np.isfinite(Ak) else 1.0
        total += cfg.omega[k] * res['a'][k] * term_delay * term_age
    return total


def agdi(cfg, res_kappa, res_nominal):
    denom = awss_soft(cfg, res_nominal)
    return awss_soft(cfg, res_kappa) / denom if denom > 0 else 0.0


def bottleneck_indicator(res):
    return res['rho_U_tot'] - res['rho_H_tot']


def objective_J(cfg, res, kappa=1.0):
    lam_k = cfg.lam(kappa)
    U = sum(cfg.u_weight[k] * res['a'][k] * lam_k[k] for k in cfg.classes)
    P = sum(cfg.w_delay[k] * res['T_E2E'][k] for k in cfg.classes)
    P_A = sum(cfg.xi_age[k] * res['PAoI'][k] for k in cfg.K_fresh)
    J = U - cfg.mu_D * P - cfg.mu_A * P_A
    return J, U, P


def orchestrate(cfg, kappa=1.0, alpha_grid=(0.5,), BUH_grid=(1.0e6,),
                 a1_grid=(1.0,), a2_grid=(1.0,), a3_grid=(1.0,),
                 enforce_monotone=True, channel_kwargs=None):
    channel_kwargs = channel_kwargs or {}
    best = None
    best_J = -np.inf

    for alpha in alpha_grid:
        for B_UH in BUH_grid:
            B_HG = cfg.B_tot - B_UH
            if B_UH <= 0 or B_HG <= 0:
                continue
            for a1 in a1_grid:
                for a2 in a2_grid:
                    for a3 in a3_grid:
                        if enforce_monotone and (a1 < a2 or a2 < a3):
                            continue
                        a = {1: a1, 2: a2, 3: a3}
                        res = evaluate_analytical(cfg, alpha, B_UH, B_HG, a,
                                                   kappa=kappa, **channel_kwargs)
                        if res['rho_U_tot'] >= 1.0 or res['rho_H_tot'] >= 1.0:
                            continue
                        feasible = True
                        for k in cfg.K_protected:
                            if res['T_E2E'][k] > cfg.D_max.get(k, np.inf):
                                feasible = False
                                break
                        if feasible:
                            for k in cfg.K_fresh:
                                if res['PAoI'][k] > cfg.A_max.get(k, np.inf):
                                    feasible = False
                                    break
                        if not feasible:
                            continue
                        J, U, P = objective_J(cfg, res, kappa=kappa)
                        if J > best_J:
                            best_J = J
                            res_full = dict(res)
                            res_full.update(J=J, U=U, P=P)
                            best = res_full
    return best


# ==============================================================================
# 6. PACKET-LEVEL DISCRETE-EVENT TANDEM PRIORITY SIMULATOR (Sec. IX)
# ==============================================================================
def _sample_service_UH(cfg: Config, k: int, alpha: float, B_UH: float,
                        bar_gamma_UH: float, m_UH: float, rng: np.random.Generator):
    eta_UH = 1.0 - alpha
    c = cfg.L[k] * np.log(2.0) / (eta_UH * B_UH)
    while True:
        Z = rng.gamma(shape=m_UH, scale=1.0 / m_UH)
        g = bar_gamma_UH * Z
        if g >= cfg.gamma_min:
            return c / np.log1p(g)


def _sample_service_HG(cfg: Config, k: int, B_HG: float, p_LoS: float,
                        bar_gamma_HG_LoS: float, bar_gamma_HG_NLoS: float,
                        m_HG_LoS: float, m_HG_NLoS: float,
                        rng: np.random.Generator):
    eta_HG = 1.0
    c = cfg.L[k] * np.log(2.0) / (eta_HG * B_HG)
    while True:
        is_los = rng.random() < p_LoS
        if is_los:
            Z = rng.gamma(shape=m_HG_LoS, scale=1.0 / m_HG_LoS)
            g = bar_gamma_HG_LoS * Z
        else:
            Z = rng.gamma(shape=m_HG_NLoS, scale=1.0 / m_HG_NLoS)
            g = bar_gamma_HG_NLoS * Z
        if g >= cfg.gamma_min:
            return c / np.log1p(g)


def simulate_priority_single_server(arrivals_by_class, service_sampler_by_class,
                                     classes_order):
    ptr = {k: 0 for k in classes_order}
    n_left = {k: len(arrivals_by_class[k]) for k in classes_order}
    total = sum(n_left.values())
    current_time = 0.0
    out = {k: [] for k in classes_order}
    count = 0
    while count < total:
        available = [k for k in classes_order
                     if ptr[k] < n_left[k] and arrivals_by_class[k][ptr[k]] <= current_time]
        if not available:
            next_times = [arrivals_by_class[k][ptr[k]] for k in classes_order if ptr[k] < n_left[k]]
            if not next_times:
                break
            current_time = min(next_times)
            continue
        k = min(available)
        arrival_t = arrivals_by_class[k][ptr[k]]
        service_t = service_sampler_by_class[k]()
        depart_t = current_time + service_t
        out[k].append((arrival_t, depart_t, service_t))
        ptr[k] += 1
        current_time = depart_t
        count += 1
    return out


def simulate_tandem(cfg, alpha, B_UH, B_HG, a, kappa=1.0,
                     horizon=7000.0, warmup=1000.0, seed=202601,
                     channel_kwargs=None) -> Dict:
    channel_kwargs = channel_kwargs or {}
    bar_gamma_UH = channel_kwargs.get('bar_gamma_UH', cfg.bar_gamma_UH)
    m_UH = channel_kwargs.get('m_UH', cfg.m_UH)
    p_LoS = channel_kwargs.get('p_LoS', cfg.p_LoS)
    bar_gamma_HG_LoS = channel_kwargs.get('bar_gamma_HG_LoS', cfg.bar_gamma_HG_LoS)
    bar_gamma_HG_NLoS = channel_kwargs.get('bar_gamma_HG_NLoS', cfg.bar_gamma_HG_NLoS)
    m_HG_LoS = channel_kwargs.get('m_HG_LoS', cfg.m_HG_LoS)
    m_HG_NLoS = channel_kwargs.get('m_HG_NLoS', cfg.m_HG_NLoS)

    rng = np.random.default_rng(seed)
    lam_k = cfg.lam(kappa)
    classes = cfg.classes
    T_end = horizon + warmup

    arrivals_stage1 = {}
    for k in classes:
        rate_k = a[k] * lam_k[k]
        if rate_k <= 1e-9:
            arrivals_stage1[k] = np.array([])
            continue
        n_expected = int(rate_k * T_end * 1.3) + 50
        gaps = rng.exponential(1.0 / rate_k, size=n_expected)
        t = np.cumsum(gaps)
        t = t[t <= T_end]
        arrivals_stage1[k] = np.sort(t)

    sampler_U = {k: (lambda kk=k: _sample_service_UH(cfg, kk, alpha, B_UH,
                                                       bar_gamma_UH, m_UH, rng))
                 for k in classes}
    stage1_out = simulate_priority_single_server(arrivals_stage1, sampler_U, classes)

    busy_U = 0.0
    for k in classes:
        for (arr_t, dep_t, svc_t) in stage1_out[k]:
            start_t = dep_t - svc_t
            if start_t >= warmup and dep_t <= T_end:
                busy_U += svc_t
    rho_U_sim = busy_U / horizon if horizon > 0 else np.nan

    arrivals_stage2 = {k: np.array([d for (_, d, _) in stage1_out[k]]) for k in classes}
    sampler_H = {k: (lambda kk=k: _sample_service_HG(cfg, kk, B_HG, p_LoS,
                                                       bar_gamma_HG_LoS, bar_gamma_HG_NLoS,
                                                       m_HG_LoS, m_HG_NLoS, rng))
                 for k in classes}
    stage2_out = simulate_priority_single_server(arrivals_stage2, sampler_H, classes)

    busy_H = 0.0
    for k in classes:
        for (arr_t, dep_t, svc_t) in stage2_out[k]:
            start_t = dep_t - svc_t
            if start_t >= warmup and dep_t <= T_end:
                busy_H += svc_t
    rho_H_sim = busy_H / horizon if horizon > 0 else np.nan

    results = {}
    for k in classes:
        gen_times = np.array([g for (g, _, _) in stage1_out[k]])
        deliv_times = np.array([d for (_, d, _) in stage2_out[k]])
        mask = (gen_times >= warmup) & (deliv_times <= T_end) & (gen_times <= T_end)
        g = gen_times[mask]
        d = deliv_times[mask]
        if len(g) < 2:
            results[k] = dict(E2E_mean=np.nan,
                            PAoI_mean=np.inf if len(g) == 0 else np.nan,
                            AoI_avg=np.inf if len(g) == 0 else np.nan,
                            n=len(g))
            continue
        T_E2E = d - g
        E2E_mean = float(np.mean(T_E2E))
        Y = np.diff(g)
        A_peak = Y + T_E2E[1:]
        PAoI_mean = float(np.mean(A_peak))
        D = np.diff(d)
        Tprev = T_E2E[:-1]
        area_mid = np.sum(Tprev * D + (D ** 2) / 2.0)

        area_head = T_E2E[0] * (d[0] - warmup) if d[0] > warmup else 0.0

        tail = T_end - d[-1]
        area_tail = T_E2E[-1] * tail + (tail ** 2) / 2.0

        area = area_mid + area_head + area_tail

        obs_window = horizon
        AoI_avg = float(area / obs_window) if obs_window > 0 else np.nan
        results[k] = dict(E2E_mean=E2E_mean, PAoI_mean=PAoI_mean,
                           AoI_avg=AoI_avg, n=len(g))

    results['_rho_U_sim'] = rho_U_sim
    results['_rho_H_sim'] = rho_H_sim
    return results