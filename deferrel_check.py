import numpy as np
from scipy.special import gammaincc, gamma as gamma_func

L = {1: 100000, 2: 200000, 3: 400000}     # bits, Table baseline
lam0 = {1: 0.4, 2: 0.6, 3: 0.8}           # pkt/s, Table baseline
alpha = 0.5
B_UH = 1.0e6
gamma_min = 0.1
Q_quad = 64
Tc = 0.002                                 # 2 ms coherence block


def outage_probability(bar_gamma, m, gamma_min):
    a = m * gamma_min / bar_gamma
    return 1.0 - gammaincc(m, a)


def truncated_service_moments(L_k, eta_ij, B_ij, bar_gamma, m, gamma_min, Q=64):
    c = L_k * np.log(2.0) / (eta_ij * B_ij)
    a = m * gamma_min / bar_gamma
    p_suc = max(gammaincc(m, a), 1e-12)

    nodes, weights = np.polynomial.laguerre.laggauss(Q)
    t = a + nodes
    log_term = np.clip(np.log1p((bar_gamma / m) * t), 1e-12, None)
    pref = np.exp(-a) / (p_suc * gamma_func(m))
    base = t ** (m - 1.0)

    I1 = pref * np.sum(weights * base / log_term)
    I2 = pref * np.sum(weights * base / (log_term ** 2))
    return c * I1, (c ** 2) * I2


def paired_sample_S(L_k, eta_ij, B_ij, bar_gamma, m, gamma_min, Tc, rng):
    c = L_k * np.log(2.0) / (eta_ij * B_ij)
    n_defer = 0
    while True:
        Z = rng.gamma(shape=m, scale=1.0 / m)
        g = bar_gamma * Z
        if g >= gamma_min:
            S_rej = c / np.log1p(g)
            S_def = S_rej + n_defer * Tc
            return S_rej, S_def, n_defer
        n_defer += 1


def run_config(label, bar_gamma_UH, m_UH, n_mc=2_000_000, seed=202601):
    rng = np.random.default_rng(seed)
    p_out = outage_probability(bar_gamma_UH, m_UH, gamma_min)

    print(f"\n{'='*78}\n{label}: bar_gamma_UH={bar_gamma_UH}, m_UH={m_UH}\n{'='*78}")
    print(f"  P(outage) = {p_out*100:.4f}%")

    eta_UH = 1.0 - alpha
    results = {}
    for k in (1, 2, 3):
        E_S_ana, _ = truncated_service_moments(
            L[k], eta_UH, B_UH, bar_gamma_UH, m_UH, gamma_min, Q_quad)

        S_rej_arr = np.empty(n_mc)
        S_def_arr = np.empty(n_mc)
        N_def_arr = np.empty(n_mc)
        for i in range(n_mc):
            s_rej, s_def, nd = paired_sample_S(L[k], eta_UH, B_UH, bar_gamma_UH,
                                                m_UH, gamma_min, Tc, rng)
            S_rej_arr[i] = s_rej
            S_def_arr[i] = s_def
            N_def_arr[i] = nd

        E_S_rej_mc = S_rej_arr.mean()
        E_S_def_mc = S_def_arr.mean()
        E_Ndef = N_def_arr.mean()
        E_extra_wait = E_S_def_mc - E_S_rej_mc   # exact paired difference
        overhead_pct = E_extra_wait / E_S_rej_mc * 100.0

        results[k] = dict(E_S_ana=E_S_ana, E_S_rej_mc=E_S_rej_mc,
                           E_S_def_mc=E_S_def_mc, E_Ndef=E_Ndef,
                           E_extra_wait=E_extra_wait, overhead_pct=overhead_pct)

        print(f"  class {k}: E[S]_ana(rejection)={E_S_ana*1e3:.6f} ms | "
              f"E[S]_MC(rejection)={E_S_rej_mc*1e3:.6f} ms | "
              f"E[S]_MC(deferral)={E_S_def_mc*1e3:.6f} ms | "
              f"E[N_defer]={E_Ndef:.6f} | "
              f"E[extra wait]={E_extra_wait*1e3:.6f} ms | "
              f"overhead={overhead_pct:.5f}%")

    rho_rej = sum(lam0[k] * results[k]["E_S_rej_mc"] for k in (1, 2, 3))
    rho_def = sum(lam0[k] * results[k]["E_S_def_mc"] for k in (1, 2, 3))
    print(f"\n  rho_U (rejection model, a_k=1) = {rho_rej:.6f}")
    print(f"  rho_U (deferral model,   a_k=1) = {rho_def:.6f}")
    print(f"  Delta rho_U = {rho_def - rho_rej:.6e}")

    return dict(p_out=p_out, per_class=results, rho_rej=rho_rej, rho_def=rho_def,
                delta_rho=rho_def - rho_rej)


if __name__ == "__main__":
    baseline = run_config("BASELINE", bar_gamma_UH=10.0, m_UH=2.0)
    pessimistic = run_config("PESSIMISTIC", bar_gamma_UH=2.0, m_UH=1.0)

    print("\n" + "=" * 78)
    print("SUMMARY (for paper paragraph)")
    print("=" * 78)
    for label, res in [("baseline", baseline), ("pessimistic", pessimistic)]:
        max_overhead = max(v["overhead_pct"] for v in res["per_class"].values())
        print(f"{label}: P(outage)={res['p_out']*100:.4f}%  "
              f"max class overhead={max_overhead:.5f}%  "
              f"Delta rho_U={res['delta_rho']:.6e}")