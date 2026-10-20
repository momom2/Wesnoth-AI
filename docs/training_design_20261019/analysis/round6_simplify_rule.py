"""Round 6, simplify stance: is the calibration slope beta_hat needed once the ridge's lambda is
chosen by progressive validation?  And what does a threshold fallback keep against the
conservative mixture?

Model: the stated Gaussian model of round5_simplify_power.py / round5_synthesis_power.py, made
multivariate so that lambda selection and beta_hat are simulated instead of assumed.
- Features x ~ N(0, Sigma), Sigma diagonal, d = 776, trace 1 (isotropic, or eigenvalues
  proportional to 1/j). Centred features, one pair type, so the intercept separates.
- True advantage at a band decision a = mu + w'x, w ~ N(0, v I): Var(w'x) is v on average.
- Labels T = a + eps, Var(eps) = sigma^2 = 3.91 (round5_synthesis_power.out at p 0.25).
- A batch holds n = 2 x 76.1 x 100,000 labels (round5_synthesis_power.py's band count).
  Its sufficient statistics are G = n Sigma (the law of large numbers at n = 1.5e7 >> d) and
  r ~ N(G w, sigma^2 G) (exact for Gaussian noise given the features). The intercept estimate
  after k batches is mu + N(0, sigma^2 / (k n)).
- Student k: ridge on batches 1..k, w_hat(lambda) = (k G)^-1-shrunk, lambda chosen on a grid
  by the summed progressive loss of students 1..k-1 on the batch after each
  (loss_j = -2 w_j' r_{j+1} + w_j' G w_j, constant terms dropped); student 1 by a split of
  batch 1 into halves. This is computable from per-batch sufficient statistics.
- beta_hat: the slope of T on the student's centred prediction over the progressive readings
  of students 4..7 on batches 5..8: sum_j w_j' r_{j+1} / sum_j w_j' G w_j.
Rules for the final student (8 batches):
  A  switch where mu_hat + w_hat' x > 0                         (this stance)
  B  switch where mu_hat + beta_hat w_hat' x > 0               (the current design)
  C  the posterior mean at the true lambda = sigma^2 / v, mu_hat  (oracle tuning)
  E  rule A with a threshold that halves A's switches          (fallback candidate)
Gain per band decision: E[a 1(c0 + c'x > 0)] = mu Phi(c0/s) + (w' Sigma c / s) phi(c0/s),
s^2 = c' Sigma c. Elo per side-game = 347.4 x 76.1 x gain. numpy and the standard library.
"""
import math

import numpy as np

D, SIGMA2, N_BAND, BATCHES = 776, 3.91, 76.1, 8
N = 2 * N_BAND * 100_000
ELO = 800 / math.log(10) * N_BAND
LAMBDAS = np.geomspace(1e2, 1e9, 57)
REPS = 300
RNG = np.random.default_rng(20261020)


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def gain(mu, w, s, c0, c):
    var = float(np.sum(s * c * c))
    if var <= 0:
        return (mu if c0 > 0 else 0.0), float(c0 > 0)
    sd = math.sqrt(var)
    cov = float(np.sum(s * w * c))
    return mu * Phi(c0 / sd) + cov / sd * phi(c0 / sd), Phi(c0 / sd)


def ridge(R, n_total, s, lam):
    return R / (n_total * s + lam)


def one_rep(mu, v, s):
    w = RNG.normal(0, math.sqrt(v), D)
    noise_sd = np.sqrt(SIGMA2 * N * s)
    r = [N * s * w + noise_sd * RNG.normal(size=D) for _ in range(BATCHES)]
    # student 1's lambda: halves of batch 1
    half_sd = np.sqrt(SIGMA2 * N / 2 * s)
    h1 = N / 2 * s * w + half_sd * RNG.normal(size=D)
    h2 = r[0] - h1
    loss = np.zeros(len(LAMBDAS))
    for li, lam in enumerate(LAMBDAS):
        wh = ridge(h1, N / 2, s, lam)
        loss[li] = -2 * wh @ h2 + wh @ (N / 2 * s * wh)
    lam_by_student = {1: LAMBDAS[int(np.argmin(loss))]}
    cum = np.zeros(len(LAMBDAS))
    num = den = 0.0
    for k in range(1, BATCHES):
        Rk = sum(r[:k])
        for li, lam in enumerate(LAMBDAS):
            wh = ridge(Rk, k * N, s, lam)
            cum[li] += -2 * wh @ r[k] + wh @ (N * s * wh)
        lam_by_student[k + 1] = LAMBDAS[int(np.argmin(cum))]
        wk = ridge(Rk, k * N, s, lam_by_student[k])
        if k >= 4:
            num += wk @ r[k]
            den += wk @ (N * s * wk)
    beta_hat = num / den if den > 0 else 0.0
    R8 = sum(r)
    mu_hat = mu + RNG.normal(0, math.sqrt(SIGMA2 / (BATCHES * N)))
    w8 = ridge(R8, BATCHES * N, s, lam_by_student[BATCHES])
    out = {}
    out["A"] = gain(mu, w, s, mu_hat, w8)
    out["B"] = gain(mu, w, s, mu_hat, beta_hat * w8)
    lam_true = SIGMA2 / v if v > 0 else LAMBDAS[-1]
    out["C"] = gain(mu, w, s, mu_hat, ridge(R8, BATCHES * N, s, lam_true))
    # E: threshold halving A's switch fraction
    frac_a = out["A"][1]
    sd_f = math.sqrt(float(np.sum(s * w8 * w8)))
    if sd_f > 0 and frac_a > 0:
        lo, hi = -10.0, 10.0
        for _ in range(60):
            t = 0.5 * (lo + hi)
            if Phi((mu_hat - t) / sd_f) > frac_a / 2:
                lo = t
            else:
                hi = t
        out["E"] = gain(mu, w, s, mu_hat - t, w8)
    else:
        out["E"] = (0.0, 0.0)
    return out, beta_hat, lam_by_student[BATCHES]


def main():
    spectra = {"isotropic": np.full(D, 1.0 / D)}
    pl = 1.0 / np.arange(1, D + 1)
    spectra["1/j"] = pl / pl.sum()
    print(f"labels per batch {N:,.0f}; sigma^2 {SIGMA2}; {REPS} replications per case; "
          "gains in Elo per side-game (mean, 10th percentile); switches per side-game")
    for name, s in spectra.items():
        print(f"\nspectrum {name}")
        for mu in (0.0, -0.0018):
            for sd in (0.0, 0.001, 0.002, 0.003, 0.005):
                v = sd * sd
                rows = {k: [] for k in "ABCE"}
                betas = []
                for _ in range(REPS):
                    out, b, _ = one_rep(mu, v, s)
                    for k in "ABCE":
                        rows[k].append(out[k])
                    betas.append(b)
                line = f"  mu {mu:+.4f} sd {sd:.3f}:"
                for k in "ABCE":
                    g = np.array([x[0] for x in rows[k]]) * ELO
                    sw = np.array([x[1] for x in rows[k]]) * N_BAND
                    line += f" {k} {g.mean():5.1f} ({np.percentile(g, 10):5.1f}) sw {sw.mean():4.1f} |"
                betas = np.array(betas)
                line += f" beta_hat median {np.median(betas):.2f}, P(beta_hat<=0) {np.mean(betas <= 0):.2f}"
                print(line)


if __name__ == "__main__":
    main()
