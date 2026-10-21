"""Round 10, invent stance: numerical checks of herded argmax (HA), the decode
this stance proposes, on synthetic probability streams (no game, no network).

HA over a recurrent option set O (a faction's recruit types). At each decision
of the class, q_t is the network's distribution over the legal options
(renormalised over them); the decoder holds W(o), zero at game start:
    W(o) += q_t(o) for legal o;  o* = argmax over legal o of W(o) (ties: larger q_t);
    W(o*) -= 1.
Tempered HA uses q_t^(1/T) renormalised; T -> 0 is argmax.

Checks:
1. fixed weights {5, 1, 1}/7: the sequence equals nginx's smooth weighted
   round-robin example (a a b a c a a);
2. without masks, max over o and t of |n_t(o) - Q_t(o)| stays below |O| - 1
   (the derived bound) for random streams, Dirichlet q with concentration 0.3..3;
3. with random masks (each option illegal with probability 0.2 at a decision,
   W frozen while illegal): the same maximum, measured;
4. tempered HA at T 0.05 and 0.2: how often it departs from argmax, by the ratio r of the top two
   probabilities (small T departs only near ties), and the share of decisions where T = 1 HA departs from
   argmax for q drawn near a fixed mixture (a stand-in for a stable recruit
   preference: q_t = Dirichlet(50 m), m = (0.5, 0.3, 0.2)).
numpy only.
"""
import numpy as np


def herd(qs, masks=None, T=1.0):
    k = qs.shape[1]
    W = np.zeros(k)
    n = np.zeros(k)
    Q = np.zeros(k)
    worst = 0.0
    picks = []
    for t, q in enumerate(qs):
        legal = np.ones(k, bool) if masks is None else masks[t]
        if not legal.any():
            picks.append(-1)
            continue
        p = np.where(legal, q, 0.0)
        p = p / p.sum()
        if T != 1.0:
            lp = np.where(legal, np.log(np.maximum(p, 1e-300)) / T, -np.inf)
            lp -= lp.max()
            p = np.where(legal, np.exp(lp), 0.0)
            p /= p.sum()
        W[legal] += p[legal]
        Q += p
        cand = np.flatnonzero(legal)
        best = cand[np.lexsort((-p[cand], -W[cand]))[0]]
        W[best] -= 1.0
        n[best] += 1
        picks.append(int(best))
        worst = max(worst, float(np.max(np.abs(n - Q))))
    return np.array(picks), worst


def main():
    rng = np.random.default_rng(20261021)
    picks, _ = herd(np.tile(np.array([5, 1, 1]) / 7.0, (7, 1)))
    print("1. {5,1,1}/7:", " ".join("abc"[i] for i in picks), "(nginx: a a b a c a a)")

    k = 7
    for conc in (0.3, 1.0, 3.0):
        worst = 0.0
        for _ in range(200):
            qs = rng.dirichlet(np.full(k, conc), size=500)
            worst = max(worst, herd(qs)[1])
        print(f"2. no masks, Dirichlet {conc}: max |n - Q| over 200 streams x 500 decisions {worst:.3f} (bound {k - 1})")
    for conc in (0.3, 1.0, 3.0):
        worst = 0.0
        for _ in range(200):
            qs = rng.dirichlet(np.full(k, conc), size=500)
            masks = rng.random((500, k)) > 0.2
            worst = max(worst, herd(qs, masks)[1])
        print(f"3. masks p 0.2, Dirichlet {conc}: max |n - Q| {worst:.3f}")

    for T in (0.05, 0.2):
        tally = {"r >= 1.5": [0, 0], "1.05 <= r < 1.5": [0, 0], "r < 1.05": [0, 0]}
        for _ in range(200):
            qs = rng.dirichlet(np.full(k, 1.0), size=200)
            top2 = np.sort(qs, 1)[:, -2:]
            r = top2[:, 1] / top2[:, 0]
            p_ha, _ = herd(qs, T=T)
            dep = p_ha != qs.argmax(1)
            for lab, m_ in (("r >= 1.5", r >= 1.5), ("1.05 <= r < 1.5", (r >= 1.05) & (r < 1.5)), ("r < 1.05", r < 1.05)):
                tally[lab][0] += int(np.sum(dep & m_))
                tally[lab][1] += int(m_.sum())
        print(f"4a. tempered HA at T {T}: departs from argmax at " + "; ".join(
            f"{a}/{b} decisions with top-two ratio {lab}" for lab, (a, b) in tally.items()))
    m = np.array([0.5, 0.3, 0.2])
    dep = cnt = 0
    for _ in range(200):
        qs = rng.dirichlet(50 * m, size=5)
        p_ha, _ = herd(qs)
        dep += int(np.sum(p_ha != qs.argmax(1)))
        cnt += 5
        picks = p_ha
    print(f"4b. q near (0.5, 0.3, 0.2), 5 decisions (a turn-1 recruit phase): HA departs from argmax at "
          f"{dep / cnt:.3f} of decisions; argmax takes the modal type at 5 of 5, HA's last stream {picks.tolist()}")


if __name__ == "__main__":
    main()
