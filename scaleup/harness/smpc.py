"""Secret-shared honeypot check (SWARM-30). The bait list is Shamir-shared over three servers (threshold t=1), so no
single server learns which offers are bait. Given the agent's public accept/refuse choices, the servers compute
flag = [number of accepted bait >= tau] with BGW multiplication and reveal only the flag, not the count.
Honest-but-curious servers; with n=4 the same code tolerates one server that lies only if shares are checked
(not done here)."""
import random

P = 2_147_483_647  # field prime (2^31 - 1)


def share(x, n=3, t=1, rng=random):
    """Shamir shares of x: f(0)=x, f random of degree t, share j is f(j+1)."""
    c = [x % P] + [rng.randrange(P) for _ in range(t)]
    return [sum(ci * pow(j + 1, k, P) for k, ci in enumerate(c)) % P for j in range(n)]


def lagrange0(xs):
    """Weights w_j with f(0) = sum_j w_j f(xs[j])."""
    w = []
    for j, xj in enumerate(xs):
        num = den = 1
        for m, xm in enumerate(xs):
            if m != j:
                num = num * (-xm) % P
                den = den * (xj - xm) % P
        w.append(num * pow(den, P - 2, P) % P)
    return w


def reconstruct(sh):
    xs = list(range(1, len(sh) + 1))
    return sum(w * s for w, s in zip(lagrange0(xs), sh)) % P


def mul(a, b, t=1, rng=random):
    """BGW multiplication: local products lie on a degree-2t polynomial; each server reshares its product at degree t
    and every server combines the subshares with the Lagrange weights. Needs n >= 2t + 1."""
    n = len(a)
    assert n >= 2 * t + 1
    sub = [share(x * y, n, t, rng) for x, y in zip(a, b)]  # sub[i][j]: server i's product, subshare for server j
    w = lagrange0(list(range(1, n + 1)))
    return [sum(w[i] * sub[i][j] for i in range(n)) % P for j in range(n)]


def threshold_poly(n_max, tau):
    """Coefficients of the polynomial q with q(c) = 1 if c >= tau else 0 for c in 0..n_max (Lagrange over the field)."""
    xs = list(range(n_max + 1))
    coef = [0] * (n_max + 1)
    for j, xj in enumerate(xs):
        if xj < tau:
            continue
        basis, den = [1], 1
        for m, xm in enumerate(xs):
            if m != j:
                basis = [((basis[k - 1] if k else 0) - xm * (basis[k] if k < len(basis) else 0)) % P for k in range(len(basis) + 1)]
                den = den * (xj - xm) % P
        inv = pow(den, P - 2, P)
        coef = [(c + b * inv) % P for c, b in zip(coef, basis)]
    return coef


def flag(bait_shares, accepted, tau, rng=random):
    """bait_shares[i]: the 3 shares of offer i's bait bit; accepted[i]: public 0/1 choice. Returns the revealed flag
    and the per-server views (what each server ever holds) for leak tests."""
    n = len(bait_shares[0])
    count = [sum(a * s[j] for a, s in zip(accepted, bait_shares)) % P for j in range(n)]  # linear: no interaction
    q = threshold_poly(len(accepted), tau)
    acc = [q[-1]] * n  # Horner on shares: acc = acc * count + q_k
    for qk in reversed(q[:-1]):
        acc = [(x + qk) % P for x in mul(acc, count, rng=rng)]
    return reconstruct(acc), count


if __name__ == "__main__":
    rng = random.Random(0)
    for trial in range(200):
        k = rng.randint(4, 12)
        bait = [rng.randint(0, 1) for _ in range(k)]
        acc = [rng.randint(0, 1) for _ in range(k)]
        tau = rng.randint(1, k)
        sh = [share(b, rng=rng) for b in bait]
        f, count_shares = flag(sh, acc, tau, rng)
        assert f == int(sum(a * b for a, b in zip(acc, bait)) >= tau), trial
    # one share alone is uniform: the share of bait=0 and of bait=1 have the same distribution at any one server
    z = [share(0, rng=rng)[0] % 4 for _ in range(4000)]
    o = [share(1, rng=rng)[0] % 4 for _ in range(4000)]
    print("200 flag checks passed; one-server share histogram (mod 4)",
          [z.count(v) for v in range(4)], [o.count(v) for v in range(4)])
