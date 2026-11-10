import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.optimize import linprog
from sklearn.svm import SVC
from sklearn.datasets import make_blobs, make_circles

rng = np.random.default_rng(0)  # seed
colors = np.array(["tab:blue", "tab:red"])


# makes grid
def makeGrid(X, pad=1.0, n=300):
    xmin = X[:, 0].min() - pad
    xmax = X[:, 0].max() + pad
    ymin = X[:, 1].min() - pad
    ymax = X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n))
    return xx, yy


##### PLOT 1 #####
def plot1():
    X, y = make_blobs(n_samples=60, centers=2, cluster_std=1.1, random_state=6)
    y2 = 2 * y - 1  # make labels -1 and 1 instead of 0 and 1
    model = SVC(kernel="linear", C=10000)
    model.fit(X, y)
    w = model.coef_[0]
    b = model.intercept_[0]
    xx, yy = makeGrid(X)
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))

    # random lines
    angle = np.arctan2(w[1], w[0])
    count = 0
    tries = 0
    while count < 10 and tries < 50000:  # dont want infinite loop
        tries = tries + 1
        a = angle + rng.uniform(-0.7, 0.7)
        wr = np.array([np.cos(a), np.sin(a)])
        temp = X @ wr
        br = -rng.uniform(temp.min(), temp.max())
        # check if it works
        if np.all(y2 * (temp + br) > 0):
            Z = (grid @ wr + br).reshape(xx.shape)
            ax[0].contour(xx, yy, Z, levels=[0], colors="gray", alpha=0.6)
            count = count + 1
    ax[0].scatter(X[:, 0], X[:, 1], c=colors[y], edgecolor="k", s=40)
    ax[0].set_title(str(count) + " different lines separate the data perfectly\nWhich one generalizes best?")
    ax[0].set_xlabel("$x_1$")
    ax[0].set_ylabel("$x_2$")

    # svm
    Z = model.decision_function(grid).reshape(xx.shape)
    ax[1].contourf(xx, yy, Z, levels=[-1, 1], colors=["gold"], alpha=0.25)
    ax[1].contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])
    ax[1].scatter(X[:, 0], X[:, 1], c=colors[y], edgecolor="k", s=40)
    sv = model.support_vectors_
    ax[1].scatter(sv[:, 0], sv[:, 1], s=220, facecolors="none", edgecolors="lime", linewidths=2.5, label="support vectors")
    m = 2 / np.linalg.norm(w)  # margin
    ax[1].set_title("SVM: maximum margin = 2/||w|| = " + str(round(m, 2)) + "\nOnly " + str(len(sv)) + " of " + str(len(X)) + " points determine the boundary")
    ax[1].legend(loc="upper right")
    ax[1].set_xlabel("$x_1$")
    ax[1].set_ylabel("$x_2$")

    fig.suptitle("1) Max margin: SVM picks the most robust linear boundary", fontsize=14, weight="bold")
    fig.tight_layout()
    return fig


##### PLOT 2 #####
def phi1(x):
    return np.c_[x, x**2]

def plot2():
    n = 25
    inside = rng.uniform(-1.0, 1.0, n)
    #outside = rng.uniform(-3, 3, n)
    outside = np.r_[rng.uniform(-3, -1.7, n // 2), rng.uniform(1.7, 3, n - n // 2)]
    x = np.r_[inside, outside]
    y = np.r_[np.zeros(n, int), np.ones(n, int)]

    model = SVC(kernel="linear", C=10000)
    model.fit(phi1(x), y)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # 1d plot
    ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].scatter(x, np.zeros(len(x)), c=colors[y], edgecolor="k", s=50, zorder=3)
    ax[0].set_yticks([])
    ax[0].set_ylim(-1, 1)
    ax[0].set_title("1D: no single threshold can separate the classes")
    ax[0].set_xlabel("x")

    # 2d plot
    P = phi1(x)
    xx, yy = np.meshgrid(np.linspace(-3.3, 3.3, 300), np.linspace(-0.8, 10, 300))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax[1].contourf(xx, yy, Z, levels=[-1, 1], colors=["gold"], alpha=0.25)
    ax[1].contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])
    t = np.linspace(-3.3, 3.3, 200)
    ax[1].plot(t, t**2, color="gray", lw=0.8, alpha=0.6)  # parabola
    ax[1].scatter(P[:, 0], P[:, 1], c=colors[y], edgecolor="k", s=50, zorder=3)
    sv = model.support_vectors_
    ax[1].scatter(sv[:, 0], sv[:, 1], s=220, facecolors="none", edgecolors="lime", linewidths=2.5, zorder=4)
    ax[1].set_title(r"2D: $\phi(x) = (x, x^2)$ makes it linearly separable")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("$x^2$")

    # back to 1d
    t = np.linspace(-3.3, 3.3, 1000)
    f = model.decision_function(phi1(t))
    ax[2].fill_between(t, -1, 1, where=f < 0, color="tab:blue", alpha=0.15, step="mid")
    ax[2].fill_between(t, -1, 1, where=f >= 0, color="tab:red", alpha=0.15, step="mid")
    ax[2].axhline(0, color="gray", lw=0.8)
    ax[2].scatter(x, np.zeros(len(x)), c=colors[y], edgecolor="k", s=50, zorder=3)
    # find where sign changes
    idx = np.where(np.diff(np.sign(f)))[0]
    bounds = t[idx]
    for i in range(len(bounds)):
        ax[2].axvline(bounds[i], color="k", lw=2)
    ax[2].set_yticks([])
    ax[2].set_ylim(-1, 1)
    ax[2].set_title("Back in 1D: the line becomes two thresholds\n(x = %.2f and x = %.2f)" % (bounds[0], bounds[-1]))
    ax[2].set_xlabel("x")

    fig.suptitle("2) From 1D to 2D: a linear boundary in a higher space is non-linear in the original", fontsize=14, weight="bold")
    fig.tight_layout()
    return fig


##### PLOT 3 #####
def phi2(X):
    return np.c_[X[:, 0], X[:, 1], X[:, 0]**2 + X[:, 1]**2]

def plot3():
    X, y = make_circles(n_samples=200, factor=0.4, noise=0.08, random_state=1)
    P = phi2(X)
    model = SVC(kernel="linear", C=1000)
    model.fit(P, y)
    w = model.coef_[0]
    b = model.intercept_[0]

    fig = plt.figure(figsize=(17, 5.5))

    # 2d
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(X[:, 0], X[:, 1], c=colors[y], edgecolor="k", s=30)
    ax1.set_aspect("equal")
    ax1.set_title("2D: no straight line can separate the rings")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")

    # 3d
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(P[:, 0], P[:, 1], P[:, 2], c=colors[y], edgecolor="k", s=25)
    gx, gy = np.meshgrid(np.linspace(-1.3, 1.3, 20), np.linspace(-1.3, 1.3, 20))
    gz = -(w[0] * gx + w[1] * gy + b) / w[2]  # solve plane for z
    ax2.plot_surface(gx, gy, gz, color="gold", alpha=0.35)
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_zlabel("$x_1^2 + x_2^2$")
    ax2.view_init(elev=15, azim=35)
    ax2.set_title(r"3D: $z = x_1^2 + x_2^2$, now a plane separates the classes")

    # back to 2d
    ax3 = fig.add_subplot(1, 3, 3)
    xx, yy = makeGrid(X, pad=0.3)
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(phi2(grid)).reshape(xx.shape)
    ax3.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], colors=["tab:blue", "tab:red"], alpha=0.15)
    ax3.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])
    ax3.scatter(X[:, 0], X[:, 1], c=colors[y], edgecolor="k", s=30)
    ax3.set_aspect("equal")
    ax3.set_title("Back in 2D: the plane becomes a circle")
    ax3.set_xlabel("$x_1$")
    ax3.set_ylabel("$x_2$")

    fig.suptitle("3) From 2D to 3D: same idea, one level up", fontsize=14, weight="bold")
    fig.tight_layout()
    return fig


##### PLOT 4 #####
# covers theorem
def cover(N, d):
    D = d + 1  # +1 because of bias
    k = np.arange(min(D, N))
    return 2 * comb(N - 1, k).sum() / 2.0**N


# test it with LP (same constraints as hard margin svm)
def test_sep(N, d, runs=200):
    ok = 0
    for i in range(runs):
        X = rng.normal(size=(N, d))
        y = 2 * rng.integers(0, 2, N) - 1
        A = -y[:, None] * np.c_[X, np.ones(N)]
        res = linprog(np.zeros(d + 1), A_ub=A, b_ub=-np.ones(N), bounds=[(None, None)] * (d + 1), method="highs")
        if res.status == 0:
            ok += 1
    return ok / runs


def plot4():
    dims = [1, 2, 3, 5, 10, 20]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    # left
    for d in dims:
        Ns = np.arange(1, 100)
        p = []
        for N in Ns:
            p.append(cover(N, d))
        ax[0].plot(Ns, p, label="d = " + str(d), lw=2)
    # simulation
    for d, mark in [(2, "o"), (5, "s")]:
        Ns = np.arange(2, 25, 2)
        res = []
        for N in Ns:
            res.append(test_sep(N, d))
        ax[0].scatter(Ns, res, marker=mark, color="k", zorder=5, s=30, label="SVM test, d = " + str(d))
    ax[0].set_xlabel("Number of points N")
    ax[0].set_ylabel("P(linearly separable)")
    ax[0].set_title("More dimensions: many more points can be separated\neven with completely random labels")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # right (normalized)
    for d in dims:
        Ns = np.arange(1, 4 * (d + 1) + 1)
        p = []
        for N in Ns:
            p.append(cover(N, d))
        ax[1].plot(Ns / (d + 1), p, label="d = " + str(d), lw=2)
    ax[1].axvline(2, color="k", ls="--")
    ax[1].text(2.05, 0.85, "N = 2(d+1)\ncapacity", fontsize=10)
    ax[1].set_xlabel("N / (d + 1)")
    ax[1].set_ylabel("P(linearly separable)")
    ax[1].set_title("Normalized: the transition is always at N = 2(d+1)\nand gets sharper the higher d is")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.suptitle("4) Cover's theorem: in high dimensions linear separation is almost free", fontsize=14, weight="bold")
    fig.tight_layout()
    return fig


# run everything
f1 = plot1()
f1.savefig("1_max_margin.png", dpi=120, bbox_inches="tight")
f2 = plot2()
f2.savefig("2_1d_to_2d.png", dpi=120, bbox_inches="tight")
f3 = plot3()
f3.savefig("3_2d_to_3d.png", dpi=120, bbox_inches="tight")
f4 = plot4()
f4.savefig("4_cover.png", dpi=120, bbox_inches="tight")
plt.show()
