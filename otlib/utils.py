import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.sparse import lil_matrix

def plot_1D_transport(a, x, b, y, P):

    fig, ax = plt.subplots(figsize=(7, 7))

    nx = len(x)
    ny = len(y)

    # Espacement entre les positions
    dx = np.min(np.diff(np.sort(x))) if nx > 1 else 1
    dy = np.min(np.diff(np.sort(y))) if ny > 1 else 1

    d = min(dx, dy)

    # Rayon maximal des cercles
    r_max = 0.5 * d

    # Fonction pour ajouter un cercle
    def add_circle(x0, y0, value, color, vmax):
        if value > 0:
            radius = r_max * value/vmax
            circle = Circle((x0, y0), radius, color=color)
            ax.add_patch(circle)

    
    vmax = max(np.max(P), np.max(a), np.max(b))

    # Retourner x et shift (x,y)
    origin = min(np.min(x), np.min(y))
    x = x - origin
    y = y - origin
    x_max = np.max(x)
    x = x_max - x

    # Cercles associés à P
    for j in range(ny):
        for i in range(nx):
            add_circle(y[j], x[i], P[i,j], 'black', vmax)

    # Cercles associés à a (à gauche)
    y_left = min(y) - 1.5 * d

    for i in range(nx):
        add_circle(y_left, x[i], a[i], 'red', vmax)

    # Cercles associés à b (en haut)
    x_top = x_max + 1.5 * d

    for j in range(ny):
        add_circle(y[j], x_top, b[j], 'blue', vmax)

    # Grille
    xmin_plot = (y_left+0.8*d)
    xmax_plot = max(y)+0.5*d
    ymin_plot = -0.7*d
    ymax_plot = x_max + 0.7*d
    
    for xi in x:
        ax.hlines(xi, xmin=y_left, xmax=xmax_plot, color='gray', linestyle='--', linewidth=0.5)    
    ax.hlines(ymax_plot, xmin=xmin_plot, xmax=xmax_plot, color='black', linestyle = '-')
    ax.hlines(ymin_plot, xmin=xmin_plot, xmax=xmax_plot, color='black', linestyle = '-')

    for yj in y:
        ax.vlines(yj, ymin=ymin_plot, ymax=x_top, color='gray', linestyle='--', linewidth=0.5)
    ax.vlines(xmin_plot, ymin=ymin_plot, ymax=ymax_plot, color='black', linestyle='-')
    ax.vlines(xmax_plot, ymin=ymin_plot, ymax=ymax_plot, color='black', linestyle='-')

    # Limites et apparence
    ax.set_xlim(y_left - d, max(y) + d)
    ax.set_ylim(min(x) - d, x_top + d)

    ax.set_aspect('equal')
    ax.axis('off')

    plt.show()

def plot_2D_transport(a,x,b,y,P):
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, j in np.argwhere(P > 1e-9):
        start, end = x[i], y[j]
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color="gray", alpha=0.5, linewidth=60 * P[i, j])

    ax.scatter(x[:, 0], x[:, 1], s=1500 * a, color="tab:blue", label="Sources")
    ax.scatter(y[:, 0], y[:, 1], s=1500 * b, color="brown", marker="s", label="Destinations")
    ax.set(xlabel="x position", ylabel="y position")
    ax.set_aspect("equal")
    ax.legend()
    plt.show()



def plot_hist_marg(hist_marg):
    idx = np.arange(len(hist_marg[0]))
    plt.plot(idx, np.log(hist_marg[0]), label='Marginal violation in a')
    plt.plot(idx, np.log(hist_marg[1]), label='Marginal violation in b')
    plt.title("Marginal violation of iterate plan")
    plt.legend()
    plt.show()


def compare_hist_marg(hists_marg, list_eps):
    for k in range(len(list_eps)):
        idx = np.arange(len(hists_marg[k]))
        plt.plot(idx, np.log(hists_marg[k]), label=f"Epsilon = {list_eps[k]}")
    plt.title("Marginal violation for different epsilon")
    plt.legend()
    plt.show()