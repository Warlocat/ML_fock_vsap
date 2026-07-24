# -*- coding: utf-8 -*-
"""
3x3 panels (rows = molecules, cols = distance 4/5/6 A), produced as TWO
separate figures:
    twoJ_vs_geometry_3col.{png,pdf}   -> 2|t|
    e1e2_vs_geometry_3col.{png,pdf}   -> |e1 - e2|

Data come from the per-molecule validation CSVs:
    benzene_b5_validation.csv, ttf_b2_validation.csv, tcnq_b2_validation.csv
with columns: geom, e1_QM, e1_ML, e2_QM, e2_ML, 2J_QM, 2J_ML,
              absE1E2_QM, absE1E2_ML, ... (all meV).

There is no explicit angle/distance column; rows are read in order as three
consecutive blocks (4 A, then 5 A, then 6 A), and within a block the angle
steps in 10-deg increments (set per molecule in MOLECULES).

Each panel: DFT (QM) = blue solid line, ML = red thin line + dots.
Per-column y-limits are shared across molecules (auto-computed per quantity).
"""
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
UP = '/mnt/user-data/uploads'

# (key, display label, csv path, angle grid for each distance block)
MOLECULES = [
    ('benzene', 'Benzene', os.path.join(UP, 'benzene_b5_validation.csv'),
     list(range(-30, 91, 10))),
    ('ttf', 'TTF', os.path.join(UP, 'ttf_b2_validation.csv'),
     list(range(-90, 91, 10))),
    ('tcnq', 'TCNQ', os.path.join(UP, 'tcnq_b2_validation.csv'),
     list(range(-90, 91, 10))),
]
DISTANCES = [(4, u'4 \u00c5'), (5, u'5 \u00c5'), (6, u'6 \u00c5')]  # block order

# Two figures, one per quantity.
QUANTITIES = [
    {'qm': '2J_QM', 'ml': '2J_ML', 'label': u'2|t| (meV)',
     'scale': {'benzene': 0.5}, 'out': 'twoJ_vs_geometry_3col',
     'ylim': {5: (0, 200), 6: (0, 100)}},          # 4 A left auto
    {'qm': 'absE1E2_QM', 'ml': 'absE1E2_ML', 'label': u'|e$_1$ - e$_2$| (meV)',
     'scale': {}, 'out': 'e1e2_vs_geometry_3col',
     'ylim': {}},                                   # all auto
]

C_DFT = '#1f77b4'
C_ML = '#d62728'
XPAD = 5


def read_molecule(path, angles):
    """{distance: {'angle', '2J_QM', '2J_ML', 'absE1E2_QM', 'absE1E2_ML'}}."""
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    n = len(angles)
    need = n * len(DISTANCES)
    if len(rows) < need:
        raise ValueError(f'{os.path.basename(path)}: expected >= {need} rows, '
                         f'got {len(rows)}')
    out = {}
    for b, (R, _) in enumerate(DISTANCES):
        block = rows[b*n:(b+1)*n]
        d = {'angle': np.array(angles, dtype=float)}
        for key in ('2J_QM', '2J_ML', 'absE1E2_QM', 'absE1E2_ML'):
            d[key] = np.array([float(r[key]) for r in block])
        out[R] = d
    return out


def make_figure(data, q):
    plt.rcParams.update({
        'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif',
        'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 13,
        'ytick.labelsize': 13, 'legend.fontsize': 13, 'axes.linewidth': 0.9,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
    })

    def scaled(mol, R, which):
        return data[mol][R][q[which]] * q['scale'].get(mol, 1.0)

    # Per-column (per-distance) shared y-limit across molecules,
    # unless a fixed override is given in q['ylim'].
    col_ylim = {}
    for R, _ in DISTANCES:
        if R in q.get('ylim', {}):
            col_ylim[R] = q['ylim'][R]
            continue
        m = 0.0
        for mol, _, _, _ in MOLECULES:
            m = max(m, np.nanmax(scaled(mol, R, 'qm')),
                    np.nanmax(scaled(mol, R, 'ml')))
        col_ylim[R] = (0, m * 1.08 if m > 0 else 1)

    nrows, ncols = len(MOLECULES), len(DISTANCES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.8, 7.6), squeeze=False)

    for i, (mol, mol_label, _, _) in enumerate(MOLECULES):
        for j, (R, dist_label) in enumerate(DISTANCES):
            ax = axes[i][j]
            ang = data[mol][R]['angle']
            qm = scaled(mol, R, 'qm')
            ml = scaled(mol, R, 'ml')

            ax.plot(ang, qm, '-', color=C_DFT, lw=1.6, label='DFT', zorder=2)
            ax.plot(ang, ml, 'o', color=C_ML, ms=4.5,
                    mec='white', mew=0.3, linestyle='None', label='ML',
                    zorder=3)

            ax.set_ylim(col_ylim[R])
            ax.set_xlim(ang.min() - XPAD, ang.max() + XPAD)
            ax.axvline(x=0, color='0.5', linestyle='--', linewidth=0.9, zorder=0)
            ax.tick_params(which='major', length=4, direction='in')
            ax.tick_params(which='minor', length=2.5, direction='in')
            ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: str(abs(int(round(x))))))

            if i == 0:
                ax.set_title(dist_label, fontsize=16, pad=4)
            if i == nrows - 1:
                ax.set_xlabel('Twist/Tilt Angle (deg)')
            if j == 0:
                ax.set_ylabel(mol_label + '\n' + q['label'], fontsize=15)

    axes[0][0].legend(loc='upper left', frameon=False, handlelength=2.0,
                      borderpad=0.3, labelspacing=0.3)

    fig.tight_layout(pad=0.6, h_pad=0.4, w_pad=0.4)
    for ext in ('png', 'pdf'):
        p = os.path.join(HERE, q['out'] + '.' + ext)
        fig.savefig(p, dpi=220, bbox_inches='tight')
        print('wrote', p)
    plt.close(fig)


def main():
    data = {mol: read_molecule(path, angles)
            for mol, _, path, angles in MOLECULES}
    for q in QUANTITIES:
        make_figure(data, q)


if __name__ == '__main__':
    main()
