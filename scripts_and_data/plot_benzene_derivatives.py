# -*- coding: utf-8 -*-
"""
Parity plot of ML vs DFT electronic coupling for substituted benzene (benzene-X)
heterodimers at a single stacked geometry.

Input CSV (benzene_derivatives_data.csv):
    substituent, label_tex, DFT_2J_meV, ML_2J_meV
  - values are RAW 2J in meV; the 0.5 per-pathway factor (degenerate benzene
    HOMO) is applied at plot time via SCALE, so the axes are |t|.
  - label_tex is the annotation drawn next to each point (mathtext allowed).

Layout constants (label offsets, which labels are bold) live in the config
block below - they are presentation, not data, so they stay out of the CSV.
"""
import os
import csv

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- config --------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'benzene_derivatives_data.csv')

# 0.5 -> per-pathway convention for the doubly degenerate benzene HOMO,
# so plotted values (and the reported MAE) are |t|, not 2|t|.
SCALE = 0.5
AXIS_LABEL = '|t| (meV)'

# Label placement: substituent -> (dx_pt, dy_pt, horizontal alignment).
# Each label is put on the SAME side of the parity line as its marker, so no
# label sits on the dashed line. Tweak here if the data changes.
OFFSETS = {
    'CH3':    (7, -13, 'left'),     # on the line -> lower right
    'NO2':    (0, -16, 'center'),   # below line; centred to stay inside axis
    'F':      (-7,  8, 'right'),    # above line -> upper left
    'OH':     (7, -13, 'left'),
    'CH2=CH': (7,   8, 'left'),     # above line; right side (near left axis)
    'NH2':    (7, -13, 'left'),
    'OCH3':   (7, -13, 'left'),
    'CN':     (-7,  8, 'right'),    # above line -> upper left
    'CHO':    (7, -13, 'left'),
    'COOH':   (7, -13, 'left'),
}
DEFAULT_OFFSET = (7, -13, 'left')
BOLD_LABELS = {'CN', 'CHO'}
LABEL_FONTSIZE = 13

C_POINT = '#1f77b4'
# ---------------------------------------------------------------------------


def load_csv(path):
    subs, labels, dft, ml = [], [], [], []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            subs.append(row['substituent'])
            labels.append(row['label_tex'])
            dft.append(float(row['DFT_2J_meV']))
            ml.append(float(row['ML_2J_meV']))
    return subs, labels, np.array(dft) * SCALE, np.array(ml) * SCALE


def main():
    subs, labels, dft, ml = load_csv(CSV_PATH)

    err = ml - dft
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    rmae = 100 * mae / np.mean(np.abs(dft))
    print(f'N = {len(dft)}, MAE = {mae:.2f} meV, RMSE = {rmse:.2f} meV, '
          f'rMAE = {rmae:.1f}%')

    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'mathtext.fontset': 'dejavuserif',
        'font.size': 12,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.linewidth': 0.9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })

    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    lo = min(dft.min(), ml.min()) - 10
    hi = max(dft.max(), ml.max()) + 10
    ax.plot([lo, hi], [lo, hi], '--', color='0.5', lw=1.0, zorder=0)

    ax.plot(dft, ml, 'o', color=C_POINT, ms=8, mfc=C_POINT, mec='black',
            mew=0.5, linestyle='None', zorder=3)

    for s, lab, x, y in zip(subs, labels, dft, ml):
        dx, dy, ha = OFFSETS.get(s, DEFAULT_OFFSET)
        ax.annotate(lab, xy=(x, y), xytext=(dx, dy),
                    textcoords='offset points', fontsize=LABEL_FONTSIZE,
                    ha=ha,
                    fontweight='bold' if s in BOLD_LABELS else 'normal')

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('DFT ' + AXIS_LABEL)
    ax.set_ylabel('ML ' + AXIS_LABEL)

    ax.text(0.04, 0.96, f'MAE = {mae:.1f} meV', transform=ax.transAxes,
            va='top', ha='left', fontsize=12,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

    ax.tick_params(which='major', length=5, direction='in')
    ax.tick_params(which='minor', length=3, direction='in')

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(HERE, 'benzene_derivatives_parity.' + ext)
        fig.savefig(p, dpi=220, bbox_inches='tight')
        print('wrote', p)


if __name__ == '__main__':
    main()
