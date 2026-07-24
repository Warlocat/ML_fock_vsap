# -*- coding: utf-8 -*-
"""
QM9 learning curves, two panels side by side:
  (a) DZ-basis reference: HOMO / LUMO / Gap (left axis, meV) and the molecular
      dipole moment (right axis, mDebye) vs training set size N.
  (b) TZ-basis reference: HOMO / LUMO / Gap (meV) vs N, extended to N=32000,
      with a dashed line at the average DZ-TZ gap difference (156 meV).

Both panels are log-log; the left (meV) axis is shared between them.
Values below are MAEs in eV / Debye and converted to meV / mDebye (x1000).
Reconstructed from the working session.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------- data ----------------
# Panel (a): DZ, N up to 16000
Na   = np.array([1000, 2000, 4000, 8000, 16000])
HOMO_a = np.array([0.110179769,
                   0.071950771,
                   0.045894698,
                   0.030909084,
                   0.023269015,
])   * 1000  # meV
LUMO_a = np.array([0.105273592,
                   0.071721981,
                   0.046342557,
                   0.031046772,
                   0.023536588,
])  * 1000  # meV
GAP_a  = np.array([0.13076386,
                   0.090441503,
                   0.059030238,
                   0.040418742,
                   0.031516755,
])  * 1000  # meV
DIP_a  = np.array([0.226758969,
                   0.151877721,
                   0.101440253,
                   0.066640285,
                   0.048723172,
])  * 1000  # mDebye

# Panel (b): TZ, N up to 32000 (no dipole)
Nb   = np.array([1000, 2000, 4000, 8000, 16000, 32000])
HOMO_b = np.array([0.11554479,
                   0.096999659,
                   0.079283071,
                   0.067055005,
                   0.052188244,
                   0.045352182,
]) * 1000
LUMO_b = np.array([0.113932337,
                   0.093442826,
                   0.078725068,
                   0.067069752,
                   0.05313874,
                   0.046838024,
]) * 1000
GAP_b  = np.array([0.143508692,
                   0.120729056,
                   0.100844669,
                   0.086638736,
                   0.069798759,
                   0.062147067]) * 1000

DZ_TZ_GAP = 156.0  # meV, average DZ-TZ gap difference (reference line in panel b)

# ---------------- style ----------------
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'mathtext.fontset': 'dejavuserif',
    'font.size': 13,
    'axes.labelsize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.top': True,
})

c_homo, c_lumo, c_gap, c_dip = '#1f77b4', '#2ca02c', '#9467bd', '#d62728'

def int_fmt(x, pos):
    return f'{int(round(x)):d}'

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))

# ================= Panel (a): DZ =================
axAR = axA.twinx()

axA.plot(Na, HOMO_a, marker='o', color=c_homo, lw=1.6, ms=6, label='HOMO')
axA.plot(Na, LUMO_a, marker='s', color=c_lumo, lw=1.6, ms=6, label='LUMO')
axA.plot(Na, GAP_a,  marker='^', color=c_gap,  lw=1.6, ms=6, label='Gap')
axAR.plot(Na, DIP_a, marker='D', color=c_dip,  lw=1.6, ms=6, label='Dipole')

axA.set_xscale('log');  axA.set_yscale('log');  axAR.set_yscale('log')
axA.set_xlim(900, 22000)
axA.set_ylim(15, 200)     # left axis (meV) - shared with panel (b)
axAR.set_ylim(10, 400)    # right axis (mDebye), dipole sits above the energies

xticks_a = [1000, 2000, 5000, 10000, 20000]
axA.xaxis.set_major_locator(mticker.FixedLocator(xticks_a))
axA.xaxis.set_major_formatter(mticker.FixedFormatter([str(t) for t in xticks_a]))
axA.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
axA.xaxis.set_minor_formatter(mticker.NullFormatter())

left_ticks_a  = [20, 30, 50, 70, 100, 200]
right_ticks_a = [10, 20, 50, 100, 200, 400]
axA.yaxis.set_major_locator(mticker.FixedLocator(left_ticks_a))
axA.yaxis.set_major_formatter(mticker.FuncFormatter(int_fmt))
axA.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
axA.yaxis.set_minor_formatter(mticker.NullFormatter())
axAR.yaxis.set_major_locator(mticker.FixedLocator(right_ticks_a))
axAR.yaxis.set_major_formatter(mticker.FuncFormatter(int_fmt))
axAR.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
axAR.yaxis.set_minor_formatter(mticker.NullFormatter())

for ax in (axA, axAR):
    ax.tick_params(which='major', length=5, direction='in')
    ax.tick_params(which='minor', length=3, direction='in')

# right axis in black (matches left)
axAR.spines['right'].set_color('black')
axAR.tick_params(axis='y', colors='black', which='both')
axAR.yaxis.label.set_color('black')

axA.set_xlabel('Training set size $N$')
axA.set_ylabel('MAE (meV)')
axAR.set_ylabel('MAE (mDebye)')

lines  = axA.get_lines() + axAR.get_lines()
labels = [l.get_label() for l in lines]
axA.legend(lines, labels, loc='upper right', frameon=True, facecolor='white',
           edgecolor='none', framealpha=1.0, handlelength=2.0,
           borderpad=0.4, labelspacing=0.35)
axA.grid(False); axAR.grid(False)

# ================= Panel (b): TZ =================
axB.plot(Nb, HOMO_b, marker='o', color=c_homo, lw=1.6, ms=6, label='HOMO')
axB.plot(Nb, LUMO_b, marker='s', color=c_lumo, lw=1.6, ms=6, label='LUMO')
axB.plot(Nb, GAP_b,  marker='^', color=c_gap,  lw=1.6, ms=6, label='Gap')

axB.set_xscale('log');  axB.set_yscale('log')
axB.set_xlim(900, 42000)
axB.set_ylim(15, 200)

xticks_b = [1000, 2000, 5000, 10000, 20000, 50000]
axB.xaxis.set_major_locator(mticker.FixedLocator(xticks_b))
axB.xaxis.set_major_formatter(mticker.FixedFormatter([str(t) for t in xticks_b]))
axB.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
axB.xaxis.set_minor_formatter(mticker.NullFormatter())

yticks_b = [20, 30, 50, 70, 100, 200]
axB.yaxis.set_major_locator(mticker.FixedLocator(yticks_b))
axB.yaxis.set_major_formatter(mticker.FuncFormatter(int_fmt))
axB.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
axB.yaxis.set_minor_formatter(mticker.NullFormatter())

axB.tick_params(which='major', length=5, direction='in', right=True)
axB.tick_params(which='minor', length=3, direction='in', right=True)

# DZ-TZ reference line
axB.axhline(y=DZ_TZ_GAP, linestyle='--', color='0.4', linewidth=1.3)
axB.text(1100, DZ_TZ_GAP * 1.04, 'Average DZ-TZ gap difference',
         fontsize=12, color='0.3', va='bottom', ha='left')

axB.set_xlabel('Training set size $N$')
axB.set_ylabel('MAE (meV)')
axB.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none',
           framealpha=1.0, handlelength=2.0, borderpad=0.4, labelspacing=0.35)
axB.grid(False)

# ================= panel labels =================
axA.text(-0.14, 1.02, '(a)', transform=axA.transAxes, fontsize=17,
         va='bottom', ha='left')
axB.text(-0.14, 1.02, '(b)', transform=axB.transAxes, fontsize=17,
         va='bottom', ha='left')

fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'qm9_learning_curves.{ext}',
                dpi=220, bbox_inches='tight')
print('done')
