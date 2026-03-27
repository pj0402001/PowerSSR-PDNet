"""
Generate a publication-style SSR-PDNet architecture figure.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


FIG_DIR = Path(__file__).parent.parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'mathtext.fontset': 'stix',
    'font.size': 10,
})

FIG_W, FIG_H = 12.5, 4.8
EDGE = '#263238'
TEXT = '#172026'
MUTED = '#5b6b73'
BG = '#f7f7f4'
GROUP_BG = '#eef2f3'
INPUT_C = '#d8e6f0'
ENC_C = '#d7efe8'
CLS_C = '#e8dcc5'
PHYS_C = '#efe2cf'
LOSS_C = '#f1efe8'
ACCENT = '#1f5f8b'
ACCENT_2 = '#9b5d2f'


fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor('white')
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')


def rounded_box(x, y, w, h, title, subtitle='', facecolor='#ffffff',
                edgecolor=EDGE, lw=1.2, title_size=10, subtitle_size=8,
                weight='bold', radius=0.12, zorder=3):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f'round,pad=0.06,rounding_size={radius}',
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.12 if subtitle else 0.0), title,
            ha='center', va='center', color=TEXT,
            fontsize=title_size, fontweight=weight, zorder=zorder + 1)
    if subtitle:
        ax.text(x, y - 0.23, subtitle,
                ha='center', va='center', color=MUTED,
                fontsize=subtitle_size, zorder=zorder + 1)
    return rect



def group_box(x, y, w, h, label):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle='round,pad=0.08,rounding_size=0.14',
        facecolor=GROUP_BG,
        edgecolor='#c9d2d6',
        linestyle='--',
        linewidth=1.0,
        zorder=1,
    )
    ax.add_patch(rect)
    ax.text(x - w / 2 + 0.16, y + h / 2 - 0.22, label,
            ha='left', va='center', fontsize=9, color=MUTED,
            fontweight='bold', zorder=2)



def arrow(x1, y1, x2, y2, color=EDGE, lw=1.7, style='-|>', rad=0.0, zorder=5):
    ax.annotate(
        '',
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            shrinkA=4,
            shrinkB=4,
            connectionstyle=f'arc3,rad={rad}',
        ),
        zorder=zorder,
    )


# Group regions
group_box(2.3, 3.5, 2.8, 1.8, 'Input coordinates')
group_box(5.2, 3.5, 2.2, 2.4, 'Shared representation')
group_box(8.4, 4.75, 3.0, 2.0, 'Security head')
group_box(8.4, 2.15, 3.0, 2.0, 'Physics head')
group_box(12.1, 3.5, 2.7, 4.4, 'Training objectives')

# Main blocks
rounded_box(1.5, 3.5, 1.8, 0.95,
            r'Generator dispatch $\mathbf{u}$',
            r'$\left(P_{G_i}\right)$ in MW',
            facecolor=INPUT_C, title_size=10)
rounded_box(3.2, 3.5, 1.5, 0.9,
            'Normalization',
            'affine scaling',
            facecolor=INPUT_C, title_size=9)
rounded_box(5.2, 3.5, 1.7, 1.25,
            'Shared encoder',
            '2x Linear - LayerNorm - SiLU',
            facecolor=ENC_C, title_size=10)
ax.text(5.2, 2.47, r'latent feature $\mathbf{z} \in \mathbb{R}^{d}$',
        ha='center', va='center', fontsize=8.5, color=ACCENT)

rounded_box(7.6, 4.75, 1.75, 0.95,
            'Residual MLP',
            'boundary-aware classifier',
            facecolor=CLS_C, title_size=9)
rounded_box(9.5, 4.75, 1.55, 0.95,
            'Security score',
            r'$\hat{p}=\sigma(\ell)$',
            facecolor=CLS_C, title_size=9)
rounded_box(7.6, 2.15, 1.75, 0.95,
            'State predictor',
            'compact voltage surrogate',
            facecolor=PHYS_C, title_size=9)
rounded_box(9.5, 2.15, 1.55, 0.95,
            'Physical states',
            r'$\hat{\mathbf{V}} \in [V_{\min}, V_{\max}]$',
            facecolor=PHYS_C, title_size=9)

rounded_box(12.1, 5.25, 2.0, 0.88,
            'Focal loss',
            'class imbalance',
            facecolor=LOSS_C, title_size=9)
rounded_box(12.1, 3.95, 2.0, 0.88,
            'Boundary contrastive loss',
            'sharp SSR frontier',
            facecolor=LOSS_C, title_size=8.7)
rounded_box(12.1, 2.65, 2.0, 0.88,
            'Primal-dual physics loss',
            r'voltage penalties with $\lambda_V$',
            facecolor=LOSS_C, title_size=8.5)
rounded_box(12.1, 1.35, 2.0, 0.88,
            'Joint objective',
            r'$\mathcal{L}_{tot}=\mathcal{L}_{focal}+\lambda_c\mathcal{L}_{ctr}+\lambda_{phys}\mathcal{L}_{phys}$',
            facecolor='#f4f2ec', title_size=8.4, subtitle_size=7.7)

# Arrows through the network
arrow(2.42, 3.5, 2.52, 3.5)
arrow(3.92, 3.5, 4.26, 3.5)
arrow(6.08, 3.72, 6.82, 4.52)
arrow(6.08, 3.28, 6.82, 2.38)
arrow(8.48, 4.75, 8.72, 4.75)
arrow(8.48, 2.15, 8.72, 2.15)
arrow(10.28, 4.75, 11.0, 5.18, color=ACCENT, lw=1.6)
arrow(10.28, 4.75, 11.0, 3.96, color=ACCENT, lw=1.6)
arrow(10.28, 2.15, 11.0, 2.72, color=ACCENT_2, lw=1.6)
arrow(12.1, 4.82, 12.1, 4.38, lw=1.2, color='#6d7b83')
arrow(12.1, 3.52, 12.1, 3.08, lw=1.2, color='#6d7b83')
arrow(12.1, 2.22, 12.1, 1.78, lw=1.2, color='#6d7b83')

# Side notes
ax.text(10.95, 6.05,
        'Classification branch', ha='center', va='center',
        fontsize=8.5, color=ACCENT, fontweight='bold')
ax.text(10.95, 0.95,
        'Physics-guided regularization', ha='center', va='center',
        fontsize=8.5, color=ACCENT_2, fontweight='bold')
ax.text(7.95, 5.55,
        'residual shortcut', ha='center', va='center',
        fontsize=7.6, color=MUTED)
arrow(5.9, 4.25, 8.35, 5.28, color='#8c8c8c', lw=1.0, rad=0.22, zorder=4)

ax.set_title(
    'SSR-PDNet Neural Architecture for Static Security Region Characterization',
    fontsize=12.5,
    fontweight='bold',
    color=TEXT,
    pad=12,
)
ax.text(0.35, 6.55,
        'Shared latent features support both security boundary prediction and physically meaningful state regularization.',
        ha='left', va='center', fontsize=8.8, color=MUTED)

plt.tight_layout(pad=0.8)
out = FIG_DIR / 'architecture.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out}')
