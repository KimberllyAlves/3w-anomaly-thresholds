"""Common figure style: white background and a fixed color per detector."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

DETS = ['IF', 'LOF', 'VAE-Rec', 'VAE-Maha']
COLOR = {'IF': '#2a78d6', 'LOF': '#eb6834', 'VAE-Rec': '#1baf7a', 'VAE-Maha': '#b07800'}
MARKER = {'IF': 'o', 'LOF': 's', 'VAE-Rec': '^', 'VAE-Maha': 'D'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8984', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})


def pct():
    return FuncFormatter(lambda v, _: f'{v * 100:.0f}%')


def grid(ax, axis='y'):
    ax.grid(axis=axis, color='#e6e5e0', lw=0.8)
    ax.set_axisbelow(True)
