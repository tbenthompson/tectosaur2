import inspect
import importlib

from pygments.formatters import HtmlFormatter
from IPython import get_ipython
from IPython.display import display, Code, HTML
import matplotlib.pyplot as plt


def import_and_display_fnc(module_name, fnc_name):
    f = getattr(importlib.import_module(module_name), fnc_name)
    inspect.currentframe().f_back.f_globals[fnc_name] = f
    formatter = HtmlFormatter()
    display(HTML(f'<style>{ formatter.get_style_defs(".highlight") }</style>'))
    display(HTML("<sub>[Library]</sub>"))
    display(Code(inspect.getsource(f), language="python"))


def configure_mpl_fast():
    get_ipython().magic("config InlineBackend.figure_format='png'")


def configure_mpl_pretty():
    get_ipython().magic("config InlineBackend.figure_format='retina'")
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def setup(pretty=True, autoreload=True):
    if autoreload:
        get_ipython().magic("load_ext autoreload")
        get_ipython().magic("autoreload 2")

    if pretty:
        configure_mpl_pretty()
    else:
        configure_mpl_fast()

    plt.rcParams["lines.linewidth"] = 2.0
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["figure.titlesize"] = 22
    plt.rcParams["savefig.transparent"] = False
