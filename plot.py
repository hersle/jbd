#!/usr/bin/env python3

import sim
import utils

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import CubicSpline

matplotlib.rcParams |= {
    "text.usetex": True,
    "font.size": 9,
    "figure.figsize": (6.0, 4.0), # default (6.4, 4.8)
    "grid.linewidth": 0.3,
    "grid.alpha": 0.2,
    "legend.labelspacing": 0.3,
    "legend.columnspacing": 1.5,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.3,
    "legend.frameon": False,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.xmargin": 0,
    "axes.ymargin": 0,
}

PARAM_PLOT_INFO = {
    "h":      {"label": r"$h$",                                 "format": lambda h:     f"${h:.2f}$",                  "colorvalue": lambda h:     h},
    "ωb0":    {"label": r"$\omega_{b0}$",                       "format": lambda ωb0:   f"${ωb0:.2f}$",                "colorvalue": lambda ωb0:   ωb0},
    "ωc0":    {"label": r"$\omega_{c0}$",                       "format": lambda ωc0:   f"${ωc0:.2f}$",                "colorvalue": lambda ωc0:   ωc0},
    "ωm0":    {"label": r"$\omega_{m0}$",                       "format": lambda ωm0:   f"${ωm0:.2f}$",                "colorvalue": lambda ωm0:   ωm0},
    "ωk0":    {"label": r"$\omega_{k0}$",                       "format": lambda ωk0:   f"${ωk0:.2f}$",                "colorvalue": lambda ωk0:   ωk0},
    "Tγ0":    {"label": r"$T_{\gamma 0}$",                      "format": lambda Tγ0:   f"${Tγ0:.4f}$",                "colorvalue": lambda Tγ0:   Tγ0},
    "Neff":   {"label": r"$N_\mathrm{eff}$",                    "format": lambda Neff:  f"${Neff:.3f}$",               "colorvalue": lambda Neff:  Neff},
    "kpivot": {"label": r"$k_\mathrm{pivot}$",                  "format": lambda kpivot:f"${kpivot}$",                 "colorvalue": lambda kpivot:kpivot},
    "Lh":     {"label": r"$L / (\mathrm{Mpc}/h)$",              "format": lambda Lh:    f"${Lh:.0f}$",                 "colorvalue": lambda Lh:    np.log2(Lh)},
    "L":      {"label": r"$L /  \mathrm{Mpc}   $",              "format": lambda L:     f"${L:.0f}$",                  "colorvalue": lambda L:     np.log2(L)},
    "Npart":  {"label": r"$N_\mathrm{part}$",                   "format": lambda Npart: f"${Npart}^3$",                "colorvalue": lambda Npart: np.log2(Npart)},
    "Ncell":  {"label": r"$N_\mathrm{cell}$",                   "format": lambda Ncell: f"${Ncell}^3$",                "colorvalue": lambda Ncell: np.log2(Ncell)},
    "Nstep":  {"label": r"$N_\mathrm{step}$",                   "format": lambda Nstep: f"${Nstep}$",                  "colorvalue": lambda Nstep: Nstep},
    "zinit":  {"label": r"$z_\mathrm{init}$",                   "format": lambda zinit: f"${zinit:.0f}$",              "colorvalue": lambda zinit: zinit},
    "ω":      {"label": r"$\omega$",                            "format": lambda ω:     f"$10^{{{np.log10(ω):.0f}}}$", "colorvalue": lambda ω:     np.log10(ω)},
    "lgω":    {"label": r"$\lg\omega$",                         "format": lambda lgω:   f"${lgω:.1f}$",                "colorvalue": lambda ω:     np.log10(ω)},
    "G0":     {"label": r"$G_0/G$",                             "format": lambda G0:    f"${G0:.02f}$",                "colorvalue": lambda G0:    G0},
    "As":     {"label": r"$A_s / 10^{-9}$",                     "format": lambda As:    f"${As/1e-9:.1f}$",            "colorvalue": lambda As:    As},
    "ns":     {"label": r"$n_s$",                               "format": lambda ns:    f"${ns}$",                     "colorvalue": lambda ns:    ns},
    "σ8":     {"label": r"$\sigma_8$",                          "format": lambda σ8:    f"${σ8}$",                     "colorvalue": lambda σ8:    σ8},
    "z":      {"label": r"$z$",                                 "format": lambda z:     f"${z}$",                      "colorvalue": lambda z:     np.log10(z+1)},
}

# linearly look up a color between a list of colors,
# optionally normalizing with values of v that should give the lower, middle and upper colors
def colorbetween(colors, v, vmin=None, vmid=None, vmax=None):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(None, colors)
    if vmin is not None and vmid is not None and vmax is not None:
        A = 0.5 / (vmax - vmid) if vmax - vmid > vmid - vmin else 0.5 / (vmid - vmin) # if/else saturates one end of color spectrum
        B = 0.5 - A * vmid
        v = A * v + B
    RGBA = cmap(v)
    RGB = RGBA[0:3]
    return RGB

def ax_set_ylim_nearest(ax, Δy):
    ymin, ymax = ax.get_ylim()
    ymin = utils.to_nearest(ymin, Δy, "floor")
    ymax = utils.to_nearest(ymax, Δy, "ceil")
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.linspace(ymin, ymax, int(np.round((ymax-ymin)/Δy))+1)) # TODO: set minor ticks every 1, 0.1, 0.01 etc. here?
    return ymin, ymax

def plot_generic(filename, curvess, colors=None, clabels=None, linestyles=None, llabels=None, title=None, xlabel=None, ylabel=None, xticks=None, yticks=None, figsize=(3.0, 2.2)):
    fig, ax = plt.subplots(figsize=figsize)

    if not colors: colors = ["black"] * len(curvess)
    if not linestyles: linestyles = ["solid"] * len(curvess[0])

    # Plot the curves; varying color first, and linestyle second
    ax.set_prop_cycle(cycler(color=colors) * cycler(linestyle=linestyles))
    for curves in curvess:
        for x, y, Δyhi, Δylo in curves:
            ax.plot(        x, y,              linewidth=1, alpha=0.5, label=None)
            ax.fill_between(x, y-Δylo, y+Δyhi, linewidth=0) # error band

    # Set axis labels and ticks from input ticks = (min, max, step)
    for label, ticks, set_label, set_ticks, set_lim, set_minor_locator in [(xlabel, xticks, ax.set_xlabel, ax.set_xticks, ax.set_xlim, ax.xaxis.set_minor_locator), (ylabel, yticks, ax.set_ylabel, ax.set_yticks, ax.set_ylim, ax.yaxis.set_minor_locator)]:
        set_label(label)
        if ticks is not None:
            min, max, stepmajor, stepminor = ticks
            set_ticks(np.linspace(min, max, int(np.round((max - min) / stepmajor)) + 1))
            set_lim(min, max)
            set_minor_locator(matplotlib.ticker.AutoMinorLocator(int(np.round(stepmajor/stepminor))))

    # Label colors (through colorbar)
    if clabels:
        cax  = make_axes_locatable(ax).append_axes("top", size="7%", pad="0%") # align colorbar axis with plot
        cmap = matplotlib.colors.ListedColormap(colors)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), cax=cax, orientation="horizontal")
        cbar.ax.set_title(title)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_tick_params(direction="out")
        cax.xaxis.set_ticks(np.linspace(0.5/len(clabels), 1-0.5/len(clabels), len(clabels)), labels=clabels)

    # Label linestyles (through legend)
    if llabels:
        ax2 = ax.twinx() # use invisible twin axis to create second legend
        ax2.get_yaxis().set_visible(False) # make invisible
        for linestyle, label in zip(linestyles, llabels):
            ax2.plot([-8, -8], [0, 1], alpha=0.5, color="black", linestyle=linestyle, linewidth=1, label=label)
        ax2.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.02))

    ax.grid(which="both")
    fig.tight_layout(pad=0)
    fig.savefig(filename)
    print("Plotted", filename)

def plot_power(filename_stem, params0, paramss, param, θGR, sources=[], nsims=1, hunits=True, divide="", subshot=False, Blims=(0.8, 1.2)):
    names = ["PBD", "PGR", "B"]
    def curve_PBD(sims, source, z):
        k, P, ΔP = sims.sims_BD.power_spectrum(source=source, z=z, hunits=hunits, subshot=subshot)
        return np.log10(k), np.log10(P), np.log10(P+ΔP)-np.log10(P), np.log10(P)-np.log10(P-ΔP)
    def curve_PGR(sims, source, z):
        k, P, ΔP = sims.sims_GR.power_spectrum(source=source, z=z, hunits=hunits, subshot=subshot)
        return np.log10(k), np.log10(P), np.log10(P+ΔP)-np.log10(P), np.log10(P)-np.log10(P-ΔP)
    def curve_B(sims, source, z):
        k, B, ΔB = sims.power_spectrum_ratio(source=source, z=z, hunits=hunits, divide=divide, subshot=subshot)
        return np.log10(k), B, ΔB, ΔB
    funcs = [curve_PBD, curve_PGR, curve_B]
    xticks = (-5, +1, 1, 0.1) # common
    ytickss = [(-6, 5, 1.0, 0.1), (-6, 5, 1.0, 0.1), (Blims[0], Blims[1], 0.10, 0.01)]
    klabel = r"k / (h/\mathrm{Mpc})" if hunits else r"k \,/\, (1/\mathrm{Mpc})"
    xlabel = f"$\lg \left[ {klabel} \\right]$" # common for PBD, PGR, B
    PBDlabel = r"P_\mathrm{BD} \,/\, (\mathrm{Mpc}/h)^3" if hunits else "P_\mathrm{BD} \,/\, \mathrm{Mpc}^3"
    PGRlabel = r"P_\mathrm{GR} \,/\, (\mathrm{Mpc}/h)^3" if hunits else "P_\mathrm{GR} \,/\, \mathrm{Mpc}^3"
    Blabel = "B" if not divide else ("B / B_" + {"class": r"\mathrm{lin}", "primordial": r"\mathrm{prim}", "scaleindependent": r"\mathrm{scale-independent}"}[divide])
    ylabels = [f"$\lg\left[ {PBDlabel} \\right]$", f"$\lg\left[ {PGRlabel} \\right]$", f"${Blabel}$"]

    for name, func, ylabel, yticks in zip(names, funcs, ylabels, ytickss): # 1) iterate over PBD(k), PGR(k), B(k)
        colors, clabels, llabels, linestyles, curvess = [], [], [], [], []
        for params in paramss: # 2) iterate over parameter to vary
            params = params.copy()

            # color and color labels
            if param:
                val  = params[param]
                val0 = params0[param]
                v    = PARAM_PLOT_INFO[param]["colorvalue"](val)  # current  (transformed) value
                v0   = PARAM_PLOT_INFO[param]["colorvalue"](val0) # fiducial (transformed) value
                vmin = PARAM_PLOT_INFO[param]["colorvalue"](np.min([params[param] for params in paramss])) # minimum  (transformed) value
                vmax = PARAM_PLOT_INFO[param]["colorvalue"](np.max([params[param] for params in paramss])) # maximum  (transformed) value
                color = colorbetween(["#0000ff", "#000000", "#ff0000"], v, vmin, v0, vmax)
                clabel = PARAM_PLOT_INFO[param]["format"](val)
                colors.append(color)
                clabels.append(clabel)

            z = params.pop("z", params0["z"])
            sims = sim.SimulationGroupPair(params, θGR, nsims)

            # curves, linestyles and their labels
            curves, linestyles, llabels = [], [], [] # only want to the last two once
            for source in sources: # 3) iterate over power spectrum source
                # linestyle and linestyle labels
                linestyles.append({"class": "solid", "cola": "dashed", "ramses": "dotted", "primordial": "dotted", "scaleindependent": "dashed"}[source])
                llabels.append({"class": r"$\textrm{linear}$", "cola": r"$\textrm{quasi-linear}$", "ramses": r"$\textrm{non-linear}$", "primordial": r"$\textrm{primordial}$", "scaleindependent": r"$\textrm{scale-independent}$"}[source])
                label = (f"${PARAM_PLOT_INFO[param]['label'][1:-1]} = {PARAM_PLOT_INFO[param]['format'](val)[1:-1]}$") if param else None
                curves.append(func(sims, source, z) + ({"class": label, "cola": None, "ramses": None, "scaleindependent": None}[source],))
            curvess.append(curves)

        title = PARAM_PLOT_INFO[param]["label"] if param else None
        plot_generic(f"{filename_stem}_{name}.pdf", curvess, colors, clabels, linestyles, llabels, title, xlabel, ylabel, xticks, yticks)

def plot_quantity_evolution(filename, params0_BD, qty_BD, qty_GR, θGR, qty="", ylabel="", logabs=False, Δyrel=None, Δyabs=None):
    sims = sim.SimulationGroupPair(params0_BD, θGR)

    simBD = sims.sims_BD[0]
    simGR = sims.sims_GR[0]

    a_BD, q_BD = qty_BD(simBD)
    a_GR, q_GR = qty_GR(simGR)
    assert np.all(np.isclose(a_BD, a_GR, atol=1e-3, rtol=1e-3)), "want a_BD ≈ a_GR"

    aeq_BD = 1 / (simBD.read_variable("class/log.txt", "radiation/matter equality at z = ") + 1) # = 1 / (z + 1)
    aeq_GR = 1 / (simGR.read_variable("class/log.txt", "radiation/matter equality at z = ") + 1) # = 1 / (z + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': (1, 1.618)}, figsize=(3.0, 3.5), sharex=True)
    ax2.set_xlabel(r"$\lg a$")
    ax2.set_ylabel(f"$\lg[{ylabel}]$" if logabs else f"${ylabel}$")

    ax1.axhline(1.0, color="gray", linestyle="dashed", linewidth=0.5)
    ax1.plot(np.log10(a_BD), q_BD / q_GR, color="black")
    ax1.set_ylabel(f"${qty}_\mathrm{{BD}}(a) / {qty}_\mathrm{{GR}}(a)$")

    ax2.plot(np.log10(a_BD), np.log10(q_BD) if logabs else q_BD, label=f"${qty}(a) = {qty}_\mathrm{{BD}}(a)$", color="blue")
    ax2.plot(np.log10(a_GR), np.log10(q_GR) if logabs else q_GR, label=f"${qty}(a) = {qty}_\mathrm{{GR}}(a)$", color="red")

    ax2.set_xlim(-10, 0)
    ax2.set_xticks(np.linspace(-10, 0, 11))

    if Δyrel: ax_set_ylim_nearest(ax1, Δyrel)
    if Δyabs: ax_set_ylim_nearest(ax2, Δyabs)

    ymin, ymax = ax2.get_ylim()

    # mark some times
    for a_BD_GR, label in zip(((aeq_BD, aeq_GR),), (r"$\rho_r = \rho_m$",)):
        for a, color, dashoffset in zip(a_BD_GR, ("blue", "red"), (0, 5)):
            for ax in [ax1, ax2]:
                ax.axvline(np.log10(a), color=color, linestyle=(dashoffset, (5, 5)), alpha=0.5, linewidth=1.0)
        ax2.text(np.log10(np.average(a_BD_GR)) - 0.10, ymin + 0.5*(ymax-ymin), label, va="bottom", rotation=90, rotation_mode="anchor")

    ax1.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))
    ax2.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))
    ax1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))
    ax2.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))

    ax1.yaxis.set_label_coords(-0.15, 0.5) # manually position to align for different invocations of this function
    ax2.yaxis.set_label_coords(-0.15, 0.5) # manually position to align for different invocations of this function

    ax1.grid(which="both")
    ax2.grid(which="both")

    ax2.legend()
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.10, top=0.98, hspace=0.1) # trial and error to get consistent plot layout...
    fig.savefig(filename)
    print("Plotted", filename)

def plot_density_evolution(filename, params0_BD, θGR):
    sims = sim.SimulationGroupPair(params0_BD, θGR)

    z_BD, ργ_BD, ρν_BD, ρb_BD, ρc_BD, ρΛϕ_BD, ρcrit_BD = sims.sims_BD[0].read_data("class/background.dat", dict=True, cols=("z", "(.)rho_g", "(.)rho_ur", "(.)rho_b", "(.)rho_cdm", "(.)rho_smg",    "(.)rho_crit"))
    z_GR, ργ_GR, ρν_GR, ρb_GR, ρc_GR, ρΛ_GR,  ρcrit_GR = sims.sims_GR[0].read_data("class/background.dat", dict=True, cols=("z", "(.)rho_g", "(.)rho_ur", "(.)rho_b", "(.)rho_cdm", "(.)rho_lambda", "(.)rho_crit"))

    aeq_BD = 1 / (sims.sims_BD[0].read_variable("class/log.txt", "radiation/matter equality at z = ") + 1) # = 1 / (z + 1)
    aeq_GR = 1 / (sims.sims_GR[0].read_variable("class/log.txt", "radiation/matter equality at z = ") + 1) # = 1 / (z + 1)

    a_BD   = 1 / (z_BD + 1)
    Ωr_BD  = (ργ_BD + ρν_BD) / ρcrit_BD
    Ωm_BD  = (ρb_BD + ρc_BD) / ρcrit_BD
    ΩΛϕ_BD =  ρΛϕ_BD          / ρcrit_BD
    Ω_BD   = Ωr_BD + Ωm_BD + ΩΛϕ_BD

    a_GR   = 1 / (z_GR + 1)
    Ωr_GR  = (ργ_GR + ρν_GR) / ρcrit_GR
    Ωm_GR  = (ρb_GR + ρc_GR) / ρcrit_GR
    ΩΛ_GR  =  ρΛ_GR          / ρcrit_GR
    Ω_GR   = Ωr_GR + Ωm_GR + ΩΛ_GR

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(np.log10(a_BD), Ωr_BD,  color="C0",    linestyle="solid", alpha=0.8, label=r"$\Omega_r^\mathrm{BD}$")
    ax.plot(np.log10(a_BD), Ωm_BD,  color="C1",    linestyle="solid", alpha=0.8, label=r"$\Omega_m^\mathrm{BD}$")
    ax.plot(np.log10(a_BD), ΩΛϕ_BD, color="C2",    linestyle="solid", alpha=0.8, label=r"$\Omega_\Lambda^\mathrm{BD}+\Omega_\phi^\mathrm{BD}$")
    ax.plot(np.log10(a_BD), Ω_BD,   color="black", linestyle="solid", alpha=0.8, label=r"$\Omega^\mathrm{BD}$")
    ax.plot(np.log10(a_GR), Ωr_GR,  color="C0",    linestyle="dashed",  alpha=0.8, label=r"$\Omega_r^\mathrm{GR}$")
    ax.plot(np.log10(a_GR), Ωm_GR,  color="C1",    linestyle="dashed",  alpha=0.8, label=r"$\Omega_m^\mathrm{GR}$")
    ax.plot(np.log10(a_GR), ΩΛ_GR,  color="C2",    linestyle="dashed",  alpha=0.8, label=r"$\Omega_\Lambda^\mathrm{GR}$")
    ax.plot(np.log10(a_GR), Ω_GR,   color="black", linestyle="dashed",  alpha=0.8, label=r"$\Omega^\mathrm{GR}$")
    ax.set_xlabel(r"$\lg a$")
    ax.set_xlim(-7, 0)
    ax.set_ylim(0.0, 1.1)
    ax_set_ylim_nearest(ax, 0.1)
    ax.set_xticks(np.linspace(-7, 0, 8))
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # minor ticks
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # minor ticks
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(filename)
    print("Plotted", filename)

def plot_parameter_samples(filename, paramss, lo, hi):
    params = list(paramss[0].keys())
    valuess = {name: [values[name] for values in paramss] for name in params}

    # remove non-varying parameters
    for name in params:
        if np.min(valuess[name]) == np.max(valuess[name]):
            del valuess[name]

    dimension = len(valuess)

    fig, axs = plt.subplots(dimension-1, dimension-1, figsize=(6.0, 6.0), squeeze=False)
    for iy, paramy in list(enumerate(valuess))[1:]: # iy = 1, ..., dim-1
        sy = valuess[paramy]
        for ix, paramx in list(enumerate(valuess))[:-1]: # ix = 0, ..., dim-2
            sx = valuess[paramx]

            ax = axs[iy-1,ix]

            # plot faces (p1, p2); but not degenerate faces (p2, p1) or "flat faces" with p1 == p2
            if ix >= iy:
                ax.set_visible(False) # hide subplot
                continue

            ax.set_xlabel(PARAM_PLOT_INFO[paramx]["label"] if iy == dimension-1 else "")
            ax.set_ylabel(PARAM_PLOT_INFO[paramy]["label"] if ix == 0           else "")
            ax.set_xlim(lo[paramx], hi[paramx]) # [lo, hi]
            ax.set_ylim(lo[paramy], hi[paramy]) # [lo, hi]
            ax.set_xticks([lo[paramx], np.round((lo[paramx]+hi[paramx])/2, 10), hi[paramx]]) # [lo, mid, hi]
            ax.set_yticks([lo[paramy], np.round((lo[paramy]+hi[paramy])/2, 10), hi[paramy]]) # [lo, mid, hi]
            ax.set_xticklabels([f"${xtick}$" if iy == dimension-1 else "" for xtick in ax.get_xticks()], rotation=90) # rotation=45, ha="right", rotation_mode="anchor") # only on very bottom
            ax.set_yticklabels([f"${ytick}$" if ix == 0           else "" for ytick in ax.get_yticks()], rotation= 0) # rotation=45, ha="right", rotation_mode="anchor") # only on very left
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # minor ticks
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # minor ticks
            ax.xaxis.set_label_coords(+0.5, -0.5) # manually align xlabels vertically   (due to ticklabels with different size)
            ax.yaxis.set_label_coords(-0.5, +0.5) # manually align ylabels horizontally (due to ticklabels with different size)
            ax.grid()

            ax.scatter(sx, sy, 5.0, c="black", edgecolors="none")

    fig.suptitle(f"$\\textrm{{${len(paramss)}$ samples}}$")
    fig.tight_layout(pad=0, rect=(0.02, 0.02, 1.0, 1.0))
    fig.savefig(filename)
    print("Plotted", filename)
