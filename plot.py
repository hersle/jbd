#!/usr/bin/env python3

import sim
import utils

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    "h":      {"label": r"$h$",                                 "format": lambda h:     f"${h}$",         "colorvalue": lambda h:     h},
    "ωb0":    {"label": r"$\omega_{b0}$",                       "format": lambda ωb0:   f"${ωb0}$",       "colorvalue": lambda ωb0:   ωb0},
    "ωc0":    {"label": r"$\omega_{c0}$",                       "format": lambda ωc0:   f"${ωc0}$",       "colorvalue": lambda ωc0:   ωc0},
    "ωm0":    {"label": r"$\omega_{m0}$",                       "format": lambda ωm0:   f"${ωm0}$",       "colorvalue": lambda ωm0:   ωm0},
    "Lh":     {"label": r"$L / (\mathrm{Mpc}/h)$",              "format": lambda Lh:    f"${Lh:.0f}$",    "colorvalue": lambda Lh:    np.log2(Lh)},
    "Npart":  {"label": r"$N_\mathrm{part}$",                   "format": lambda Npart: f"${Npart}^3$",   "colorvalue": lambda Npart: np.log2(Npart)},
    "Ncell":  {"label": r"$N_\mathrm{cell}$",                   "format": lambda Ncell: f"${Ncell}^3$",   "colorvalue": lambda Ncell: np.log2(Ncell)},
    "Nstep":  {"label": r"$N_\mathrm{step}$",                   "format": lambda Nstep: f"${Nstep}$",     "colorvalue": lambda Nstep: Nstep},
    "zinit":  {"label": r"$z_\mathrm{init}$",                   "format": lambda zinit: f"${zinit:.0f}$", "colorvalue": lambda zinit: zinit},
    "lgω":    {"label": r"$\lg \omega$",                        "format": lambda lgω:   f"${lgω}$",       "colorvalue": lambda lgω:   lgω},
    "G0/G":   {"label": r"$G_0/G$",                             "format": lambda G0_G:  f"${G0_G:.02f}$", "colorvalue": lambda G0_G:  G0_G},
    "Ase9":   {"label": r"$A_s / 10^{-9}$",                     "format": lambda Ase9:  f"${Ase9}$",      "colorvalue": lambda Ase9:  Ase9},
    "ns":     {"label": r"$n_s$",                               "format": lambda ns:    f"${ns}$",        "colorvalue": lambda ns:    ns},
    "σ8":     {"label": r"$\sigma(R=8\,\mathrm{Mpc}/h,\,z=0)$", "format": lambda σ8:    f"${σ8}$",        "colorvalue": lambda σ8:    σ8},
}

def plot_power_spectra(filename, sims, labelfunc = lambda params: None, colorfunc = lambda params: "black"):
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    ax.set_xlabel(r"$\lg[k / (1/\mathrm{Mpc})]$")
    ax.set_ylabel(r"$\lg[P / \mathrm{Mpc}^3]$")

    k, P, ΔP = sims.power_spectrum(linear=True)
    ax.plot(np.log10(k), np.log10(P), color="black", alpha=0.5, linewidth=1, linestyle="dashed")

    for i, sim in enumerate(sims):
        k, P = sim.power_spectrum(linear=False)
        ax.plot(np.log10(k), np.log10(P), color="black", alpha=0.5, linewidth=0.1, linestyle="solid")

    k, P, ΔP = sims.power_spectrum(linear=False)
    ax.fill_between(np.log10(k), np.log10(P-ΔP), np.log10(P+ΔP), color="black", alpha=0.2, edgecolor=None)
    ax.plot(        np.log10(k), np.log10(P),                    color="black", alpha=1.0, linewidth=1.0, linestyle="solid")

    ax.set_xlim(-2, 1)
    ax.set_ylim(1, 4)
    ax.set_xticks([-2, -1, 0, 1])  # also sets xlims
    ax.set_yticks([1, 2, 3, 4]) # also sets xlims
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # one minor tick per 0.01

    fig.tight_layout()
    fig.savefig(filename)
    print("Plotted", filename)

# TODO: plot_single and plot_pair generic functions?

def plot_sequence(filename, paramss, nsims, θGR, labeltitle=None, labelfunc = lambda params: None, colorfunc = lambda params: "black", ax=None, divide_linear = False, logy = False, ylims = None):
    fig, ax = plt.subplots(figsize=(3.0, 2.7))

    ax.set_xlabel(r"$\lg\left[k_\mathrm{BD} / (1/\mathrm{Mpc})\right]$")
    ylabel = r"B(k_\mathrm{BD})"
    if divide_linear: ylabel = f"{ylabel} / B_\mathrm{{linear}}(k_\mathrm{{BD}})"
    if logy:          ylabel = f"\lg [ |{ylabel}| - 1 ]"
    ax.set_ylabel(f"${ylabel}$")

    # Dummy legend plot
    ax2 = ax.twinx() # use invisible twin axis to create second legend
    ax2.get_yaxis().set_visible(False) # make invisible
    ax2.fill_between([-4, -4], [0, 1], alpha=0.2, color="black", edgecolor=None,                  label=r"$B = \langle B \rangle \pm \Delta B$")
    ax2.plot(        [-4, -4], [0, 1], alpha=1.0, color="black", linestyle="solid",  linewidth=1, label=r"$B = \langle B \rangle$")
    ax2.plot(        [-4, -4], [0, 1], alpha=0.5, color="black", linestyle="dashed", linewidth=1, label=r"$B = B_\mathrm{lin}$")
    ax2.legend(loc="upper left", bbox_to_anchor=(-0.02, 0.97))

    for params_BD in paramss:
        sims = sim.SimulationGroupPair(params_BD, θGR, nsims)

        klin, Blin, _  = sims.power_spectrum_ratio(linear=True)
        k,    B   , ΔB = sims.power_spectrum_ratio(linear=False)

        klin, Blin = klin[klin > 1e-3], Blin[klin > 1e-3]
        k, B, ΔB = k[k > 1e-3], B[k > 1e-3], ΔB[k > 1e-3]

        y    = B    / np.interp(k, klin, Blin) if divide_linear else B
        Δy   = ΔB   / np.interp(k, klin, Blin) if divide_linear else ΔB
        ylin = Blin /                    Blin  if divide_linear else Blin

        def T(y): return np.log10(np.abs(y-1)) if logy else y

        ax.plot(np.log10(klin),      T(ylin),          color=colorfunc(params_BD), alpha=1.0, linewidth=1, linestyle="dashed", label=None)
        ax.plot(np.log10(k),         T(y),             color=colorfunc(params_BD), alpha=1.0, linewidth=1, linestyle="solid",  label=None)
        ax.fill_between(np.log10(k), T(y-Δy), T(y+Δy), color=colorfunc(params_BD), alpha=0.2, edgecolor=None) # error band

    Δy = 1.0 if logy else 0.05
    ymin, ymax = ax.get_ylim() if ylims is None else ylims
    #ymin = np.log10(0.95) if logy else 0.95
    #ymax = np.log10(1.05) if logy else 1.05
    ymin = utils.to_nearest(ymin, Δy, "floor")
    ymax = utils.to_nearest(ymax, Δy, "ceil")

    ax.set_xticks([-3, -2, -1, 0, 1])
    ax.set_yticks(np.linspace(ymin, ymax, int(np.round((ymax-ymin)/Δy))+1)) # every Δy
    ax.set_xlim(-3, +1)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # 10 minor ticks
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10)) # 10 minor ticks

    labels = [f"${labelfunc(params_BD)}$" for params_BD in paramss]
    colors = [colorfunc(params_BD) for params_BD in paramss]
    cax  = make_axes_locatable(ax).append_axes("top", size="7%", pad="0%") # align colorbar axis with plot
    cmap = matplotlib.colors.ListedColormap(colors)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), cax=cax, orientation="horizontal")
    cbar.ax.set_title(labeltitle)
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_tick_params(direction="out")
    cax.xaxis.set_ticks(np.linspace(0.5/len(labels), 1-0.5/len(labels), len(labels)), labels=labels)

    ax.grid(which="both")
    fig.tight_layout(pad=0)
    fig.savefig(filename)
    print("Plotted", filename)

def plot_convergence(filename, params0, param, vals, θGR, nsims=5, lfunc=None, **kwargs):
    paramss = [utils.dictupdate(params0, {param: val}) for val in vals] # generate all parameter combinations

    # linear color C = A*v + B so C(v0) = 0.5 (black) and C(vmin) = 0.0 (blue) *or* C(vmax) = 1.0 (red)
    # (black = neutral = fiducial; red = warm = greater; blue = cold = smaller)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blueblackred", ["#0000ff", "#000000", "#ff0000"], N=256)
    def colorfunc(params):
        v    = PARAM_PLOT_INFO[param]["colorvalue"](params[param])  # current  (transformed) value
        v0   = PARAM_PLOT_INFO[param]["colorvalue"](params0[param]) # fiducial (transformed) value
        vmin = np.min(PARAM_PLOT_INFO[param]["colorvalue"](vals))   # minimum  (transformed) value
        vmax = np.max(PARAM_PLOT_INFO[param]["colorvalue"](vals))   # maximum  (transformed) value
        A = 0.5 / (vmax - v0) if vmax - v0 > v0 - vmin else 0.5 / (v0 - vmin) # if/else saturates one end of color spectrum
        B = 0.5 - A * v0
        return cmap(A * v + B)

    def labelfunc(params): return PARAM_PLOT_INFO[param]["format"](params[param])

    plot_sequence(filename, paramss, nsims, θGR, PARAM_PLOT_INFO[param]["label"], labelfunc, colorfunc, **kwargs)

def plot_quantity_evolution(filename, params0_BD, qty_BD, qty_GR, θGR, qty="", ylabel="", logabs=False, Δyrel=None, Δyabs=None):
    sims = sim.SimulationGroupPair(params0_BD, θGR)

    bg_BD = sims.sims_BD[0].read_data("class_background.dat", dict=True)
    bg_GR = sims.sims_GR[0].read_data("class_background.dat", dict=True)

    # want to plot 1e-10 <= a <= 1, so cut away a < 1e-11
    bg_BD = {key: bg_BD[key][1/(bg_BD["z"]+1) > 1e-11] for key in bg_BD}
    bg_GR = {key: bg_GR[key][1/(bg_GR["z"]+1) > 1e-11] for key in bg_GR}

    a_BD = 1 / (bg_BD["z"] + 1) # = 1 / (z + 1)
    a_GR = 1 / (bg_GR["z"] + 1) # = 1 / (z + 1)

    aeq_BD = 1 / (sims.sims_BD[0].read_variable("class.log", "radiation/matter equality at z = ") + 1) # = 1 / (z + 1)
    aeq_GR = 1 / (sims.sims_GR[0].read_variable("class.log", "radiation/matter equality at z = ") + 1) # = 1 / (z + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': (1, 1.618)}, figsize=(3.0, 3.5), sharex=True)
    ax2.set_xlabel(r"$\lg a$")
    ax2.set_ylabel(f"$\lg[{ylabel}]$" if logabs else f"${ylabel}$")

    ax1.axhline(1.0, color="gray", linestyle="dashed", linewidth=0.5)
    ax1.plot(np.log10(a_BD), qty_BD(bg_BD, sims.params_BD) / qty_GR(bg_GR, sims.params_GR), color="black")
    ax1.set_ylabel(f"${qty}_\mathrm{{BD}}(a) / {qty}_\mathrm{{GR}}(a)$")

    ax2.plot(np.log10(a_BD), np.log10(qty_BD(bg_BD, sims.params_BD)) if logabs else qty_BD(bg_BD, sims.params_BD), label=f"${qty}(a) = {qty}_\mathrm{{BD}}(a)$", color="blue")
    ax2.plot(np.log10(a_GR), np.log10(qty_GR(bg_GR, sims.params_GR)) if logabs else qty_GR(bg_GR, sims.params_GR), label=f"${qty}(a) = {qty}_\mathrm{{GR}}(a)$", color="red")

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

    ax2.legend()
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.10, top=0.98, hspace=0.1) # trial and error to get consistent plot layout...
    fig.savefig(filename)
    print("Plotted", filename)

def plot_density_evolution(filename, params0_BD, θGR):
    sims = sim.SimulationGroupPair(params0_BD, θGR)

    z_BD, ργ_BD, ρν_BD, ρb_BD, ρc_BD, ρΛϕ_BD, ρcrit_BD = sims.sims_BD[0].read_data("class_background.dat", dict=True, cols=("z", "(.)rho_g", "(.)rho_ur", "(.)rho_b", "(.)rho_cdm", "(.)rho_smg",    "(.)rho_crit"))
    z_GR, ργ_GR, ρν_GR, ρb_GR, ρc_GR, ρΛ_GR,  ρcrit_GR = sims.sims_GR[0].read_data("class_background.dat", dict=True, cols=("z", "(.)rho_g", "(.)rho_ur", "(.)rho_b", "(.)rho_cdm", "(.)rho_lambda", "(.)rho_crit"))

    aeq_BD = 1 / (sims.sims_BD[0].read_variable("class.log", "radiation/matter equality at z = ") + 1) # = 1 / (z + 1)
    aeq_GR = 1 / (sims.sims_GR[0].read_variable("class.log", "radiation/matter equality at z = ") + 1) # = 1 / (z + 1)

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

def plot_parameter_samples(filename, samples, lo, hi, labels):
    params = list(lo.keys())
    dimension = len(params)

    fig, axs = plt.subplots(dimension-1, dimension-1, figsize=(6.0, 6.0))
    #fig = matplotlib.figure.Figure(figsize=(6.0, 6.0))
    #axs = fig.subplots(dimension-1, dimension-1)
    for iy in range(1, dimension):
        paramy = params[iy]
        sy = [sample[paramy] for sample in samples]
        for ix in range(0, dimension-1):
            paramx = params[ix]
            sx = [sample[paramx] for sample in samples]

            ax = axs[iy-1,ix]

            # plot faces (p1, p2); but not degenerate faces (p2, p1) or "flat faces" with p1 == p2
            if ix >= iy:
                ax.set_visible(False) # hide subplot
                continue

            ax.set_xlabel(param_labels[paramx] if iy == dimension-1 else "")
            ax.set_ylabel(param_labels[paramy] if ix == 0           else "")
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

            ax.scatter(sx, sy, 2.0, c="black", edgecolors="none")

    fig.suptitle(f"$\\textrm{{${len(samples)}$ Latin hypercube samples}}$")
    fig.tight_layout(pad=0, rect=(0.02, 0.02, 1.0, 1.0))
    fig.savefig(filename)
    print("Plotted", filename)
