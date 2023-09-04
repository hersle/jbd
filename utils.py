import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt

def to_nearest(number, nearest, mode="round"):
    func = {"round": np.round, "ceil": np.ceil, "floor": np.floor}[mode]
    return func(np.round(number / nearest, 7)) * nearest # round to many digits first to eliminate floating point errors (this is only used for plotting purposes, anyway)

def ax_set_ylim_nearest(ax, Δy):
    ymin, ymax = ax.get_ylim()
    ymin = to_nearest(ymin, Δy, "floor")
    ymax = to_nearest(ymax, Δy, "ceil")
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.linspace(ymin, ymax, int(np.round((ymax-ymin)/Δy))+1)) # TODO: set minor ticks every 1, 0.1, 0.01 etc. here?
    return ymin, ymax

def dictupdate(dict, add={}, remove=[]):
    dict = dict.copy() # don't modify input!
    for key in remove:
        del dict[key]
    for key in add:
        dict[key] = add[key]
    return dict

def dictkeycount(dict, keys, number=None):
    if number is None: number = len(keys)
    return len(set(dict).intersection(set(keys))) == number

def dict2json(dict, sort=False, unicode=False):
    return json.dumps(dict, sort_keys=sort, ensure_ascii=not unicode, indent="\t")

def json2dict(jsonstr):
    return json.loads(jsonstr)

def hashstr(str):
    return hashlib.md5(str.encode('utf-8')).hexdigest()

def hashdict(dict):
    return hashstr(dict2json(dict, sort=True)) # https://stackoverflow.com/a/10288255

def luastr(var):
    if isinstance(var, bool):
        return str(var).lower() # Lua uses true, not True
    elif isinstance(var, str):
        return '"' + str(var) + '"' # enclose in ""
    elif isinstance(var, list):
        return "{" + ", ".join(luastr(el) for el in var) + "}" # Lua uses {} for lists
    else:
        return str(var) # go with python's string representation

# Utility function for verifying that two quantities q1 and q2 are (almost) the same
def check_values_are_close(q1, q2, a1=None, a2=None, name="", atol=0, rtol=0, plot=True):
    are_arrays = isinstance(q1, np.ndarray) and isinstance(q2, np.ndarray)
    if are_arrays:
        # If q1 and q2 are function values at a1 and a2,
        # first interpolate them to common values of a
        # and compare them there
        if a1 is not None and a2 is not None:
            a = a1 if np.min(a1) > np.min(a2) else a2 # for the comparison, use largest a-values
            q1 = np.interp(a, a1, q1)
            q2 = np.interp(a, a2, q2)

        if plot: # for debugging
            plt.plot(np.log10(a), q1)
            plt.plot(np.log10(a), q2)
            plt.savefig("check.png")

        # If q1 and q2 are function values (now at common a1=a2=a),
        # pick out scalars for which the deviation is greatest
        i = np.argmax(np.abs(q1 - q2))
        a  = a[i]
        q1 = q1[i]
        q2 = q2[i]

    # do same test as np.isclose: https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
    # (use atol != 0 for quantities close to zero, and rtol otherwise)
    tol = atol + np.abs(q2) * rtol
    are_close = np.abs(q1 - q2) < tol

    message  = f"q1 = {name}_class = {q1:e}" + (f" (picked values with greatest difference at a = {a})" if are_arrays else "") + "\n"
    message += f"q2 = {name}_cola  = {q2:e}" + (f" (picked values with greatest difference at a = {a})" if are_arrays else "") + "\n"
    message += ("^^ PASSED" if are_close else "FAILED") + f" test |q1-q2| = {np.abs(q1-q2):.2e} < {tol:.2e} = tol = atol + rtol*|q2| with atol={atol:.1e}, rtol={rtol:.1e}" + "\n"

    assert are_close, (f"{name} is not consistent in CLASS and COLA:\n" + message)

# propagate error in f(x1, x2, ...) given
# * df_dx = [df_dx1, df_dx2, ...] (list of numbers): derivatives df/dxi evaluated at mean x
# * xs    = [x1s,    x2s,    ...] (list of lists of numbers): list of observations xis for each variable xi
# (for reference,       see e.g. https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Non-linear_combinations)
# (for an introduction, see e.g. https://veritas.ucd.ie/~apl/labs_master/docs/2020/DA/Matrix-Methods-for-Error-Propagation.pdf)
def propagate_error(df_dx, xs):
    return np.sqrt(np.transpose(df_dx) @ np.cov(xs) @ df_dx)
