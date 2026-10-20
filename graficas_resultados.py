"""
=========================================================================
 QBLADE RESULTS PLOTS - NREL 5 MW (STEP 1 and STEP 2)
 -------------------------------------------------------------------------
 Reads the .txt files exported from QBlade and generates the report plots:
    - Power coefficient Cp vs TSR          (Rotor BEM)
    - Power, Pitch and Thrust vs V         (Turbine BEM)
    - Blade geometry: chord, twist and thickness vs radius
 comparing STEP1 / STEP2 and without / with corrections.

 Expected folder structure:
    CALCULO AUTOMATICO/.../Resultados/STEP1/  Cp*.txt  Potencia*.txt ...
    CALCULO AUTOMATICO/.../Resultados/STEP2/  (same files; skipped if missing)

 Simulation names must contain "STEP1"/"STEP2" and "sin_corr"/"con_corr".
 Otherwise add them to the ALIASES dict below.
=========================================================================
"""

import csv
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# CONFIGURACION
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent           # carpeta "Estudo em Python"

# Si "Resultados" no está junto al script, se busca automáticamente
RESULTS_FOLDER = BASE_DIR / "Resultados"
STEPS          = ["STEP1", "STEP2"]

# Archivo de pala para los gráficos de geometría
BLADE_FILE = BASE_DIR.parent / "QBlade_Re6.0" / "NREL5MW_STEP1_N9.bld"

# Aliases: nombres de simulaciones QBlade que no siguen el patrón estándar
ALIASES = {
    "NREL5MW_STEP2_interp Simulation": "STEP2_con_corr",
}

R      = 63.0   # radio del rotor [m]
P_RATED = 5e6  # potencia nominal [W]

# Variables: (prefijo, clave, xlabel, ylabel, factor, título)
VARIABLES = [
    ("Cp",       "Cp",     r"Tip speed ratio $\lambda$ [-]", r"$C_P$ [-]",        1.0,  "Power coefficient"),
    ("Potencia", "Power",  "Wind speed V [m/s]",              "Power [MW]",        1e-6, "Power curve"),
    ("Pitch",    "Pitch",  "Wind speed V [m/s]",              "Pitch angle [deg]", 1.0,  "Blade pitch angle"),
    ("Thrust",   "Thrust", "Wind speed V [m/s]",              "Thrust [kN]",       1e-3, "Rotor thrust (axial force)"),
]

STEP_COLOR = {"STEP1": [0.000, 0.447, 0.741], "STEP2": [0.850, 0.325, 0.098]}
CORR_LINE  = {"sin": "--", "con": "-"}
CORR_TEXT  = {"sin": "no corrections", "con": "tip loss + 3D corrections"}
STEP_TEXT  = {"STEP1": "STEP 1 (clean, N=9)", "STEP2": "STEP 2 (0.7 clean + 0.3 turb.)"}


# =========================================================================
# BUSQUEDA DE LA CARPETA DE RESULTADOS
# =========================================================================

def _find_results_folder(base: Path) -> Path:
    """Busca recursivamente una carpeta 'Resultados' que contenga 'STEP1'."""
    candidata = base / "Resultados"
    if (candidata / "STEP1").is_dir():
        return candidata
    for d in base.rglob("Resultados"):
        if d.is_dir() and (d / "STEP1").is_dir():
            return d
    raise FileNotFoundError(
        f'Carpeta "Resultados/STEP1" no encontrada en:\n  {base}'
    )


results_folder = _find_results_folder(BASE_DIR)
print(f"Leyendo resultados de:\n  {results_folder}\n")
fig_folder = results_folder / "Figures"
fig_folder.mkdir(exist_ok=True)


# =========================================================================
# LECTURA DE EXPORTACIONES QBLADE
# =========================================================================

def leer_qblade_export(ruta: Path) -> list[dict]:
    """
    Lee un archivo .txt exportado de QBlade.
    Línea 2: nombres de simulaciones (separados por tabulador)
    Línea 3: cabeceras de columnas
    Resto  : datos; "END" marca el fin de una serie corta.

    Devuelve lista de dicts: {'name': str, 'x': array, 'y': array}
    """
    lineas = ruta.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lineas) < 4:
        return []

    nombres = [s.strip() for s in lineas[1].split("\t")]
    n_col   = len(lineas[2].split("\t"))

    M = np.full((len(lineas) - 3, n_col), np.nan)
    for i, ln in enumerate(lineas[3:]):
        if not ln.strip():
            continue
        tok = ln.strip().split("\t")
        n   = min(len(tok), n_col)
        for j in range(n):
            try:
                M[i, j] = float(tok[j])
            except ValueError:
                pass   # "END" -> NaN (ya inicializado)

    series = []
    for k in range(min(len(nombres), n_col // 2)):
        nm = nombres[k]
        if not nm or nm.startswith("["):
            continue
        x   = M[:, 2 * k]
        y   = M[:, 2 * k + 1]
        ok  = ~np.isnan(x) & ~np.isnan(y)
        series.append({"name": nm, "x": x[ok], "y": y[ok]})
    return series


def _clasificar_serie(nombre: str) -> tuple[str, str]:
    """Extrae (step, corr) de un nombre de serie. Devuelve ('', '') si no reconoce."""
    step = ""
    t = re.search(r'STEP\s*_?(\d)', nombre, re.IGNORECASE)
    if t:
        step = f"STEP{t.group(1)}"
    corr = ""
    if re.search(r'sin[_ ]?corr', nombre, re.IGNORECASE):
        corr = "sin"
    if re.search(r'con[_ ]?corr', nombre, re.IGNORECASE):
        corr = "con"
    return step, corr


# =========================================================================
# LECTURA DE TODAS LAS SERIES
# =========================================================================
data = {pref: [] for pref, *_ in VARIABLES}

for pref, *_ in VARIABLES:
    series_leidas: list[dict] = []
    names_seen: set[str] = set()

    for step in STEPS:
        folder = results_folder / step
        if not folder.is_dir():
            continue
        for txt in folder.glob(f"{pref}*.txt"):
            for s in leer_qblade_export(txt):
                if s["name"] in names_seen:
                    continue
                names_seen.add(s["name"])

                # Aplicar alias si corresponde
                nombre_clasificar = ALIASES.get(s["name"], s["name"])
                st, corr = _clasificar_serie(nombre_clasificar)

                if not st or not corr:
                    print(f'  AVISO: serie "{s["name"]}" ignorada (nombre no reconocido)')
                    continue
                series_leidas.append({**s, "step": st, "corr": corr})

    data[pref] = series_leidas
    print(f"{pref:<9}: {len(series_leidas)} series leídas")


# =========================================================================
# GRAFICAS INDIVIDUALES
# =========================================================================
for pref, clave, xlabel, ylabel, factor, titulo in VARIABLES:
    series = data[pref]
    if not series:
        continue

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{titulo} - NREL 5 MW"); ax.grid(True)

    for s in series:
        label = f"{STEP_TEXT[s['step']]} - {CORR_TEXT[s['corr']]}"
        ax.plot(s["x"], s["y"] * factor,
                CORR_LINE[s["corr"]], color=STEP_COLOR[s["step"]],
                linewidth=1.6, label=label)
        if pref == "Cp":
            i_max = np.argmax(s["y"])
            ax.plot(s["x"][i_max], s["y"][i_max],
                    "o", color=STEP_COLOR[s["step"]],
                    markerfacecolor=STEP_COLOR[s["step"]])

    if pref == "Cp":
        ax.axhline(16 / 27, linestyle=":", color="k", linewidth=0.8, label="Betz limit")

    ax.legend(loc="best")
    fig.tight_layout()
    out = fig_folder / f"{clave}.png"
    fig.savefig(out, dpi=200)
    print(f"  Figura guardada: {out}")


# =========================================================================
# PANEL RESUMEN 2x2
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("NREL 5 MW - QBlade BEM results")
axes_flat = axes.flatten()

for idx, (pref, clave, xlabel, ylabel, factor, titulo) in enumerate(VARIABLES):
    ax = axes_flat[idx]
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(titulo); ax.grid(True)
    for s in data[pref]:
        label = f"{STEP_TEXT[s['step']]} - {CORR_TEXT[s['corr']]}"
        ax.plot(s["x"], s["y"] * factor,
                CORR_LINE[s["corr"]], color=STEP_COLOR[s["step"]],
                linewidth=1.4, label=label)
    if pref == "Cp":
        ax.axhline(16 / 27, linestyle=":", color="k", linewidth=0.8)
        ax.legend(loc="lower center", fontsize=7)

fig.tight_layout()
out = fig_folder / "Summary.png"
fig.savefig(out, dpi=200)
print(f"  Figura resumen guardada: {out}")


# =========================================================================
# TABLA RESUMEN
# =========================================================================
print(f"\n{'Step':<6} {'Corrections':<28} {'Cp_max':>8} {'TSR_opt':>8} "
      f"{'V_rated':>12} {'T_max[kN]':>12} {'V(T_max)':>10} {'Pitch@25':>10}")

csv_rows = []
for step in STEPS:
    for corr in ["sin", "con"]:
        def _find(key):
            return next((s for s in data[key] if s["step"] == step and s["corr"] == corr), None)

        s_cp = _find("Cp");  s_pw = _find("Potencia")
        s_th = _find("Thrust"); s_pi = _find("Pitch")

        if s_cp is None and s_pw is None:
            continue

        cp_max  = float(np.max(s_cp["y"])) if s_cp else float("nan")
        tsr_opt = float(s_cp["x"][np.argmax(s_cp["y"])]) if s_cp else float("nan")

        v_rated = float("nan")
        if s_pw is not None:
            idx_v = np.where(s_pw["y"] >= 0.999 * P_RATED)[0]
            if len(idx_v):
                v_rated = float(s_pw["x"][idx_v[0]])

        t_max = float(np.max(s_th["y"])) if s_th else float("nan")
        v_tmax = float(s_th["x"][np.argmax(s_th["y"])]) if s_th else float("nan")
        pitch_end = float(s_pi["y"][-1]) if s_pi is not None and len(s_pi["y"]) else float("nan")

        print(f"{step:<6} {CORR_TEXT[corr]:<28} {cp_max:8.4f} {tsr_opt:8.2f} "
              f"{v_rated:12.2f} {t_max/1e3:12.1f} {v_tmax:10.2f} {pitch_end:10.2f}")
        csv_rows.append([step, CORR_TEXT[corr], cp_max, tsr_opt,
                         v_rated, t_max / 1e3, v_tmax, pitch_end])

if csv_rows:
    csv_file = fig_folder / "Results_summary.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Step", "Corrections", "Cp_max", "TSR_opt",
                    "V_rated_ms", "Thrust_max_kN", "V_at_Thrust_max_ms",
                    "Pitch_at_25ms_deg"])
        w.writerows(csv_rows)
    print(f"\n  CSV resumen guardado: {csv_file}")


# =========================================================================
# GEOMETRIA DE LA PALA
# =========================================================================

def _espesor_perfil(nombre_plr: str) -> tuple[float, str]:
    """Extrae el espesor relativo [%] y el nombre del perfil del nombre del .plr."""
    t = re.search(r'du[_ -]?(\d{2})', nombre_plr, re.IGNORECASE)
    if t:
        return float(t.group(1)), f"DU{t.group(1)}"
    if re.search(r'na(ca)?[_ -]?64', nombre_plr, re.IGNORECASE):
        return 18.0, "NACA64"
    if re.search(r'cylinder|circle', nombre_plr, re.IGNORECASE):
        return 100.0, "Cylinder"
    return float("nan"), "?"


def leer_geometria_bld(ruta: Path) -> dict:
    """Lee radio, cuerda, twist y perfil de cada estación de un .bld."""
    lineas  = ruta.read_text(encoding="utf-8", errors="replace").splitlines()
    G       = {"r": [], "chord": [], "twist": [], "thick": [], "airfoil": []}
    en_tabla = False
    for ln in lineas:
        if "POLAR_FILE" in ln:
            en_tabla = True; continue
        if not en_tabla:
            continue
        m = re.match(r'^(.*\s)(\S+\.plr)\s*$', ln)
        if not m:
            continue
        vals = [float(v) for v in m.group(1).split() if re.match(r'^-?\d', v)]
        if len(vals) < 3:
            continue
        th, af = _espesor_perfil(m.group(2))
        G["r"].append(vals[0]); G["chord"].append(vals[1]); G["twist"].append(vals[2])
        G["thick"].append(th);  G["airfoil"].append(af)
    return {k: np.array(v) if k != "airfoil" else v for k, v in G.items()}


if BLADE_FILE.is_file():
    G = leer_geometria_bld(BLADE_FILE)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.suptitle("NREL 5 MW blade geometry")

    axes[0].plot(G["r"], G["chord"], "k-o", markersize=3, linewidth=1.4)
    axes[0].set_xlabel("Radius r [m]"); axes[0].set_ylabel("Chord c [m]")
    axes[0].set_title("Chord distribution"); axes[0].set_xlim(0, R); axes[0].grid(True)

    axes[1].plot(G["r"], G["twist"], "k-o", markersize=3, linewidth=1.4)
    axes[1].set_xlabel("Radius r [m]"); axes[1].set_ylabel("Twist angle [deg]")
    axes[1].set_title("Twist distribution"); axes[1].set_xlim(0, R); axes[1].grid(True)

    axes[2].step(G["r"], G["thick"], "k-", where="post", linewidth=1.4)
    # Etiquetar cada perfil único
    seen = set()
    for i, af in enumerate(G["airfoil"]):
        if af not in seen:
            axes[2].text(G["r"][i], G["thick"][i], f"  {af}",
                         va="bottom", fontsize=8)
            seen.add(af)
    axes[2].set_xlabel("Radius r [m]")
    axes[2].set_ylabel("Relative thickness t/c [%]")
    axes[2].set_title("Thickness distribution")
    axes[2].set_xlim(0, R); axes[2].set_ylim(0, 110); axes[2].grid(True)

    fig.tight_layout()
    out = fig_folder / "Blade_geometry.png"
    fig.savefig(out, dpi=200)
    print(f"  Figura geometría guardada: {out}")
else:
    print(f"  AVISO: archivo de pala no encontrado: {BLADE_FILE}")

print(f"\nFiguras guardadas en:\n  {fig_folder}")
plt.show()
