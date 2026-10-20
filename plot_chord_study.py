"""
=========================================================================
 TIP CHORD STUDY - ONE COMPARISON FIGURE - NREL 5 MW (clean polars)
 -------------------------------------------------------------------------
 Base blade (STEP 1) vs tip chord variants (+/-5%, +/-7%) in a single
 2x2 figure: Cp vs TSR, Power, Pitch and Thrust vs wind speed.

 - Reads EVERY .txt in the two folders below (file names do not matter).
 - The variable is detected from the QBlade column header.
 - Plots EVERY simulation found (names need no special format).
 - Uses its own folder ESTUDIO_CUERDA/ (created and filled automatically).
=========================================================================
"""

import re
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# CONFIGURACION
# -------------------------------------------------------------------------
BASE_DIR  = Path(__file__).parent           # carpeta "Estudo em Python"

STUDY_DIR = BASE_DIR / "ESTUDIO_CUERDA"
BASE_QBLADE_DIR = BASE_DIR / "Resultados" / "STEP1"
TIP_QBLADE_DIR  = BASE_DIR / "Resultados" / "TIP_CORRECTIONS"

OUT_FILE  = STUDY_DIR / "TipChord_comparison.png"

# Variables reconocidas por la cabecera de QBlade
VARS = [
    ("Cp",     "Power Coefficient",  r"Tip speed ratio $\lambda$ [-]", r"$C_P$ [-]",        1.0,  "Power coefficient"),
    ("Power",  "Power [W]",          "Wind speed V [m/s]",              "Power [MW]",        1e-6, "Power curve"),
    ("Pitch",  "Pitch [deg]",        "Wind speed V [m/s]",              "Pitch angle [deg]", 1.0,  "Blade pitch angle"),
    ("Thrust", "Thrust [N]",         "Wind speed V [m/s]",              "Thrust [kN]",       1e-3, "Rotor thrust"),
]


# =========================================================================
# PREPARACION DE CARPETAS Y COPIA DE .txt
# =========================================================================

def _copiar_txt(origen: Path, destino: Path) -> int:
    """Copia todos los .txt de origen a destino. Devuelve el número copiado."""
    if not origen.is_dir():
        return 0
    n = 0
    for f in origen.glob("*.txt"):
        shutil.copy2(f, destino / f.name)
        n += 1
    return n


def _buscar_y_copiar(destino: Path, base: Path, nombre_match,
                     excluir: Path, etiqueta: str) -> None:
    """
    Si destino no tiene .txt, busca recursivamente en base la carpeta
    cuyo nombre satisface nombre_match (fuera de excluir) con más .txt
    y los copia.
    """
    if list(destino.glob("*.txt")):
        return
    mejor    = None
    n_mejor  = 0
    for d in base.rglob("*"):
        if not d.is_dir():
            continue
        try:
            d.relative_to(excluir)
            continue           # dentro de la carpeta excluida
        except ValueError:
            pass
        if nombre_match(d.name):
            n = len(list(d.glob("*.txt")))
            if n > n_mejor:
                mejor = d;  n_mejor = n
    if mejor is None:
        raise FileNotFoundError(
            f'No se encontró carpeta "{etiqueta}" con .txt en:\n  {base}\n'
            f'Copia los .txt de QBlade manualmente a:\n  {destino}'
        )
    for f in mejor.glob("*.txt"):
        shutil.copy2(f, destino / f.name)
    print(f"Copiados {n_mejor} .txt desde {mejor}")


# Crear carpetas si no existen
base_dir = STUDY_DIR / "BASE"
tip_dir  = STUDY_DIR / "TIP"
base_dir.mkdir(parents=True, exist_ok=True)
tip_dir.mkdir(parents=True, exist_ok=True)

# Copiar desde las fuentes conocidas
_copiar_txt(BASE_QBLADE_DIR, base_dir)
_copiar_txt(TIP_QBLADE_DIR,  tip_dir)

# Fallback: buscar automáticamente si las carpetas fuente no existen
_buscar_y_copiar(base_dir, BASE_DIR,
                 lambda n: n.upper() == "STEP1",
                 STUDY_DIR, "STEP1")
_buscar_y_copiar(tip_dir, BASE_DIR,
                 lambda n: n.upper().startswith("TIP"),
                 STUDY_DIR, "TIP...")

print(f"Carpeta base : {base_dir}  ({len(list(base_dir.glob('*.txt')))} .txt)")
print(f"Carpeta tip  : {tip_dir}   ({len(list(tip_dir.glob('*.txt')))} .txt)\n")


# =========================================================================
# LECTURA
# =========================================================================

def leer_qblade_export(ruta: Path) -> tuple[list[dict], str]:
    """
    Lee un .txt de QBlade. Devuelve (lista de series, cabecera de columnas).
    Cada serie: {'name': str, 'x': array, 'y': array}
    """
    lineas = ruta.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lineas) < 4:
        return [], ""

    nombres = [s.strip() for s in lineas[1].split("\t")]
    header  = lineas[2]
    n_col   = len(header.split("\t"))

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
                pass

    series = []
    for k in range(min(len(nombres), n_col // 2)):
        nm = nombres[k]
        if not nm or nm.startswith("["):
            continue
        x  = M[:, 2 * k];  y = M[:, 2 * k + 1]
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() > 1:
            series.append({"name": nm, "x": x[ok], "y": y[ok]})
    return series, header


D: dict[str, list[dict]] = {v[0]: [] for v in VARS}

for f_idx, folder in enumerate([base_dir, tip_dir]):
    is_base = (f_idx == 0)
    for txt in folder.glob("*.txt"):
        series, header = leer_qblade_export(txt)
        # Detectar la variable por la cabecera
        v_idx = next(
            (i for i, v in enumerate(VARS)
             if v[1].lower() in header.lower()),
            None
        )
        if v_idx is None:
            print(f"  omitido {txt.name} (variable no reconocida)")
            continue
        key = VARS[v_idx][0]
        names_seen = {s["name"] for s in D[key]}

        for s in series:
            nm = s["name"]
            if nm in names_seen:
                continue
            # Carpeta BASE: solo series STEP1; Carpeta TIP: solo series TIP
            if is_base and not re.search(r'STEP\s*_?1', nm, re.IGNORECASE):
                continue
            if not is_base:
                if re.search(r'STEP', nm, re.IGNORECASE) and \
                   not re.search(r'TIP', nm, re.IGNORECASE):
                    continue
            D[key].append({**s, "is_base": is_base})
            names_seen.add(nm)

for key, series in D.items():
    names = " | ".join(s["name"] for s in series)
    print(f"{key:<7}: {names}")


# =========================================================================
# ESTILOS
# =========================================================================

def _estilo(nombre: str, is_base: bool):
    """Devuelve (etiqueta, color, grosor, estilo_linea) según el nombre."""
    corr = ""
    if re.search(r'sin[_ ]?corr', nombre, re.IGNORECASE): corr = " (no corr.)"
    if re.search(r'con[_ ]?corr', nombre, re.IGNORECASE): corr = " (tip loss + 3D)"
    ls = "--" if "no corr" in corr else "-"

    if is_base:
        return f"Base blade{corr}", [0, 0, 0], 2.0, ls

    t = re.search(r'TIP_?([pm])_?0?(\d+)', nombre, re.IGNORECASE)
    if not t:
        t = re.search(r'([pm])_?0?(\d+)', nombre, re.IGNORECASE)

    lw = 1.4
    if t is None:
        return nombre, [0.5, 0.5, 0.5], lw, ls

    sgn = 1 if t.group(1).lower() == "p" else -1
    pct = int(t.group(2))
    label = f"Tip chord {sgn*pct:+d}%{corr}"

    if sgn > 0:
        col = np.array([0.00, 0.25, 0.65]) + (7 - pct) / 7 * np.array([0.35, 0.35, 0.30])
    else:
        col = np.array([0.60, 0.00, 0.00]) + (7 - pct) / 7 * np.array([0.33, 0.40, 0.35])
    col = np.clip(col, 0, 1).tolist()
    return label, col, lw, ls


# =========================================================================
# FIGURA 2x2
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
fig.suptitle("NREL 5 MW - base blade vs tip chord variants")
axes_flat = axes.flatten()

for idx, (key, header_kw, xlabel, ylabel, factor, titulo) in enumerate(VARS):
    ax = axes_flat[idx]
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(titulo); ax.grid(True)

    for s in D[key]:
        lbl, col, lw, ls = _estilo(s["name"], s["is_base"])
        ax.plot(s["x"], s["y"] * factor, ls, color=col,
                linewidth=lw, label=lbl)

    if idx == 0 and D[key]:
        ax.legend(loc="lower center", fontsize=8, framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT_FILE, dpi=200)
print(f"\nFigura guardada: {OUT_FILE}")
plt.show()
