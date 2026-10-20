"""
=========================================================================
 TIP CHORD SENSITIVITY - BLADE FILE GENERATOR - NREL 5 MW (clean polars)
 -------------------------------------------------------------------------
 Takes the clean-polar blade (NREL5MW_STEP1_N9.bld) and creates new .bld
 files in which the chord of the TIP region is scaled by +5%, -5%,
 +7% and -7%. Everything else (positions, twist, offsets, pitch axis,
 polars) is copied unchanged, so only the chord effect is studied.

 Tip region definition (editable below):
    r <  R_RAMP           -> chord unchanged            (factor = 1)
    R_RAMP <= r < R_TIP   -> linear transition          (avoids a step)
    r >= R_TIP            -> full change                (factor = 1 + dc)
 Set R_RAMP = R_TIP for an abrupt change.

 The new .bld files are written in the same folder as the base blade,
 next to the .plr polars, so QBlade finds them when importing.
=========================================================================
"""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# CONFIGURACION
# -------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.parent    # carpeta "CALCULO AUTOMATICO"
BLADE_FILE = BASE_DIR / "QBlade_Re6.0" / "NREL5MW_STEP1_N9.bld"

CHORD_CHANGES = [+5, -5, +7, -7]   # cambios de corda na punta [%]

R       = 63.0          # radio del rotor [m]
R_TIP   = 0.80 * R      # cambio completo desde aquí hasta la punta  (r/R = 0.80)
R_RAMP  = 0.70 * R      # la transición lineal empieza aquí          (r/R = 0.70)


# =========================================================================
# LECTURA DEL BLADE BASE
# =========================================================================

def _replace_token(line: str, n: int, value: float) -> str:
    """
    Reemplaza el n-ésimo token (1-indexado) de una línea numérica,
    conservando el espaciado original y el número de decimales.
    """
    tokens = re.split(r'(\s+)', line)        # alterna [token, sep, token, sep, ...]
    words  = [t for t in tokens if t.strip()]
    spaces = [t for t in tokens if not t.strip()]

    # Reconstruir la lista intercalada de palabras y separadores
    # para poder sustituir la n-ésima palabra (1-indexed)
    idx_word = 0
    result   = []
    word_pos = []
    for t in tokens:
        if t.strip():
            idx_word += 1
            word_pos.append(len(result))
        result.append(t)

    if n > idx_word:
        return line

    # Detectar número de decimales del token original
    target_pos = word_pos[n - 1]
    old_tok    = result[target_pos]
    m = re.search(r'\.(\d+)', old_tok)
    nd = len(m.group(1)) if m else 4
    new_tok = f"{value:.{nd}f}"

    # Ajustar el separador siguiente para mantener el ancho de columna
    d = len(new_tok) - len(old_tok)
    sep_pos = target_pos + 1
    if sep_pos < len(result):
        sep = result[sep_pos]
        if d > 0 and len(sep) > d + 1:
            result[sep_pos] = sep[:len(sep) - d]
        elif d < 0:
            result[sep_pos] = sep + " " * (-d)

    result[target_pos] = new_tok
    return "".join(result)


def leer_bld(ruta: Path):
    """
    Lee un .bld de QBlade.
    Devuelve:
        lineas    : lista de strings (todo el archivo)
        idx_filas : índices de las filas de datos (las que tienen .plr al final)
        r         : array de posiciones radiales [m]
        c0        : array de cuerda base [m]
    """
    if not ruta.is_file():
        raise FileNotFoundError(
            f"Archivo de pala no encontrado:\n  {ruta}\n"
            "Ejecuta primero generador_datos_qblade.py"
        )
    lineas = ruta.read_text(encoding="utf-8", errors="replace").splitlines()

    # Encontrar el índice de la cabecera POLAR_FILE
    i_header = next((i for i, ln in enumerate(lineas) if "POLAR_FILE" in ln), None)
    if i_header is None:
        raise ValueError(f'No se encontró la cabecera "POLAR_FILE" en {ruta}')

    idx_filas = [
        i for i in range(i_header + 1, len(lineas))
        if re.search(r'\S+\.plr\s*$', lineas[i])
    ]

    r  = np.array([float(lineas[i].split()[0]) for i in idx_filas])
    c0 = np.array([float(lineas[i].split()[1]) for i in idx_filas])
    return lineas, idx_filas, r, c0


# =========================================================================
# LECTURA
# =========================================================================
lineas, idx_filas, r, c0 = leer_bld(BLADE_FILE)
out_folder = BLADE_FILE.parent
base_name  = BLADE_FILE.stem

print(f"Pala base: {base_name}  ({len(r)} estaciones, r = {r[0]:.2f} - {r[-1]:.2f} m)")

# Factor de ponderación por estación: 0 (sin cambio) ... 1 (cambio completo)
if R_TIP > R_RAMP:
    w = np.clip((r - R_RAMP) / (R_TIP - R_RAMP), 0.0, 1.0)
else:
    w = (r >= R_TIP).astype(float)

n_afectadas = int(np.sum(w > 0))
print(f"Estaciones afectadas: {n_afectadas}  "
      f"(transición desde r = {R_RAMP:.1f} m, cambio completo desde r = {R_TIP:.1f} m)\n")

# =========================================================================
# ESCRITURA DE LAS NUEVAS PALAS
# =========================================================================
chords_all = np.zeros((len(r), len(CHORD_CHANGES)))
names      = []

for j, dc in enumerate(CHORD_CHANGES):
    factor = 1.0 + w * dc / 100.0
    c      = c0 * factor
    chords_all[:, j] = c

    tag  = f"{'p' if dc >= 0 else 'm'}{abs(dc):02d}"   # p05, m05, p07, m07
    name = f"NREL5MW_TIP_{tag}"
    names.append(name)

    new_lines = lineas.copy()

    # Sustituir la cuerda (2ª columna) en cada fila de datos
    for k, idx in enumerate(idx_filas):
        new_lines[idx] = _replace_token(lineas[idx], 2, c[k])

    # Actualizar el nombre del objeto
    for i, ln in enumerate(new_lines):
        if "OBJECTNAME" in ln:
            new_lines[i] = f"{'%-40s' % name} OBJECTNAME"
            break

    out_file = out_folder / f"{name}.bld"
    out_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"{dc:+3d}%  ->  {name}.bld   "
          f"(cuerda en punta: {c0[-1]:.3f} -> {c[-1]:.3f} m)")

# =========================================================================
# TABLA DE COMPROBACION
# =========================================================================
header = f"{'r [m]':>8} {'r/R':>7} {'Base':>8}"
for n in names:
    header += f" {n:>9}"
print(f"\n{header}")
for k in range(len(r)):
    if w[k] == 0:
        continue
    row = f"{r[k]:8.3f} {r[k]/R:7.3f} {c0[k]:8.3f}"
    for j in range(len(CHORD_CHANGES)):
        row += f" {chords_all[k, j]:9.3f}"
    print(row)

# =========================================================================
# GRAFICA
# =========================================================================
COLORES = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.466, 0.674, 0.188],
    [0.494, 0.184, 0.556],
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("NREL 5 MW - tip chord variants")

for ax in axes:
    ax.plot(r, c0, "k-o", markersize=3, linewidth=1.8, label="Base (clean polars)")
    for j, dc in enumerate(CHORD_CHANGES):
        ax.plot(r, chords_all[:, j], "-", color=COLORES[j], linewidth=1.3,
                label=f"Tip chord {dc:+d}%")
    ax.axvline(R_RAMP, linestyle=":", color="k", linewidth=0.8)
    ax.axvline(R_TIP,  linestyle="--", color="k", linewidth=0.8)
    ax.set_xlabel("Radius r [m]")
    ax.set_ylabel("Chord c [m]")
    ax.grid(True)

axes[0].set_title("Chord distribution")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_xlim(0, R)

axes[1].set_title("Tip region (zoom)")
axes[1].set_xlim(R_RAMP - 0.1 * R, R)

fig.tight_layout()
out_png = out_folder / "Tip_chord_variants.png"
fig.savefig(out_png, dpi=200)
print(f"\nFigura guardada: {out_png}")
print(f"\nArchivos escritos en:\n  {out_folder}")
plt.show()
