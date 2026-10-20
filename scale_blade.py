"""
=========================================================================
 STEP 4 - SCALE THE NREL 5 MW BLADE TO A NEW ROTOR RADIUS
 -------------------------------------------------------------------------
 Geometric scaling of a QBlade .bld file by the factor  s = R_new / R_ref:
    - radial positions (POS)        x s
    - chord                         x s
    - in/out-of-plane offsets       x s
    - twist, pitch axis, polars     unchanged (same airfoils, same shape)
 So the hub radius also scales (1.5 m -> 1.5*s), i.e. the WHOLE rotor is
 scaled, which keeps the blade aerodynamically similar (same Cp-TSR curve
 except for Reynolds effects).

 Also prints the turbine parameters to enter in QBlade for the scaled
 rotor (rotor speed range, rated power, expected rated wind speed).

 Output: QBlade_Re6.0/<name>.bld   (next to the polars, so QBlade finds them)
         QBlade_Re6.0/<name>_geometry.png
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

R_REF  = None    # radio de referencia [m]; None = leer de la punta del .bld
R_NEW  = 90.0    # NUEVO radio de rotor [m]   <-- cambia aquí (p.ej. 75, 90)

# Turbina de referencia (NREL 5 MW)
P_REF       = 5e6      # potencia nominal [W]
RPM_MIN_REF = 6.9      # [rpm]
RPM_MAX_REF = 12.1     # [rpm]
TSR_DESIGN  = 7.55
CP_MAX      = 0.473    # Cp máx con tip loss + 3D (resultado STEP 1)
ETA         = 0.944    # eficiencia del generador (Loss Factor 0.056)
RHO         = 1.225    # densidad del aire [kg/m³]

# Potencia nominal de la nueva turbina:
#   None -> P_REF * s^2 (misma velocidad nominal que el NREL 5 MW)
#   número en W, p.ej. 5e6 -> mantiene 5 MW (la V_rated cambia)
P_NEW = 5e6


# =========================================================================
# FUNCIONES AUXILIARES
# =========================================================================

def _replace_token(line: str, n: int, value: float) -> str:
    """Reemplaza el n-ésimo token (1-indexado), conservando espaciado y decimales."""
    tokens  = re.split(r'(\s+)', line)
    result  = list(tokens)
    word_pos = [i for i, t in enumerate(tokens) if t.strip()]

    if n > len(word_pos):
        return line

    target_pos = word_pos[n - 1]
    old_tok    = result[target_pos]
    m = re.search(r'\.(\d+)', old_tok)
    nd = len(m.group(1)) if m else 4
    new_tok = f"{value:.{nd}f}"

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


# =========================================================================
# LECTURA Y ESCALONAMIENTO
# =========================================================================
if not BLADE_FILE.is_file():
    raise FileNotFoundError(f"Archivo de pala no encontrado:\n  {BLADE_FILE}")

lineas = BLADE_FILE.read_text(encoding="utf-8", errors="replace").splitlines()

# Filas de datos: terminan en .plr
idx_filas = [
    i for i, ln in enumerate(lineas)
    if re.search(r'\S+\.plr\s*$', ln)
]

# Radio de referencia (punta de la pala)
if R_REF is None:
    vals_tip = [float(v) for v in lineas[idx_filas[-1]].split()
                if re.match(r'^-?\d', v)]
    R_REF = vals_tip[0]

s    = R_NEW / R_REF
name = f"NREL5MW_scaled_R{R_NEW:.0f}"

# Columnas a escalar: 1=POS, 2=CHORD, 4=OFFSET_X, 5=OFFSET_Y (1-indexado)
COLS_ESCALAR = [1, 2, 4, 5]

r0, c0, tw = [], [], []   # original
r1, c1     = [], []        # escalonado

new_lineas = lineas.copy()
for idx in idx_filas:
    vals = [float(v) for v in lineas[idx].split()
            if re.match(r'^-?\d', v)]
    r0.append(vals[0]);  c0.append(vals[1]);  tw.append(vals[2])

    ln = lineas[idx]
    for col in COLS_ESCALAR:
        ln = _replace_token(ln, col, vals[col - 1] * s)
    new_lineas[idx] = ln

    vals_new = [float(v) for v in ln.split() if re.match(r'^-?\d', v)]
    r1.append(vals_new[0]);  c1.append(vals_new[1])

r0, c0, tw = np.array(r0), np.array(c0), np.array(tw)
r1, c1     = np.array(r1), np.array(c1)

# Actualizar OBJECTNAME
for i, ln in enumerate(new_lineas):
    if "OBJECTNAME" in ln:
        new_lineas[i] = f"{'%-40s' % name} OBJECTNAME"
        break

out_folder = BLADE_FILE.parent
out_file   = out_folder / f"{name}.bld"
out_file.write_text("\n".join(new_lineas) + "\n", encoding="utf-8")

# =========================================================================
# PARAMETROS DE LA TURBINA
# =========================================================================
if P_NEW is None:
    P_NEW = P_REF * s**2

rpm_min = RPM_MIN_REF / s
rpm_max = RPM_MAX_REF / s
A       = np.pi * R_NEW**2
V_rated     = (P_NEW / (ETA * 0.5 * RHO * A * CP_MAX)) ** (1 / 3)
V_rated_ref = (P_REF / (ETA * 0.5 * RHO * np.pi * R_REF**2 * CP_MAX)) ** (1 / 3)
tip_speed   = rpm_max * 2 * np.pi / 60 * R_NEW
Re_factor   = s

print(f"Pala escalonada guardada:\n  {out_file}\n")
print(f"Factor de escala s = {s:.4f}   (R: {R_REF:.2f} -> {R_NEW:.2f} m)")
print(f"Radio del cubo     : {r0[0]:.3f} -> {r1[0]:.3f} m")
print(f"Punta de la pala   : {r0[-1]:.3f} -> {r1[-1]:.3f} m")
print(f"Cuerda máxima      : {c0.max():.3f} -> {c1.max():.3f} m\n")

print("---- Datos de la turbina para QBlade (Turbine BEM -> New) ----")
print(f"  Turbine Blade          : {name}")
print(f"  Power Regulation       : Pitch      Transmission: Variable")
print(f"  Rated Power            : {P_NEW:.0f} W   ({P_NEW/1e6:.2f} MW)")
print(f"  V Cut In / Cut Out     : 3 / 25 m/s")
print(f"  Rot. Speed Min / Max   : {rpm_min:.2f} / {rpm_max:.2f} rpm")
print(f"  TSR at Design Point    : {TSR_DESIGN:.2f}")
print(f"  Loss Factor            : {1-ETA:.3f}      Fixed Pitch: 0   Fixed Losses: 0\n")
print(f"Velocidad nominal esperada : {V_rated:.2f} m/s   (NREL 5 MW: {V_rated_ref:.2f} m/s)")
print(f"Velocidad de punta a max rpm: {tip_speed:.1f} m/s  "
      f"(NREL 5 MW: {RPM_MAX_REF*2*np.pi/60*R_REF:.1f} m/s)")
print(f"Número de Reynolds         : ~{Re_factor:.2f}x el valor NREL "
      f"(p.ej. 6e6 -> {6*Re_factor:.1f}e6)")

# =========================================================================
# GRAFICA
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
fig.suptitle(f"Blade scaling: NREL 5 MW -> R = {R_NEW:.0f} m  (s = {s:.3f})")

axes[0].plot(r0, c0, "k-o", markersize=3, linewidth=1.5,
             label=f"NREL 5 MW (R = {R_REF:.0f} m)")
axes[0].plot(r1, c1, "-o", color=[0, 0.447, 0.741], markersize=3, linewidth=1.5,
             label=f"Scaled (R = {R_NEW:.0f} m)")
axes[0].set_xlabel("Radius r [m]"); axes[0].set_ylabel("Chord c [m]")
axes[0].set_title("Chord distribution"); axes[0].legend(loc="upper right")
axes[0].grid(True)

axes[1].plot(r0, tw, "k-o", markersize=3, linewidth=1.5,
             label=f"NREL 5 MW (R = {R_REF:.0f} m)")
axes[1].plot(r1, tw, "-o", color=[0, 0.447, 0.741], markersize=3, linewidth=1.5,
             label=f"Scaled (R = {R_NEW:.0f} m)")
axes[1].set_xlabel("Radius r [m]"); axes[1].set_ylabel("Twist angle [deg]")
axes[1].set_title("Twist distribution (unchanged)"); axes[1].legend(loc="upper right")
axes[1].grid(True)

fig.tight_layout()
out_png = out_folder / f"{name}_geometry.png"
fig.savefig(out_png, dpi=200)
print(f"\nFigura guardada: {out_png}")
plt.show()
