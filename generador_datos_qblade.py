"""
=========================================================================
 PREPARAR POLARES Y PALA PARA QBLADE - NREL 5 MW
 -------------------------------------------------------------------------
 Usa las polares calculadas con XFOIL (calculo_automatico_xfoil.py) y
 genera los archivos que QBlade importa directamente:

   STEP 1 -> polares limpias (N = 9)
   STEP 2 -> polares interpoladas  0.7*limpia(N=9) + 0.3*turbulenta(N=1)

 Para cada perfil:
   1) Lee las polares N1 y N9 (.pol de XFOIL)
   2) STEP 2: mezcla 0.7*N9 + 0.3*N1 sobre una malla común de alfa
   3) Extrapola a 360° (-180..180) con el método de Viterna
   4) Escribe un archivo .plr (formato "QBlade Multi RE Polar File")
 Después escribe dos palas .bld (STEP1 y STEP2) que apuntan a esas polares.

 Si exportas tu pala actual desde QBlade (.bld), el script la usa como base
 y solo sustituye la columna POLAR_FILE (geometría idéntica a la tuya).
 Si no, genera la pala con la tabla de Jonkman et al. (2009).

 En QBlade: Blade Design -> Import Blade -> elegir el .bld generado.
=========================================================================
"""

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# CONFIGURACION
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent     # carpeta "CALCULO AUTOMATICO"

RE          = 6e6                           # Re de las polares a usar
ETIQUETA_RE = f"{RE / 1e6:.1f}"            # "6.0"
CARPETA_POL = BASE_DIR / ETIQUETA_RE

PESO_LIMPIA = 0.7                           # STEP 2: 0.7*N9 + 0.3*N1
PESO_TURB   = 0.3
N_LIMPIA    = 9
N_TURB      = 1

D_ALFA_MALLA = 0.25                         # paso de la malla común [°]
AR           = 17                           # alargamiento (NREL 5 MW)
CD_MAX       = 1.11 + 0.018 * AR           # CD a 90° para Viterna (= 1.416)
ALFA_LIMITES = (-np.inf, np.inf)            # recorta datos XFOIL si hace falta

# Pala base exportada desde QBlade (dejar None para usar la tabla de Jonkman)
BLD_BASE = BASE_DIR / "Qblade" / "New_Blade.bld"

# Perfiles: (carpeta, prefijo .pol, nombre en QBlade, espesor [%])
PERFILES = [
    ("DU40",   "DU40_A17",   "DU40_A17",   40),
    ("DU35",   "DU35_A17",   "DU35_A17",   35),
    ("DU30",   "DU30_A17",   "DU30_A17",   30),
    ("DU25",   "DU25_A17",   "DU25_A17",   25),
    ("DU21",   "DU21_A17",   "DU21_A17",   21),
    ("Naca64", "NACA64_A17", "NACA64_A17", 18),
]

CARPETA_SAL = BASE_DIR / f"QBlade_Re{ETIQUETA_RE}"
CARPETA_SAL.mkdir(exist_ok=True)


# =========================================================================
# FUNCIONES DE LECTURA
# =========================================================================

def leer_polar(ruta: Path) -> np.ndarray:
    """
    Lee un .pol de XFOIL.
    Devuelve array (N, 4): [alfa, CL, CD, CM], ordenado y sin repetidos.
    """
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe la polar:\n  {ruta}")

    lineas = ruta.read_text(encoding="utf-8", errors="replace").splitlines()

    idx_ini = None
    for i, ln in enumerate(lineas):
        if "------" in ln:
            idx_ini = i
            break
    if idx_ini is None:
        raise ValueError(f"Polar sin cabecera reconocible: {ruta}")

    datos = []
    for ln in lineas[idx_ini + 1:]:
        vals = ln.split()
        if len(vals) >= 5:
            try:
                # columnas XFOIL: alfa CL CD CDp CM  → tomamos [0,1,2,4]
                datos.append([float(vals[0]), float(vals[1]),
                               float(vals[2]), float(vals[4])])
            except ValueError:
                continue

    if len(datos) < 10:
        raise ValueError(f"Polar casi vacía ({len(datos)} puntos): {ruta}")

    arr = np.array(datos)
    _, idx = np.unique(arr[:, 0], return_index=True)
    return arr[idx]                             # [alfa, CL, CD, CM]


def recortar(P: np.ndarray, lim: tuple) -> np.ndarray:
    """Recorta la polar al rango de ángulos [lim[0], lim[1]]."""
    return P[(P[:, 0] >= lim[0]) & (P[:, 0] <= lim[1])]


# =========================================================================
# EXTRAPOLACION DE VITERNA
# =========================================================================

def _coef_viterna(alpha_s: float, CL_s: float, CD_s: float,
                  CD_max: float) -> dict:
    """Calcula los coeficientes A2 y B2 de Viterna ajustados al punto (alpha_s, CL_s, CD_s)."""
    sin_s = np.sin(np.deg2rad(alpha_s))
    cos_s = np.cos(np.deg2rad(alpha_s))
    A2 = (CL_s - CD_max * sin_s * cos_s) * sin_s / cos_s**2
    B2 = (CD_s - CD_max * sin_s**2) / cos_s
    return {"CD_max": CD_max, "A2": A2, "B2": B2}


def _eval_viterna(alpha: np.ndarray, v: dict):
    """Evalúa las fórmulas de Viterna en el vector alpha (en grados, > 0)."""
    al = np.deg2rad(np.maximum(alpha, 1e-3))
    CL = v["CD_max"] / 2 * np.sin(2 * al) + v["A2"] * np.cos(al)**2 / np.sin(al)
    CD = v["CD_max"] * np.sin(al)**2 + v["B2"] * np.cos(al)
    return CL, CD


def extrapolar_viterna(a: np.ndarray, C: np.ndarray,
                       CD_max: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Extiende una polar [alfa, CL, CD, CM] a -180..180° (método Viterna & Corrigan).

    Parámetros
    ----------
    a     : vector de ángulos de ataque [°], debe contener valores negativos y positivos
    C     : array (N, 3) con columnas [CL, CD, CM]
    CD_max: CD a 90° (parámetro de Viterna)

    Devuelve
    --------
    A_ext : vector de ángulos extendido [-180..180]
    T_ext : array (M, 3) con [CL, CD, CM] extendidos
    """
    if a[0] >= 0 or a[-1] <= 0:
        raise ValueError(
            f"La polar debe cubrir alfa negativos y positivos "
            f"(rango actual: {a[0]:.1f}° a {a[-1]:.1f}°)"
        )

    aL, aH = float(a[0]), float(a[-1])
    CLL, CDL, CML = C[0, 0],  C[0, 1],  C[0, 2]
    CLH, CDH, CMH = C[-1, 0], C[-1, 1], C[-1, 2]
    CD0  = float(np.min(C[:, 1]))
    paso = 1.0

    vH = _coef_viterna(aH,   CLH,  CDH, CD_max)   # lado positivo
    vL = _coef_viterna(-aL, -CLL,  CDL, CD_max)   # lado negativo (espejo)

    # ---- Lado positivo: aH → 90 → (180-aH) → 180
    s1 = np.arange(np.ceil(aH) + paso, 90 + paso, paso)
    s1 = s1[s1 > aH]
    s2 = np.arange(91, 180 - aH + paso, paso)
    s3 = np.arange(np.ceil(180 - aH), 180 + paso, paso)
    s3 = s3[(s3 > 180 - aH) & (s3 > 90)]

    cl1, cd1 = _eval_viterna(s1, vH)
    cl2, cd2 = _eval_viterna(180 - s2, vH)
    cl2 = -0.7 * cl2

    clB = -0.7 * CLH
    cl3 = clB * (180 - s3) / aH
    cd3 = CD0 + (CDH - CD0) * (180 - s3) / aH

    # ---- Lado negativo: aL → -90 → -(180-|aL|) → -180
    aLa = -aL   # positivo
    t1  = np.arange(np.floor(aL), -90 - paso, -paso)
    t1  = t1[t1 < aL]
    t2  = np.arange(-91, -180 + aLa - paso, -paso)
    t3  = np.arange(np.floor(-180 + aLa), -180 - paso, -paso)
    t3  = t3[(t3 < -180 + aLa) & (t3 < -90)]

    m1, d1 = _eval_viterna(-t1, vL);  m1 = -m1
    m2, d2 = _eval_viterna(180 + t2, vL);  m2 = 0.7 * m2
    m3 = 0.7 * (-CLL) * (180 + t3) / aLa
    d3 = CD0 + (CDL - CD0) * (180 + t3) / aLa

    A_ext = np.concatenate([t3[::-1], t2[::-1], t1[::-1], a, s1, s2, s3])
    CL_ext = np.concatenate([m3[::-1], m2[::-1], m1[::-1], C[:, 0], cl1, cl2, cl3])
    CD_ext = np.concatenate([d3[::-1], d2[::-1], d1[::-1], C[:, 1], cd1, cd2, cd3])

    # CM: dentro del rango usa los datos; fuera, aproximación de placa plana
    # con continuidad forzada en los extremos
    CN  = CL_ext * np.cos(np.deg2rad(A_ext)) + CD_ext * np.sin(np.deg2rad(A_ext))
    xcp = 0.25 + 0.175 * (2 * np.abs(np.mod(A_ext + 90, 180) - 90) / 180)
    CMp = -CN * (xcp - 0.25)

    CM_ext         = CMp.copy()
    dentro         = (A_ext >= aL) & (A_ext <= aH)
    CM_ext[dentro] = C[:, 2]

    offH = CMH - float(np.interp(aH, A_ext, CMp))
    offL = CML - float(np.interp(aL, A_ext, CMp))
    pos  = A_ext > aH
    neg  = A_ext < aL
    CM_ext[pos] = CMp[pos] + offH * np.maximum(0, 1 - (A_ext[pos] - aH) / (90 - aH))
    CM_ext[neg] = CMp[neg] + offL * np.maximum(0, 1 - (aL - A_ext[neg]) / (90 + aL))

    # Asegurar que empieza en -180 y termina en 180
    if A_ext[0] > -180:
        A_ext  = np.concatenate([[-180.0], A_ext])
        CL_ext = np.concatenate([[0.0],    CL_ext])
        CD_ext = np.concatenate([[CD0],    CD_ext])
        CM_ext = np.concatenate([[0.0],    CM_ext])
    if A_ext[-1] < 180:
        A_ext  = np.concatenate([A_ext,  [180.0]])
        CL_ext = np.concatenate([CL_ext, [0.0]])
        CD_ext = np.concatenate([CD_ext, [CD0]])
        CM_ext = np.concatenate([CM_ext, [0.0]])

    T_ext = np.column_stack([CL_ext, CD_ext, CM_ext])
    return A_ext, T_ext


# =========================================================================
# ESCRITURA DE ARCHIVOS QBLADE
# =========================================================================

def _separador(texto: str = "", ancho: int = 116) -> str:
    base = "-" * 40 + texto
    return base.ljust(ancho, "-")


def escribir_plr(ruta: Path, nom_polar: str, nom_perfil: str,
                 espesor: float, Re: float,
                 A: np.ndarray, T: np.ndarray) -> None:
    """
    Escribe un archivo .plr en formato "QBlade Multi RE Polar File".

    Parámetros
    ----------
    ruta      : ruta de salida del archivo
    nom_polar : nombre de la polar (POLARNAME)
    nom_perfil: nombre del perfil (FOILNAME)
    espesor   : espesor relativo en % (THICKNESS)
    Re        : número de Reynolds
    A         : vector de ángulos de ataque [°]
    T         : array (N, 3) con [CL, CD, CM]
    """
    ahora = datetime.now()
    lineas = [
        _separador("QBlade Multi RE Polar File"),
        "Generated with : Python (generador_datos_qblade.py)",
        "Archive Format: 310001",
        f"Time : {ahora.strftime('%H:%M:%S')}",
        f"Date : {ahora.strftime('%d.%m.%Y')}",
        "",
        _separador("Object Names"),
        f"{'%-40s' % nom_polar} {'POLARNAME':<18} - the polar name",
        f"{'%-40s' % nom_perfil} {'FOILNAME':<18} - the airfoil name to which the polar(s) belong",
        "",
        _separador("Parameters"),
        f"{'%-40s' % f'{espesor:.1f}'} {'THICKNESS':<18} - the airfoil thickness",
        f"{'%-40s' % '0'} {'ISDECOMPOSED':<18} - is the polar decomposed (add Cl_Sep, Cl_att and f_st columns)",
        f"{'%-20s' % 'REYNOLDS'}{'%-20s' % f'{Re:.4E}'}- the list of Reynolds numbers for the imported polars",
        "",
        _separador("Polar Data"),
        f"{'AOA':<20}{'CL':<20}{'CD':<20}{'CM':<20}",
    ]

    datos = "\n".join(
        f"{row[0]:<20.6f}{row[1]:<20.6f}{row[2]:<20.6f}{row[3]:<20.6f}"
        for row in np.column_stack([A, T])
    )

    ruta.write_text("\n".join(lineas) + "\n" + datos + "\n", encoding="utf-8")


def pala_jonkman() -> tuple[np.ndarray, list[str]]:
    """
    Geometría NREL 5 MW de Jonkman et al. (2009), tabla 3-1.

    Devuelve
    --------
    pala   : array (19, 3) con columnas [r [m], twist [°], chord [m]]
    perfiles: lista de 19 nombres de perfil por estación
    """
    pala = np.array([
        [ 1.5000, 13.308, 3.542],
        [ 2.8667, 13.308, 3.542],
        [ 5.6000, 13.308, 3.854],
        [ 8.3333, 13.308, 4.167],
        [11.7500, 13.308, 4.557],
        [15.8500, 11.480, 4.652],
        [19.9500, 10.162, 4.458],
        [24.0500,  9.011, 4.249],
        [28.1500,  7.795, 4.007],
        [32.2500,  6.544, 3.748],
        [36.3500,  5.361, 3.502],
        [40.4500,  4.188, 3.256],
        [44.5500,  3.125, 3.010],
        [48.6500,  2.319, 2.764],
        [52.7500,  1.526, 2.518],
        [56.1667,  0.863, 2.313],
        [58.9000,  0.370, 2.086],
        [61.6333,  0.106, 1.419],
        [63.0000,  0.000, 1.419],
    ])
    perfiles = [
        "Cylinder1", "Cylinder1", "Cylinder1", "Cylinder2",
        "DU40", "DU35", "DU35",
        "DU30", "DU25", "DU25", "DU21", "DU21",
        "Naca64", "Naca64", "Naca64", "Naca64", "Naca64", "Naca64", "Naca64",
    ]
    return pala, perfiles


def escribir_bld(ruta: Path, nombre: str,
                 pala: np.ndarray, perfiles: list[str],
                 mapa_polar: dict) -> None:
    """
    Escribe un archivo .bld en formato "QBlade Blade Definition File".

    Parámetros
    ----------
    ruta      : ruta de salida
    nombre    : nombre del objeto en QBlade (OBJECTNAME)
    pala      : array (N, 3) con [r, twist, chord]
    perfiles  : lista de N strings con el nombre de perfil por estación
    mapa_polar: dict {nombre_perfil: nombre_archivo.plr}
    """
    lineas = [
        _separador("QBlade Blade Definition File"),
        "Generated with : Python (generador_datos_qblade.py)",
        "Archive Format: 310002",
        "",
        _separador("Object Name"),
        f"{'%-40s' % nombre} OBJECTNAME",
        "",
        _separador("Parameters"),
        f"{'%-40s' % 'HAWT'} ROTORTYPE",
        f"{'%-40s' % '3'} NUMBLADES",
        "",
        _separador("Blade Data"),
        f"{'POS [m]':<11}{'CHORD [m]':<12}{'TWIST [deg]':<13}"
        f"{'OFFSET_X [m]':<14}{'OFFSET_Y [m]':<14}{'TAXIS [-]':<11}POLAR_FILE",
    ]

    for i, fila in enumerate(pala):
        r, twist, chord = fila
        plr = mapa_polar[perfiles[i]]
        lineas.append(
            f"{r:<11.4f}{chord:<12.4f}{twist:<13.4f}"
            f"{0.0:<14.4f}{0.0:<14.4f}{0.5:<11.4f}{plr}"
        )

    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def cambiar_polares_bld(entrada: Path, salida: Path,
                        nombre: str, mapa_polar: dict) -> None:
    """
    Copia un .bld exportado por QBlade y sustituye la columna POLAR_FILE
    según el perfil que aparece en el nombre del .plr original.
    """
    claves = [
        (r"(cylinder|circle)_?1|CD_?0\.50", "Cylinder1"),
        (r"(cylinder|circle)_?2|CD_?0\.35", "Cylinder2"),
        (r"du[_ -]?40",                     "DU40"),
        (r"du[_ -]?35",                     "DU35"),
        (r"du[_ -]?30",                     "DU30"),
        (r"du[_ -]?25",                     "DU25"),
        (r"du[_ -]?21",                     "DU21"),
        (r"na(ca)?[_ -]?64",                "Naca64"),
    ]

    lineas = entrada.read_text(encoding="utf-8", errors="replace").splitlines()
    en_tabla = False
    nuevas   = []

    for ln in lineas:
        if "POLAR_FILE" in ln:
            en_tabla = True
            nuevas.append(ln)
            continue
        if "OBJECTNAME" in ln:
            nuevas.append(f"{'%-40s' % nombre} OBJECTNAME")
            continue
        if not en_tabla or not ln.strip():
            nuevas.append(ln)
            continue

        # La polar es siempre la última palabra de la línea
        m = re.match(r'^(.*\s)(\S+\.plr)\s*$', ln)
        if not m:
            nuevas.append(ln)
            continue

        prefijo, plr_viejo = m.group(1), m.group(2)
        nuevo_plr = ""
        for patron, clave in claves:
            if re.search(patron, plr_viejo, re.IGNORECASE):
                nuevo_plr = mapa_polar.get(clave, "")
                break

        if not nuevo_plr:
            print(f"  AVISO: estación sin perfil reconocido, se deja igual: {plr_viejo}")
            nuevas.append(ln)
        else:
            nuevas.append(prefijo + nuevo_plr)

    salida.write_text("\n".join(nuevas) + "\n", encoding="utf-8")


# =========================================================================
# BUCLE PRINCIPAL - POLARES
# =========================================================================
n_perf    = len(PERFILES)
plr_step1 = {}   # {carpeta: nombre_archivo.plr}
plr_step2 = {}

fig, axes = plt.subplots(2, n_perf, figsize=(4 * n_perf, 7))
fig.suptitle("Polares 360° — STEP1 (azul) vs STEP2 (naranja)")

for p, (carp, pref, nom_q, esp) in enumerate(PERFILES):

    # ---- Leer polares N9 (limpia) y N1 (turbulenta)
    ruta_n9 = CARPETA_POL / carp / f"N{N_LIMPIA}" / \
              f"{pref}_Clean_Re{ETIQUETA_RE}_M0-2_n{N_LIMPIA}.pol"
    ruta_n1 = CARPETA_POL / carp / f"N{N_TURB}" / \
              f"{pref}_Clean_Re{ETIQUETA_RE}_M0-2_n{N_TURB}.pol"

    P9 = recortar(leer_polar(ruta_n9), ALFA_LIMITES)
    P1 = recortar(leer_polar(ruta_n1), ALFA_LIMITES)

    # ---- STEP 1: polar limpia (N=9) en malla común
    a_min1 = np.ceil( P9[0,  0] / D_ALFA_MALLA) * D_ALFA_MALLA
    a_max1 = np.floor(P9[-1, 0] / D_ALFA_MALLA) * D_ALFA_MALLA
    a1     = np.arange(a_min1, a_max1 + D_ALFA_MALLA / 2, D_ALFA_MALLA)
    C1     = np.column_stack([
        np.interp(a1, P9[:, 0], P9[:, i]) for i in [1, 2, 3]
    ])
    A1, T1 = extrapolar_viterna(a1, C1, CD_MAX)

    # ---- STEP 2: 0.7*N9 + 0.3*N1 en rango común de ambas polares
    a_min2 = max(P9[0, 0],  P1[0, 0])
    a_max2 = min(P9[-1, 0], P1[-1, 0])
    a_min2 = np.ceil( a_min2 / D_ALFA_MALLA) * D_ALFA_MALLA
    a_max2 = np.floor(a_max2 / D_ALFA_MALLA) * D_ALFA_MALLA
    a2     = np.arange(a_min2, a_max2 + D_ALFA_MALLA / 2, D_ALFA_MALLA)
    C2     = np.column_stack([
        PESO_LIMPIA * np.interp(a2, P9[:, 0], P9[:, i]) +
        PESO_TURB   * np.interp(a2, P1[:, 0], P1[:, i])
        for i in [1, 2, 3]
    ])
    A2, T2 = extrapolar_viterna(a2, C2, CD_MAX)

    # ---- Escribir .plr
    n1 = f"{nom_q}_Re{ETIQUETA_RE}_N{N_LIMPIA}"
    n2 = (f"{nom_q}_Re{ETIQUETA_RE}_"
          f"{round(100*PESO_LIMPIA):02d}N{N_LIMPIA}_"
          f"{round(100*PESO_TURB):02d}N{N_TURB}")

    escribir_plr(CARPETA_SAL / f"{n1}.plr", n1, nom_q, esp, RE, A1, T1)
    escribir_plr(CARPETA_SAL / f"{n2}.plr", n2, nom_q, esp, RE, A2, T2)
    plr_step1[carp] = f"{n1}.plr"
    plr_step2[carp] = f"{n2}.plr"
    print(f"{carp:<7}  STEP1: {n1}.plr")
    print(f"         STEP2: {n2}.plr")

    # ---- Gráficas de control
    ax_cl = axes[0, p]
    ax_cd = axes[1, p]
    ax_cl.plot(A1, T1[:, 0], "b-",  label="STEP1 N9")
    ax_cl.plot(A2, T2[:, 0], "r--", label=f"STEP2 {PESO_LIMPIA:.1f}N9+{PESO_TURB:.1f}N1")
    ax_cl.set_title(carp); ax_cl.set_xlim(-180, 180); ax_cl.grid(True)
    if p == 0: ax_cl.set_ylabel("$C_L$")
    ax_cd.plot(A1, T1[:, 1], "b-")
    ax_cd.plot(A2, T2[:, 1], "r--")
    ax_cd.set_xlabel(r"$\alpha$ [°]"); ax_cd.set_xlim(-180, 180); ax_cd.grid(True)
    if p == 0: ax_cd.set_ylabel("$C_D$")

axes[0, 0].legend(loc="upper left", fontsize=7)
fig.tight_layout()
fig.savefig(CARPETA_SAL / "polares_360.png", dpi=150)
print(f"\nFigura polares guardada: {CARPETA_SAL / 'polares_360.png'}")

# ---- Cilindros de la raíz
CILINDROS = [("Cylinder1", 0.50), ("Cylinder2", 0.35)]
for nom_cil, cd_cil in CILINDROS:
    aC = np.arange(-180, 181, 5, dtype=float)
    Tc = np.column_stack([
        np.zeros_like(aC),
        cd_cil * np.ones_like(aC),
        np.zeros_like(aC),
    ])
    nC = f"{nom_cil}_Cd{cd_cil:.2f}"
    escribir_plr(CARPETA_SAL / f"{nC}.plr", nC, nom_cil, 100.0, RE, aC, Tc)
    plr_step1[nom_cil] = f"{nC}.plr"
    plr_step2[nom_cil] = f"{nC}.plr"


# =========================================================================
# PALAS (.bld)
# =========================================================================
if not BLD_BASE.is_file():
    pala, perf_est = pala_jonkman()
    escribir_bld(CARPETA_SAL / "NREL5MW_STEP1_N9.bld",
                 "NREL5MW_STEP1_N9", pala, perf_est, plr_step1)
    escribir_bld(CARPETA_SAL / "NREL5MW_STEP2_interp.bld",
                 "NREL5MW_STEP2_interp", pala, perf_est, plr_step2)
else:
    cambiar_polares_bld(BLD_BASE, CARPETA_SAL / "NREL5MW_STEP1_N9.bld",
                        "NREL5MW_STEP1_N9", plr_step1)
    cambiar_polares_bld(BLD_BASE, CARPETA_SAL / "NREL5MW_STEP2_interp.bld",
                        "NREL5MW_STEP2_interp", plr_step2)

print(f"\nArchivos listos em:\n  {CARPETA_SAL}")
plt.show()
