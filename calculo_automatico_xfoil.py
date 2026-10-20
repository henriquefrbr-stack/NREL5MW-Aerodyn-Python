"""
=========================================================================
 CALCULO AUTOMATICO DE POLARES CON XFOIL - NREL 5 MW
 -------------------------------------------------------------------------
 Para cada perfil de la pala (DU21, DU25, DU30, DU35, DU40, NACA64):
   1) Crea la carpeta  <Re>/<Perfil>/N<n>/
   2) Copia las coordenadas del perfil
   3) Escribe el script Xfoil_WTTpolar.xfl con el Re y el Ncrit del caso
   4) Ejecuta XFOIL (con límite de tiempo por si se queda colgado)
   5) Lee la polar resultante (.pol)
 Casos:
   - Turbulento    : Ncrit = 1
   - No turbulento : Ncrit = 9
 Al final dibuja Cl, Cd y Cl/Cd de ambos casos para cada perfil.

 Coloca este script en la carpeta "CALCULO AUTOMATICO" y ejecútalo.
=========================================================================
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# CONFIGURACION
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent          # carpeta "CALCULO AUTOMATICO"

RE       = 6e6       # número de Reynolds
MACH     = 0.2
N_PANEL  = 200       # paneles (ppar)
ITER_MAX = 200       # iteraciones viscosas
ALFA_MAX = 32        # barrido de 0 a +ALFA_MAX y de -0.1 a -ALFA_MAX
D_ALFA   = 0.1
TIMEOUT  = 600       # segundos máximos por cálculo de XFOIL

# Casos de transición: (etiqueta, Ncrit)
CASOS = [
    ("Turbulento",    1),
    ("No turbulento", 9),
]

# Perfiles: (nombre de carpeta, archivo de coordenadas, prefijo de la polar)
PERFILES = [
    ("DU21",   "DU21_A17_coords.txt",   "DU21_A17"),
    ("DU25",   "DU25_A17_coords.txt",   "DU25_A17"),
    ("DU30",   "DU30_A17_coords.txt",   "DU30_A17"),
    ("DU35",   "DU35_A17_coords.txt",   "DU35_A17"),
    ("DU40",   "DU40_A17_coords.txt",   "DU40_A17"),
    ("Naca64", "NACA64_A17_coords.txt", "NACA64_A17"),
]

# De dónde se toman las coordenadas y el ejecutable
CARPETA_ORIGEN = BASE_DIR / "0.75Re"
XFOIL_EXE      = CARPETA_ORIGEN / "DU30" / "N9" / "xfoil.exe"

# Carpeta de salida y etiqueta del Re en los nombres (e.g. 6e6 -> "6.0")
ETIQUETA_RE   = f"{RE / 1e6:.1f}"
CARPETA_SAL   = BASE_DIR / ETIQUETA_RE


# =========================================================================
# FUNCIONES
# =========================================================================

def escribir_xfl(ruta: Path, coords: str, pol: str,
                 re: float, mach: float, ncrit: int,
                 npanel: int, iter_max: int,
                 alfa_max: float, d_alfa: float) -> None:
    """Escribe el script de comandos para XFOIL (.xfl)."""
    lines = [
        f"load ./{coords}",
        "NREL5MW",
        "",
        "ppar",
        "n",
        str(npanel),
        "",
        "",
        "oper",
        "visc",
        f"{int(re)}",          # 6000000
        "m",
        f"{mach:g}",
        "iter",
        str(iter_max),
        "vpar",
        "xtr",
        "1",
        "1",
        "n",
        f"{ncrit:g}",
        "",
        "",
        "",
        "oper",
        "alfa 0",
        "pacc",
        pol,
        "",
        f"aseq 0 {alfa_max:g} {d_alfa:g}",
        "pacc",
        "init",
        "alfa 0",
        "pacc",
        pol,
        "",
        f"aseq {-d_alfa:g} {-alfa_max:g} {d_alfa:g}",
        "pacc",
        "",
        "",
        "wdef",
        "",
        "quit",
    ]
    ruta.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")


def ejecutar_xfoil(exe: Path, dir_trabajo: Path,
                   archivo_xfl: Path, archivo_log: Path,
                   timeout: int) -> bool:
    """
    Lanza XFOIL en dir_trabajo con el .xfl como stdin.
    Devuelve True si terminó dentro del timeout, False si se mató por tiempo.
    """
    with open(archivo_xfl, "r", encoding="ascii") as stdin_f, \
         open(archivo_log, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            [str(exe)],
            stdin=stdin_f,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(dir_trabajo),
        )
        try:
            proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return False


def leer_polar(ruta: Path) -> np.ndarray:
    """
    Lee una polar de XFOIL.
    Devuelve array (N, 5): [alfa, CL, CD, CDp, CM]
    ordenado por alfa y sin ángulos repetidos.
    """
    texto = ruta.read_text(encoding="utf-8", errors="replace")
    lineas = texto.splitlines()

    # Buscar la línea de guiones que separa la cabecera de los datos
    idx_ini = None
    for i, ln in enumerate(lineas):
        if "------" in ln:
            idx_ini = i
            break
    if idx_ini is None:
        return np.zeros((0, 5))

    datos = []
    for ln in lineas[idx_ini + 1:]:
        valores = ln.split()
        if len(valores) >= 5:
            try:
                fila = [float(v) for v in valores[:5]]
                datos.append(fila)
            except ValueError:
                continue

    if not datos:
        return np.zeros((0, 5))

    arr = np.array(datos)
    # Ordenar por alfa y eliminar duplicados (alfa=0 aparece dos veces)
    _, idx_uniq = np.unique(arr[:, 0], return_index=True)
    return arr[idx_uniq]


def borrar_si_existe(ruta: Path) -> None:
    if ruta.is_file():
        ruta.unlink()


# =========================================================================
# COMPROBACIONES
# =========================================================================
if not XFOIL_EXE.is_file():
    raise FileNotFoundError(f"No se encuentra xfoil.exe en:\n  {XFOIL_EXE}")

for nombre, coords, _ in PERFILES:
    f = CARPETA_ORIGEN / nombre / "N9" / coords
    if not f.is_file():
        raise FileNotFoundError(f"No se encuentran las coordenadas:\n  {f}")


# =========================================================================
# BUCLE DE CALCULO
# =========================================================================
# polares[p][k] = array (N,5) para perfil p y caso k
polares: list[list[np.ndarray | None]] = [
    [None] * len(CASOS) for _ in PERFILES
]

t_total = time.time()

for p, (nombre, coords, prefijo) in enumerate(PERFILES):
    for k, (etiqueta, ncrit) in enumerate(CASOS):

        dir_caso = CARPETA_SAL / nombre / f"N{ncrit}"
        dir_caso.mkdir(parents=True, exist_ok=True)

        # 1) Coordenadas
        shutil.copy(CARPETA_ORIGEN / nombre / "N9" / coords, dir_caso)

        # 2) Limpiar resultados anteriores
        archivo_pol = f"{prefijo}_Clean_Re{ETIQUETA_RE}_M0-2_n{ncrit}.pol"
        borrar_si_existe(dir_caso / archivo_pol)
        borrar_si_existe(dir_caso / "xfoil.def")

        # 3) Script de XFOIL
        archivo_xfl = dir_caso / "Xfoil_WTTpolar.xfl"
        escribir_xfl(archivo_xfl, coords, archivo_pol,
                     RE, MACH, ncrit, N_PANEL, ITER_MAX, ALFA_MAX, D_ALFA)

        # 4) Ejecutar XFOIL
        print(f"[{nombre} | N = {ncrit} | Re = {RE:.2e}] Ejecutando XFOIL... ",
              end="", flush=True)
        t0 = time.time()
        ok = ejecutar_xfoil(XFOIL_EXE, dir_caso,
                            archivo_xfl, dir_caso / "xfoil_log.txt",
                            TIMEOUT)
        elapsed = time.time() - t0
        if ok:
            print(f"terminado en {elapsed:.1f} s")
        else:
            print(f"TIMEOUT ({TIMEOUT} s): se usa lo calculado hasta ese momento")

        # 5) Leer la polar
        ruta_pol = dir_caso / archivo_pol
        if ruta_pol.is_file():
            polares[p][k] = leer_polar(ruta_pol)
            print(f"    {len(polares[p][k])} ángulos convergidos -> {archivo_pol}")
        else:
            print(f"  AVISO: No se generó la polar {ruta_pol}")

elapsed_total = time.time() - t_total
print(f"\nCálculo completo en {elapsed_total / 60:.1f} min")

# Guardar datos en numpy
np.save(
    CARPETA_SAL / f"polares_Re{ETIQUETA_RE}.npy",
    {"polares": polares, "perfiles": PERFILES, "casos": CASOS,
     "Re": RE, "Mach": MACH},
    allow_pickle=True,
)


# =========================================================================
# GRAFICAS
# =========================================================================
estilos = ["-", "--"]
colores = ["tab:blue", "tab:orange"]

for p, (nombre, _, _) in enumerate(PERFILES):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{nombre}  -  Re = {RE:.1e},  M = {MACH:.1f}")

    titulos = ["Sustentación", "Resistencia", "Eficiencia"]
    ylabels = ["$C_L$", "$C_D$", "$C_L / C_D$"]

    for ax, titulo, ylabel in zip(axes, titulos, ylabels):
        ax.set_xlabel(r"$\alpha$ [°]")
        ax.set_ylabel(ylabel)
        ax.set_title(titulo)
        ax.grid(True)

    for k, (etiqueta, ncrit) in enumerate(CASOS):
        P = polares[p][k]
        if P is None or len(P) == 0:
            continue
        lbl = f"{etiqueta} (N = {ncrit})"
        alfa = P[:, 0]
        cl   = P[:, 1]
        cd   = P[:, 2]

        axes[0].plot(alfa, cl,      estilos[k], color=colores[k],
                     linewidth=1.3, label=lbl)
        axes[1].plot(alfa, cd,      estilos[k], color=colores[k],
                     linewidth=1.3, label=lbl)
        axes[2].plot(alfa, cl / cd, estilos[k], color=colores[k],
                     linewidth=1.3, label=lbl)

    axes[0].legend(loc="best")
    fig.tight_layout()

    out_png = CARPETA_SAL / f"{nombre}_Re{ETIQUETA_RE}.png"
    fig.savefig(out_png, dpi=150)
    print(f"Figura guardada: {out_png}")

plt.show()
