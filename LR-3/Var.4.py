"""ЛР-3, варіант 4: нормовані ДС пірамідального рупора.
Serhii Zeibel gr/539
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks

WAVELENGTH = 0.028  # м
APERTURE_A = 0.13  # м, розмір у площині H
APERTURE_B = 0.13  # м, розмір у площині E
LEVEL = 0.707
OUT = Path(__file__).resolve().parent


def fh(theta):
    """Кут у радіанах. Стійка форма, зокрема при q=1 (границя π/4)."""
    theta = np.asarray(theta)
    q = 2 * APERTURE_A / WAVELENGTH * np.abs(np.sin(theta))
    aperture = np.pi / 2 * np.sinc((q - 1) / 2) / (q + 1)
    return (1 + np.cos(theta)) / 2 * np.abs(aperture)


def fe(theta):
    """np.sinc(x) = sin(πx)/(πx), np.sinc(0) = 1."""
    theta = np.asarray(theta)
    return (1 + np.cos(theta)) / 2 * np.abs(
        np.sinc(APERTURE_B / WAVELENGTH * np.sin(theta)))


def characteristics(fun, zeros):
    half = np.degrees(brentq(lambda t: float(fun(t)) - LEVEL,
                            0, np.radians(zeros[0]), xtol=1e-14))
    grid = np.linspace(0, np.pi / 2, 90001)
    indices, _ = find_peaks(fun(grid))
    peaks = []
    for i in indices:
        optimum = minimize_scalar(lambda t: -float(fun(t)),
                                  bounds=(grid[i-1], grid[i+1]),
                                  method='bounded', options={'xatol': 1e-14})
        value = float(fun(optimum.x))
        peaks.append({'angle': float(np.degrees(optimum.x)),
                      'amplitude': value, 'db': float(20*np.log10(value))})
    return {'half_angle': float(half), 'width': float(2*half),
            'zeros': zeros, 'peaks': peaks, 'edge': float(fun(np.pi/2))}


def calculate():
    zeros_h = [float(np.degrees(np.arcsin((2*n+1)*WAVELENGTH/(2*APERTURE_A))))
               for n in range(1, 100) if (2*n+1)*WAVELENGTH <= 2*APERTURE_A]
    zeros_e = [float(np.degrees(np.arcsin(n*WAVELENGTH/APERTURE_B)))
               for n in range(1, 100) if n*WAVELENGTH <= APERTURE_B]
    return {'wavelength': WAVELENGTH, 'a': APERTURE_A, 'b': APERTURE_B,
            'level': LEVEL, 'H': characteristics(fh, zeros_h),
            'E': characteristics(fe, zeros_e)}


def plot(results):
    plt.rcParams.update({'font.size': 10})
    angles = np.linspace(-90, 90, 18001)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.8), constrained_layout=True)
    for ax, plane, fun, color in zip(axes, ('H', 'E'), (fh, fe), ('royalblue', 'crimson')):
        r = results[plane]
        ax.plot(angles, fun(np.radians(angles)), color=color, lw=1.2, label=f'$F_{plane}(θ)$')
        ax.axhline(LEVEL, color='gray', ls='--', lw=.8, label='Рівень 0,707')
        half = r['half_angle']
        ax.scatter([-half, half], [LEVEL]*2, color=color, s=25, zorder=4)
        ax.vlines([-half, half], 0, LEVEL, color=color, ls=':', lw=.9)
        ax.annotate('', xy=(half, LEVEL), xytext=(-half, LEVEL),
                    arrowprops={'arrowstyle':'<->', 'color':color, 'lw':1.3})
        ax.text(0, .77, f'Δθ{plane} = {r["width"]:.2f}°', ha='center', color=color,
                bbox={'facecolor':'white', 'edgecolor':'none', 'pad':1})
        for sign in (-1, 1):
            ax.scatter(sign*np.array(r['zeros']), np.zeros(len(r['zeros'])),
                       marker='x', color='black', s=25, zorder=4)
            ax.scatter([sign*p['angle'] for p in r['peaks']],
                       [p['amplitude'] for p in r['peaks']], color='darkgreen', s=20, zorder=4)
        for p in r['peaks']:
            ax.annotate(f'{p["angle"]:.2f}°; {p["amplitude"]:.3f}',
                        xy=(p['angle'],p['amplitude']), xytext=((-28 if p['angle'] > 80 else 0),14),
                        textcoords='offset points', ha='center', fontsize=8)
        ax.set(xlim=(-90,90), ylim=(-.02,1.08), xlabel='θ, градуси', ylabel=f'$F_{plane}(θ)$',
               title=f'Нормована ДС у площині {plane}')
        ax.set_xticks(np.arange(-90,91,15)); ax.grid(ls='--', alpha=.35)
        ax.legend(loc='upper right', fontsize=9)
    fig.savefig(OUT/'LR3_rectangular.png', dpi=300)
    fig2 = plt.figure(figsize=(8,5.2))
    ax = fig2.add_subplot(111, projection='polar')
    for plane, fun, color in zip(('H','E'), (fh,fe), ('royalblue','crimson')):
        ax.plot(np.radians(angles), fun(np.radians(angles)), color=color, lw=1.2,
                label=f'$F_{plane}(θ)$')
        half=np.radians(results[plane]['half_angle'])
        ax.scatter([-half,half],[LEVEL,LEVEL],color=color,s=22,zorder=4)
        ax.plot(np.linspace(-half,half,100),np.full(100,LEVEL),color=color,lw=2)
        ax.plot([-half,-half],[0,LEVEL],color=color,ls=':',lw=.8)
        ax.plot([half,half],[0,LEVEL],color=color,ls=':',lw=.8)
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_thetamin(-90);ax.set_thetamax(90);ax.set_thetagrids(range(-90,91,15))
    ax.set_ylim(0,1);ax.set_yticks([.2,.4,.6,.8,1]);ax.set_yticklabels([])
    for radius in (.2,.4,.6,.8,1):
        ax.text(np.radians(55),radius-.025,f'{radius:.1f}',fontsize=8,ha='left',va='bottom')
    ax.grid(ls='--',alpha=.4)
    ax.set_title('Нормовані ДС пірамідального рупора',y=.80,pad=8)
    ax.legend(loc='upper right',bbox_to_anchor=(.98,.31),fontsize=9)
    fig2.text(.5,.035,f'Дуги на рівні 0,707: ΔθH = {results["H"]["width"]:.2f}°; '
              f'ΔθE = {results["E"]["width"]:.2f}°',ha='center',fontsize=10)
    fig2.savefig(OUT/'LR3_polar.png',dpi=300,bbox_inches='tight')
    plt.show()


def main():
    r=calculate()
    #(OUT/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'LR3_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f,delimiter=';')
        writer.writerow(['Площина','№','θ нуля, °','θ максимуму, °','F максимуму','Рівень, дБ'])
        for plane in ('H','E'):
            v=r[plane]
            print(f'Δθ{plane} = {v["width"]:.4f}°; половинний кут = {v["half_angle"]:.4f}°')
            for i in range(max(len(v['zeros']),len(v['peaks']))):
                z=f'{v["zeros"][i]:.4f}' if i<len(v['zeros']) else '—'
                p=v['peaks'][i] if i<len(v['peaks']) else None
                row=[plane,i+1,z,*([f'{p["angle"]:.4f}',f'{p["amplitude"]:.6f}',f'{p["db"]:.2f}'] if p else ['—']*3)]
                writer.writerow(row);print(' | '.join(map(str,row)))
    plot(r)


if __name__=='__main__':
    main()
