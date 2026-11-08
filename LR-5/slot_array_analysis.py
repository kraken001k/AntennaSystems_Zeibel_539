from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar

# Початкові дані
WAVELENGTH = 0.035  # λ = 3,5 см.
GUIDE_A = 0.023  # a = 23 мм, широка стінка хвилеводу МЕК-100.
N = 14
SLOT_LENGTH = 0.016  # Повна довжина щілини 1,6 см.
SPACING = 0.020  # d = 2 см, крок між сусідніми щілинами.
NU = 1  # ν = 1: шахове розташування дає додатковий фазовий зсув π.
LEVEL = 0.707  # Заданий амплітудний рівень для ширини головної пелюстки.
OUT = Path(__file__).resolve().parent
SHOW_PLOTS = True  # False — зберегти рисунки без відкривання вікон.

# (1.1)–(1.4). ξ у цій роботі — фазовий зсув у радіанах.
CRITICAL_WAVELENGTH = 2 * GUIDE_A
GUIDE_WAVELENGTH = WAVELENGTH / np.sqrt(1 - (WAVELENGTH / CRITICAL_WAVELENGTH)**2)
K = 2 * np.pi / WAVELENGTH
XI = 2 * np.pi * SPACING / GUIDE_WAVELENGTH - NU * np.pi


def phase(theta):
    """(1.5). Кут theta всередині розрахунку задано в радіанах."""
    return K * SPACING * np.sin(theta) - XI


def slot_h(theta):
    """(2.1). F1H = cos[(π/2)sinθ] / cosθ — напівхвильова модель щілини.
    Довжина SLOT_LENGTH близька до λ/2; у цій наближеній формулі
    вона не є окремим аргументом. При θ=±90° границя функції дорівнює 0.
    """
    theta = np.asarray(theta, dtype=float)
    value = np.zeros_like(theta)
    np.divide(np.cos((np.pi / 2) * np.sin(theta)), np.cos(theta),
              out=value, where=np.abs(np.cos(theta)) > 1e-10)
    return np.abs(value)


def array_factor(theta):
    """(2.2). Модуль Fc = sin(Nψ/2) / [N sin(ψ/2)].
    При ψ=2πq маємо невизначеність 0/0. Граничне значення МОДУЛЯ
    дорівнює 1, тому такі напрями не можна позначати нулями.
    """
    psi = phase(theta)
    value = np.ones_like(np.asarray(psi, dtype=float))
    denominator = N * np.sin(psi / 2)
    np.divide(np.sin(N * psi / 2), denominator, out=value,
              where=np.abs(denominator) > 1e-12)
    return np.abs(value)


def raw_h(theta):
    """(2.3). Повна амплітудна функція QH до нормування."""
    return slot_h(theta) * array_factor(theta)


def find_zeros():
    """(3.1), (3.2). Нулі решітки у передній напівплощині.
    Виключаємо m, кратні N: у них чисельник і знаменник зануляються разом.
    Крайові нулі ±90° додає множник одиночної щілини.
    """
    zeros = []
    for m in range(-2 * N, 2 * N + 1):
        sine = (m * WAVELENGTH / (SPACING * N)
                + WAVELENGTH / GUIDE_WAVELENGTH
                - NU * WAVELENGTH / (2 * SPACING))
        if abs(sine) <= 1 and m % N != 0:
            zeros.append(float(np.degrees(np.arcsin(sine))))
    return sorted([-90.0, *zeros, 90.0])


def preliminary_estimates():
    """(3.3), (3.4). Початкові оцінки двох сусідніх бічних пелюсток.
    n=-2 та n=1 відповідають першій оцінці зліва і справа від головної.
    Це точки максимуму чисельника, а не точні максимуми повної ДС.
    """
    estimates = []
    for n in (-2, 1):
        sine = ((2 * n + 1) * WAVELENGTH / (2 * SPACING * N)
                + WAVELENGTH / GUIDE_WAVELENGTH
                - NU * WAVELENGTH / (2 * SPACING))
        angle = float(np.degrees(np.arcsin(sine)))
        amplitude = 1 / (N * abs(np.sin((2 * n + 1) * np.pi / (2 * N))))
        estimates.append({'n': n, 'angle': angle, 'array_amplitude': amplitude})
    return estimates


def analyse():
    """Уточнюємо максимуми повної ДС між її сусідніми нулями.
    Кути в таблицях і результатах цієї функції подаються у градусах.
    """
    zeros = find_zeros()
    peaks = []
    for left, right in zip(zeros[:-1], zeros[1:]):
        optimum = minimize_scalar(
            lambda degree: -float(raw_h(np.radians(degree))),
            bounds=(left, right), method='bounded', options={'xatol': 1e-12})
        peaks.append({'angle': float(optimum.x), 'raw': float(-optimum.fun),
                      'zero_left': left, 'zero_right': right})

    main = max(peaks, key=lambda p: p['raw'])
    scale = main['raw']  # (2.4). M — найбільше значення QH.
    for p in peaks:
        p['amplitude'] = p['raw'] / scale  # (2.5). Нормована повна ДС.
        p['db'] = 20 * np.log10(p['amplitude'])  # (4.3).
        p['main'] = p is main

    # (4.1), (4.2). Межі рівня 0,707 шукаємо з двох боків головного максимуму.
    equation = lambda degree: float(raw_h(np.radians(degree))) / scale - LEVEL
    left = brentq(equation, main['zero_left'], main['angle'], xtol=1e-12)
    right = brentq(equation, main['angle'], main['zero_right'], xtol=1e-12)
    strongest_side = max((p for p in peaks if not p['main']), key=lambda p: p['amplitude'])
    return {'zeros': zeros, 'peaks': peaks, 'main': main, 'scale': scale,
            'left': left, 'right': right, 'width': right - left,
            'strongest_side': strongest_side, 'estimates': preliminary_estimates()}


def normalized_h(theta, result):
    """(2.5). FH = QH/M; нормуємо повний добуток, а не лише решітку."""
    return raw_h(theta) / result['scale']


def save_tables(r):
    """Підсумок та таблиця нулів і максимумів для звіту."""
    theta_c = np.degrees(np.arcsin(XI / (K * SPACING)))  # (1.6).
    with (OUT / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Параметр', 'Значення', 'Одиниця'])
        rows = [('λхв', GUIDE_WAVELENGTH * 100, 'см'), ('k', K, 'рад/м'),
                ('ξ', XI, 'рад'), ('θc решітки', theta_c, '°'),
                ('M', r['scale'], ''), ('θmax повної ДС', r['main']['angle'], '°'),
                ('θL', r['left'], '°'), ('θR', r['right'], '°'),
                ('Δθ на рівні 0.707', r['width'], '°'),
                ('θ найбільшої бічної пелюстки', r['strongest_side']['angle'], '°'),
                ('F найбільшої бічної пелюстки', r['strongest_side']['amplitude'], ''),
                ('L найбільшої бічної пелюстки', r['strongest_side']['db'], 'дБ')]
        for name, value, unit in rows:
            writer.writerow([name, f'{value:.9f}', unit])
    with (OUT / 'extrema.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['№', 'θ нуля, °', 'θ максимуму, °', 'FH', 'L, дБ', 'Тип максимуму'])
        # Нулі й максимуми — незалежні послідовності в окремих стовпцях.
        for i in range(max(len(r['zeros']), len(r['peaks']))):
            zero = f'{r["zeros"][i]:.6f}' if i < len(r['zeros']) else '—'
            p = r['peaks'][i] if i < len(r['peaks']) else None
            values = ([f'{p["angle"]:.6f}', f'{p["amplitude"]:.6f}', f'{p["db"]:.2f}',
                       'Головний' if p['main'] else 'Бічний'] if p else ['—'] * 4)
            writer.writerow([i + 1, zero, *values])


def make_plots(r):
    # ДС несиметрична: обчислюємо весь інтервал, нічого не віддзеркалюємо.
    degrees = np.linspace(-90, 90, 36001)
    theta = np.radians(degrees)
    fh = normalized_h(theta, r)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.plot(degrees, fh, color='royalblue', lw=1.3, label='$F_H(θ)$')
    ax.plot(degrees, array_factor(theta), color='gray', ls=':', lw=.9, label='Множник решітки $|F_c|$')
    ax.plot(degrees, slot_h(theta), color='darkorange', ls='--', lw=.9, label='ДС щілини $F_{1H}$')
    ax.axhline(LEVEL, color='black', ls='--', lw=.7, label='Рівень 0,707')
    ax.scatter(r['zeros'], np.zeros(len(r['zeros'])), marker='x', color='black', s=25, zorder=5)
    ax.scatter([p['angle'] for p in r['peaks']], [p['amplitude'] for p in r['peaks']],
               color='darkgreen', s=22, zorder=5)
    ax.scatter([r['left'], r['right']], [LEVEL] * 2, color='royalblue', s=28, zorder=6)
    ax.vlines([r['left'], r['right']], 0, LEVEL, color='royalblue', ls=':', lw=.8)
    ax.annotate('', xy=(r['right'], LEVEL), xytext=(r['left'], LEVEL),
                arrowprops={'arrowstyle': '<->', 'color': 'royalblue', 'lw': 1.3})
    ax.annotate(f'ΔθH = {r["width"]:.2f}°', xy=((r['left'] + r['right']) / 2, LEVEL),
                xytext=(6, .79), color='royalblue',
                arrowprops={'arrowstyle': '-', 'color': 'royalblue', 'lw': .7})
    ax.annotate(f'θmax = {r["main"]["angle"]:.2f}°', xy=(r['main']['angle'], 1),
                xytext=(-57, 1.01), fontsize=9, arrowprops={'arrowstyle': '->', 'color': 'black'})
    ax.set(xlim=(-90, 90), ylim=(-.035, 1.12), xlabel='θ, градуси', ylabel='Нормована амплітуда',
           title='ДС у площині H: варіант 4; тип 1; N=14; λ=3,5 см; d=2 см; ν=1')
    ax.set_xticks(np.arange(-90, 91, 15)); ax.grid(ls='--', alpha=.3)
    ax.legend(loc='upper right', fontsize=8)
    fig.savefig(OUT / 'LR5_H_rectangular.png', dpi=300)

    figp = plt.figure(figsize=(9, 6))
    ax = figp.add_subplot(111, projection='polar')
    ax.plot(theta, fh, color='royalblue', lw=1.3, label='$F_H(θ)$')
    ax.scatter(np.radians([p['angle'] for p in r['peaks']]),
               [p['amplitude'] for p in r['peaks']], color='darkgreen', s=20, zorder=4)
    ax.scatter(np.radians(r['zeros']), np.zeros(len(r['zeros'])), marker='x', color='black', s=18)
    arc = np.radians(np.linspace(r['left'], r['right'], 100))
    ax.plot(arc, np.full_like(arc, LEVEL), color='royalblue', lw=2)
    ax.scatter(np.radians([r['left'], r['right']]), [LEVEL] * 2, color='royalblue', s=25, zorder=5)
    for degree in (r['left'], r['right']):
        ax.plot(np.radians([degree, degree]), [0, LEVEL], color='royalblue', ls=':', lw=.7)
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_thetamin(-90); ax.set_thetamax(90); ax.set_thetagrids(range(-90, 91, 15))
    ax.set_ylim(0, 1.05); ax.set_yticks([.25, .5, .75, 1]); ax.set_yticklabels([])
    for radius in (.25, .5, .75, 1):
        ax.text(np.radians(60), radius, f'{radius:g}', fontsize=8)
    ax.set_title('Полярна ДС у площині H', y=.80)
    ax.grid(ls='--', alpha=.35)
    figp.text(.5, .08, f'Дуга на рівні 0,707: ΔθH = {r["width"]:.2f}°; θmax = {r["main"]["angle"]:.2f}°.',
              ha='center', fontsize=10)
    figp.savefig(OUT / 'LR5_H_polar.png', dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close('all')


def main():
    r = analyse()
    theta_c = np.degrees(np.arcsin(XI / (K * SPACING)))
    print(f'λхв={GUIDE_WAVELENGTH * 100:.6f} см; k={K:.6f} рад/м; ξ={XI:.6f} рад; ν={NU}')
    print(f'Максимум решітки: θc={theta_c:.6f}°')
    print(f'Максимум повної ДС: θmax={r["main"]["angle"]:.6f}°; M={r["scale"]:.9f}')
    print(f'Межі 0.707: [{r["left"]:.6f}°, {r["right"]:.6f}°]; повна ширина={r["width"]:.6f}°')
    for estimate in r['estimates']:
        print(f'Початкова оцінка n={estimate["n"]}: θ={estimate["angle"]:.6f}°; '
              f'|Fc|={estimate["array_amplitude"]:.6f}')
    print('Нулі H: ' + '; '.join(f'{t:.4f}°' for t in r['zeros']))
    print('Максимуми H: θ, ° | FH | L, дБ | Тип')
    for p in r['peaks']:
        print(f'{p["angle"]:9.4f} | {p["amplitude"]:.6f} | {p["db"]:7.2f} | '
              + ('Головний' if p['main'] else 'Бічний'))
    save_tables(r)
    make_plots(r)


if __name__ == '__main__':
    main()
