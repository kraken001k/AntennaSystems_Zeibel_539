"""Розрахунок діаграм направленості діелектричних стрижневих антен.

Для одного та двох стрижнів обчислюються нормовані амплітудні діаграми
в площинах H та E. Визначаються напрямки максимумів, нулі діаграм
і ширина головної пелюстки на амплітудному рівні 0,707.
Кожна діаграма нормується окремо за максимумом у діапазоні −90°...90°.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from scipy.signal import find_peaks
from scipy.optimize import brentq, minimize_scalar

# Геометричні розміри задано в метрах. Кути всередині моделі - у радіанах.
WAVELENGTH = 0.035 # Довжина хвилі у вільному просторі: 3,5 см.
LENGTH = 0.237 # Довжина діелектричного стрижня: 23,7 см.
DIAMETER = 0.023 # Діаметр стрижня: 2,3 см.
SPACING = 0.12 # Відстань між осями двох стрижнів: 12 см.
XI = 1.30  # Коефіцієнт сповільнення хвилі; прийняте наближене значення.
LEVEL = 0.707 # Амплітудний рівень для визначення ширини головної пелюстки.
OUT = Path(__file__).resolve().parent
SHOW_PLOTS = True  # False лише зберегти файли, не відкривати вікна


def lambda1(x):
    """Обчислити Λ_1(x) = 2J_1(x)/x, де J_2 - функція Бесселя першого роду.
    У точці x = 0 використовується граничне значення Λ_1(0) = 1.
    Маска where запобігає діленню на нуль і нестійкості поблизу нуля.
    """
    x = np.asarray(x, dtype=float)
    result = np.ones_like(x)
    np.divide(2*j1(x), x, out=result, where=np.abs(x)>1e-10)
    return result


def raw_pattern(theta, rods=1, plane='H'):
    """Обчислити ненормований модуль амплітудної діаграми.
    theta - кут відносно осі антени в радіанах; rods - кількість стрижнів.
    plane задає площину розрахунку H або E.
    """
    theta = np.asarray(theta, dtype=float)
    # Аргументи множників, що враховують довжину та поперечний розмір стрижня.
    t = LENGTH/WAVELENGTH*(XI-np.cos(theta))
    z = np.pi*DIAMETER/WAVELENGTH*np.sin(theta)
    # Постійний множник скорочується після нормування за максимумом.
    value = np.sinc(t)*lambda1(z)
    # У площині E діаграма містить додатковий множник cos(theta).
    if plane == 'E':
        value = value*np.cos(theta)
    # Для двох синфазних стрижнів додається інтерференційний множник.
    if rods == 2:
        value = value*np.cos(np.pi*SPACING/WAVELENGTH*np.sin(theta))
    return np.abs(value)


def zero_angles(rods, plane):
    """Знайти аналітичні кути нулів діаграми в діапазоні 0°...90°.
    Нулі визначаються окремо для кожного множника; повтори видаляються.
    Для від’ємних кутів нулі симетричні відносно осі антени.
    """
    zeros = []
    # Нулі sinc(t) відповідають додатним цілим t = m
    # З рівняння t = (LENGTH/WAVELENGTH) * (XI − cos(theta)) знаходимо кут
    # Для заданого XI > 1 аргумент t додатний у всьому діапазоні
    for m in range(1, 100):
        cosine = XI-m*WAVELENGTH/LENGTH
        if 0 <= cosine < 1:
            zeros.append(float(np.degrees(np.arccos(cosine))))
    # Нулі множника двох стрижнів: sin(theta) = (2m + 1) * λ/(2 * SPACING)
    # Враховуємо лише розв’язки з допустимим значенням синуса.
    if rods == 2:
        for m in range(100):
            sine = (2*m+1)*WAVELENGTH/(2*SPACING)
            if 0 < sine <= 1:
                zeros.append(float(np.degrees(np.arcsin(sine))))
    # Це менше першого додатного нуля J_1, тому Λ_1 додаткових нулів не дає.
    # У площині E множник cos(theta) дорівнює нулю при 90 град .
    if plane == 'E':
        zeros.append(90.0)
    return sorted(set(round(z, 10) for z in zeros))


def analyse(rods, plane):
    """Знайти максимум, ширину головної пелюстки та локальні максимуми
    Пошук виконується на додатній половині діаграми завдяки її симетрії
    Кути в результаті та ширина пелюстки подаються в градусах
    """
    # Щільна сітка дає початкові інтервали для уточнення максимумів і перетинів.
    grid = np.linspace(0, 90, 90001)
    fun = lambda degree: float(raw_pattern(np.radians(degree), rods, plane))
    values = raw_pattern(np.radians(grid), rods, plane)
    indices, _ = find_peaks(values)
    candidates = []
    # Положення кожного піка уточнюємо мінімізацією від’ємної амплітуди.
    for i in indices:
        res = minimize_scalar(lambda t: -fun(t), bounds=(grid[i-1], grid[i+1]),
                              method='bounded', options={'xatol':1e-12})
        candidates.append((float(res.x), fun(res.x)))
    # find_peaks не перевіряє крайні точки, тому осьовий максимум додаємо окремо.
    if values[0] > values[1]:
        candidates.insert(0,(0.0,fun(0)))
    # Межа 90град не вважається внутрішнім максимумом
    # Для нормування порівнюємо всі знайдені піки та обидві межі діапазону.
    peak = max(candidates+[(0.,fun(0)),(90.,fun(90))],key=lambda p:p[1])
    scale = peak[1]
    norm = lambda degree: fun(degree)/scale
    crossings = []
    y = values/scale-LEVEL
    for i in np.flatnonzero(y[:-1]*y[1:] < 0):
        crossings.append(brentq(lambda t:norm(t)-LEVEL,grid[i],grid[i+1],xtol=1e-12))
    center=peak[0]
    # Для осьової пелюстки ліва межа є дзеркальним відображенням правої.
    # Для відхиленої пелюстки беремо найближчі перетини з обох боків піка.
    if center == 0:
        left=-crossings[0];right=crossings[0]
    else:
        left=max(c for c in crossings if c<center)
        right=min(c for c in crossings if c>center)
    # Переведення амплітуди в децибели
    # Позначка main стосується головного піка на додатній половині діаграми.
    peaks=[{'angle':t,'amplitude':v/scale,'db':20*np.log10(v/scale),
            'main':abs(t-center)<1e-4} for t,v in candidates]
    return {'rods':rods,'plane':plane,'scale':scale,'main_angle':center,
            'left':left,'right':right,'width':right-left,
            'axis':norm(0),'edge':norm(90),
            'zeros':zero_angles(rods,plane),'peaks':peaks}


def normalized(theta, result):
    """Нормувати діаграму за максимумом знайденим у функції analyse"""
    return raw_pattern(theta,result['rods'],result['plane'])/result['scale']


def save_tables(results):
    """Зберегти ширини пелюсток, нулі та максимуми"""
    with (OUT/'widths.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f,delimiter=';')
        writer.writerow(['Стрижнів','Площина','θ головного максимуму, °','θ лівої межі, °',
                         'θ правої межі, °','Ширина на рівні 0.707, °','F(0)'])
        for r in results:
            writer.writerow([r['rods'],r['plane'],*[f'{r[k]:.6f}' for k in
                             ('main_angle','left','right','width','axis')]])
    with (OUT/'extrema.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f,delimiter=';')
        writer.writerow(['Стрижнів','Площина','№','θ нуля, °','θ максимуму, °','F','дБ','Тип'])
        for r in results:
            for i in range(max(len(r['zeros']),len(r['peaks']))):
                z=f'{r["zeros"][i]:.4f}' if i<len(r['zeros']) else '—'
                p=r['peaks'][i] if i<len(r['peaks']) else None
                writer.writerow([r['rods'],r['plane'],i+1,z,
                                 *([f'{p["angle"]:.4f}',f'{p["amplitude"]:.6f}',f'{p["db"]:.2f}',
                                    'Головна' if p['main'] else 'Бічна'] if p else ['—']*4)])


def make_plots(results):
    """Побудувати декартові й полярні діаграми для одного та двох стрижнів
    Хрестиками позначено нулі, зеленими точками максимуми
    Межі головних пелюсток показано на амплітудному рівні LEVEL
    """
    degrees=np.linspace(-90,90,18001)
    theta=np.radians(degrees)
    for rods in (1,2):
        subset=[r for r in results if r['rods']==rods]
        fig,axes=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
        for ax,r,color in zip(axes,subset,('royalblue','crimson')):
            ax.plot(degrees,normalized(theta,r),color=color,lw=1.2,label=f'$F_{{{rods}{r["plane"]}}}(θ)$')
            ax.axhline(LEVEL,color='gray',ls='--',lw=.8,label='Рівень 0,707')
            for sign in (-1,1):
                ax.scatter(sign*np.array(r['zeros']),np.zeros(len(r['zeros'])),marker='x',color='black',s=25,zorder=5)
                ax.scatter([sign*p['angle'] for p in r['peaks']],
                           [p['amplitude'] for p in r['peaks']],color='darkgreen',s=20,zorder=4)
                lo,hi=sorted([sign*r['left'],sign*r['right']])
                ax.scatter([lo,hi],[LEVEL]*2,color=color,s=23,zorder=6)
                ax.vlines([lo,hi],0,LEVEL,color=color,ls=':',lw=.8)
                ax.annotate('',xy=(hi,LEVEL),xytext=(lo,LEVEL),
                            arrowprops={'arrowstyle':'<->','color':color,'lw':1.3})
            mid=(r['left']+r['right'])/2
            ax.annotate(f'Δθ = {r["width"]:.2f}°',xy=(mid,LEVEL),xytext=(mid+12,.86),
                        ha='center',color=color,fontsize=10,
                        arrowprops={'arrowstyle':'-','color':color,'lw':.6})
            ax.set(xlim=(-90,90),ylim=(-.025,1.12),xlabel='θ, градуси',ylabel='Нормована амплітуда',
                   title=f'{rods} стрижень' if rods==1 else '2 стрижні')
            ax.set_title(f'{rods} стрижень, площина {r["plane"]}' if rods==1 else f'2 стрижні, площина {r["plane"]}')
            ax.set_xticks(np.arange(-90,91,15));ax.grid(ls='--',alpha=.35)
            ax.legend(loc='upper left',fontsize=9)
        fig.suptitle(f'ДСА, варіант 4: λ = 3,5 см; ξ ≈ {XI:.2f}'+('; h = 12 см' if rods==2 else ''),fontsize=12)
        fig.savefig(OUT/f'{rods}rod_rectangular.png',dpi=300)
        figp,axes=plt.subplots(1,2,figsize=(11,5),subplot_kw={'projection':'polar'})
        for ax,r,color in zip(axes,subset,('royalblue','crimson')):
            ax.plot(theta,normalized(theta,r),color=color,lw=1.2)
            for sign in (-1,1):
                lo,hi=sorted([sign*r['left'],sign*r['right']])
                ax.plot(np.radians(np.linspace(lo,hi,100)),np.full(100,LEVEL),color=color,lw=2)
                ax.scatter(np.radians([lo,hi]),[LEVEL]*2,s=18,color=color,zorder=5)
                ax.scatter(np.radians(sign*np.array(r['zeros'])),np.zeros(len(r['zeros'])),marker='x',s=20,color='black')
                ax.scatter(np.radians([sign*p['angle'] for p in r['peaks']]),
                           [p['amplitude'] for p in r['peaks']],s=15,color='darkgreen',zorder=4)
            ax.set_theta_zero_location('N');ax.set_theta_direction(-1)
            ax.set_thetamin(-90);ax.set_thetamax(90);ax.set_thetagrids(range(-90,91,30))
            ax.set_ylim(0,1.05);ax.set_yticks([.25,.5,.75,1]);ax.set_yticklabels([])
            for radius in (.25,.5,.75,1):ax.text(np.radians(60),radius,f'{radius:g}',fontsize=7)
            ax.set_title(f'{rods} стрижень, {r["plane"]}' if rods==1 else f'2 стрижні, {r["plane"]}',y=.8)
            ax.grid(ls='--',alpha=.4)
        figp.text(.5,.08,'Кольорові дуги — ширина головних пелюсток на рівні 0,707; зелені точки — максимуми.',ha='center',fontsize=9)
        figp.savefig(OUT/f'{rods}rod_polar.png',dpi=300,bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close('all')


def main():
    results=[analyse(n,p) for n in (1,2) for p in ('H','E')]
    print(f'λ = {WAVELENGTH*100:.1f} см; l = {LENGTH*100:.1f} см; h = {SPACING*100:.1f} см; ξ ≈ {XI:.2f}')
    for r in results:
        print(f'{r["rods"]} стрижні, {r["plane"]}: θmax = ±{r["main_angle"]:.4f}°; '
              f'межі додатної головної пелюстки [{r["left"]:.4f}°, {r["right"]:.4f}°]; '
              f'ширина = {r["width"]:.4f}°; F(0) = {r["axis"]:.6f}')
        for p in r['peaks']:
            print(f'  {p["angle"]:.4f}° | F={p["amplitude"]:.6f} | {p["db"]:.2f} дБ | '
                  +('Головна' if p['main'] else 'Бічна'))
    save_tables(results)
    make_plots(results)


if __name__=='__main__':
    main()
