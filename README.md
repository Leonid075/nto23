# nto23
## Решение командного этапа НТО 2023
В решении используется свёрточный автоэнкодер, его код взят из https://github.com/wengong-jin/hgraph2graph и статьи https://arxiv.org/pdf/2002.03230.pdf,
и модель logP из индивидуального этапа


### Автоэнкодер
Модель автоэнкодера `/polymers/poly_hgraph/` обучалась мной на неразмеченных данных `/data/all.txt`

Инструкция по препроцессингу данных, обучению проверки модели `/polymers/README.md`

Предобученные мной веса `/data/model30.pt`


### Генерация новых молекул
Код генератора `/polymers/generate.py`

При первой итерации `serch_step` список молекул переводится в тензоры и проходит через энкодер для получения векторного представления

При каждой итерации `serch_step` из списка векторных предсталений молекул берутся 2 рандомных вектора и передаются в функцию `sample_new`. В ней вычисляются `n-2` веторных представлений, лежащих между исходными (`/polymers/poly_hgraph/hgnn.py:89`):
```
for t in np.linspace(0., 1., n)[1:-1]:
  z.append(frm * (1-t) + to * t)
```
после чего декодер восстанавливает их до мелекулярных графов и конвертирует в SMILES. Из них отбираются "хорошие" молекулы. На каждой итерации `serch_step` проходит несколько (5-6) итераций генерации.

### Железо и софт
Весь код запускался на ОС ubuntu 22, python 3.11, CUDA 11.2, GPU Nvidia RTX3070

 * PyTorch >= 1.0.0
 * networkx
 * RDKit >= 2019.03
 * numpy
 * Python >= 3.6
 * LightGBM
