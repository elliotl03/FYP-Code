import matplotlib.pyplot as plt
import numpy as np

# 130 W data: [pressure, Tvib]
data_130 = np.array([
    [10, 6786],
    [15, 6624],
    [20, 6400],
    [25, 6329],
    [30, 6410],
    [35, 6268],
    [40, 6118],
    [45, 6153],
    [50, 6196],
], dtype=float)

# 180 W data: [pressure, Tvib]
data_180 = np.array([
    [10, 6578],
    [15, 6481],
    [20, 6355],
    [25, 6123],
    [30, 6278],
    [35, 6225],
    [40, 6269],
    [45, 6096],
    [50, 6036],
], dtype=float)

# 230 W data: [pressure, Tvib]
data_230 = np.array([
    [10, 6330],
    [15, 6217],
    [20, 6233],
    [25, 6020],
    [30, 6121],
    [35, 6251],
    [40, 5942],
    [45, 5953],
    [50, 6055],
], dtype=float)

# 300 W data: [pressure, Tvib]
data_300 = np.array([
    [10, 6383],
    [15, 6377],
    [20, 6355],
    [25, 6192],
    [30, 6158],
    [35, 6161],
    [40, 6223],
    [45, 6231],
    [50, 6134],
], dtype=float)

# 400 W data: [pressure, Tvib]
data_400 = np.array([
    [10, 6233],
    [15, 6130],
    [20, 6049],
    [25, 6179],
    [30, 6074],
], dtype=float)

Tvib_err = 37.0   # standard deviation from repeat measurements

plt.figure(figsize=(8.8, 5.8))

plt.errorbar(data_130[:, 0], data_130[:, 1], yerr=Tvib_err, fmt='^-',
             color='tab:green', markersize=7, linewidth=1.4, capsize=3, label='130 W')

plt.errorbar(data_180[:, 0], data_180[:, 1], yerr=Tvib_err, fmt='d-',
             color='tab:purple', markersize=7, linewidth=1.4, capsize=3, label='180 W')

plt.errorbar(data_230[:, 0], data_230[:, 1], yerr=Tvib_err, fmt='o-',
             color='tab:blue', markersize=7, linewidth=1.4, capsize=3, label='230 W')

plt.errorbar(data_300[:, 0], data_300[:, 1], yerr=Tvib_err, fmt='v-',
             color='tab:red', markersize=7, linewidth=1.4, capsize=3, label='300 W')

plt.errorbar(data_400[:, 0], data_400[:, 1], yerr=Tvib_err, fmt='s-',
             color='tab:orange', markersize=7, linewidth=1.4, capsize=3, label='400 W')

plt.xlabel("Pressure (mTorr)")
plt.ylabel(r"$T_{\mathrm{vib}}$ (K)")
plt.title(r"$T_{\mathrm{vib}}$ versus pressure")
plt.ylim(5500, 7000)
plt.xlim(8, 52)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

data_130 = np.array([
    [10, 6786],
    [15, 6624],
    [20, 6400],
    [25, 6329],
    [30, 6410],
    [35, 6268],
    [40, 6118],
    [45, 6153],
    [50, 6196],
], dtype=float)

data_180 = np.array([
    [10, 6578],
    [15, 6481],
    [20, 6355],
    [25, 6123],
    [30, 6278],
    [35, 6225],
    [40, 6269],
    [45, 6096],
    [50, 6036],
], dtype=float)

data_230 = np.array([
    [10, 6330],
    [15, 6217],
    [20, 6233],
    [25, 6020],
    [30, 6121],
    [35, 6251],
    [40, 5942],
    [45, 5953],
    [50, 6055],
], dtype=float)

data_300 = np.array([
    [10, 6383],
    [15, 6377],
    [20, 6355],
    [25, 6192],
    [30, 6158],
    [35, 6161],
    [40, 6223],
    [45, 6231],
    [50, 6134],
], dtype=float)

data_400 = np.array([
    [10, 6233],
    [15, 6130],
    [20, 6049],
    [25, 6179],
    [30, 6074],
], dtype=float)

Tvib_err = 37.0  # standard deviation from repeat measurements

def add_series(data, color, marker, label):
    p = data[:, 0]
    T = data[:, 1]

    fit = np.polyfit(p, T, 1)
    xfit = np.linspace(p.min(), p.max(), 200)
    yfit = fit[0] * xfit + fit[1]

    plt.errorbar(
        p, T, yerr=Tvib_err,
        fmt=marker, color=color, markersize=7,
        linestyle='None', capsize=3, #label=f"{label} data"
    )
    plt.plot(
        xfit, yfit, '-', color=color, linewidth=1.5,
        alpha=0.85, label=f"{label} trend"
    )

plt.figure(figsize=(8.8, 5.8))

add_series(data_130, 'tab:green', '^', '130 W')
add_series(data_180, 'tab:purple', 'd', '180 W')
add_series(data_230, 'tab:blue', 'o', '230 W')
add_series(data_300, 'tab:red', 'v', '300 W')
add_series(data_400, 'tab:orange', 's', '400 W')

plt.xlabel("Pressure (mTorr)")
plt.ylabel(r"$T_{\mathrm{vib}}$ (K)")
plt.title(r"$T_{\mathrm{vib}}$ versus pressure")
plt.ylim(5500, 7000)
plt.xlim(8, 52)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()