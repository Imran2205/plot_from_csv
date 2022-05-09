import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


def my_ticks_y(x, pos):
    if x == 0:
        return "$0$"
    exponent = -6
    co_eff = x/10**exponent
    return r"${:2.0f} \times 10^{{ {:2d} }}$".format(co_eff, exponent)


def my_ticks_x(y, pos):
    if y == 0:
        return "$0$"
    exponent = -9
    co_eff = y/10**exponent
    return r"${:.2f} \times 10^{{ {:2d} }}$".format(co_eff, exponent)


input_file = 'data_test.csv'

df = pd.read_csv(input_file)
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.astype(np.float64)

X = []
Y = []

labels = []

for i in range(0, df.shape[1], 2):
    labels.append(df.columns[i+1])
    X.append([x for x in df[df.columns[i]]])
    Y.append([y for y in df[df.columns[i+1]]])

fig, ax = plt.subplots()

for j in range(len(Y)):
    ax.plot(X[j], Y[j], label=labels[j])

start_x, end_x = ax.get_xlim()
# ax.xaxis.set_ticks(np.arange(start_x + abs((end_x - start_x) / 10), end_x, abs((end_x - start_x) / 10)))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(my_ticks_x))

start_y, end_y = ax.get_ylim()
# ax.yaxis.set_ticks(np.arange(start_y + abs((end_y - start_y) / 10), end_y, abs((end_y - start_y) / 10)))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(my_ticks_y))

plt.xlabel("Time(s)")
plt.ylabel("Current(A)")
plt.title("Current vs Time")
plt.xticks(rotation=90)
# plt.ticklabel_format(axis='y', style='sci', scilimits=(-4, 0), useMathText=True)

plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.legend(ncol=2, handleheight=0.005, labelspacing=0.002, bbox_to_anchor=(1.0, 1.0), prop={'size': 7})


plt.show()
