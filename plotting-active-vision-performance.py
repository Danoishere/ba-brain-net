import matplotlib
import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib.ticker import MaxNLocator

"""
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


"""
plt.rcParams['figure.figsize'] = (7,3.6)

with open('eval-result.json') as f:
    data = json.load(f)

for key in data:
    rec = data[key]
    result = np.array(rec['data'])
    xvals = np.arange(len(result)) + 1

    mean = np.mean(result, axis=1)
    std = np.std(result, axis=1)

    label = rec['label']
    plt.plot(xvals,  mean, label=label)

ax = plt.gca()
plt.xlabel('Frame')
plt.ylabel('Success Rate of Object Enumeration Stream')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.tight_layout()
#plt.savefig('plot-successrate-active-vision-enum-stream.pgf')

plt.show()