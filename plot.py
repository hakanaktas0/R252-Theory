import pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

with open('stats_add_edges.pkl','rb') as f:
    stats = pickle.load(f)

import matplotlib.pyplot as plt
import numpy as np
energies = []
for i  in range(10):
    plt.plot(np.array(stats[i*1000]['epoch']), np.array(stats[i*1000]['val_acc']))
    energies.append(stats[i*1000]['dirichlet_energy'])
plt.legend(energies)
plt.show()