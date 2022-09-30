import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.pyplot import yticks
# style.use('ggplot')



parser = argparse.ArgumentParser(description="Fit MCCE with various datasets.")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    required=True,
    help="Path where results are saved",
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    default="adult",
    help="Datasets for experiment. Options are adult, give_me_some_credit, and compas.",
)
parser.add_argument(
    "-k",
    "--k",
    type=int,
    default=1000,
    help="Number of samples for each test observation.",
)
args = parser.parse_args()

data_name = args.dataset
k = args.k
path = args.path
seed = 1

mcce = pd.read_csv(os.path.join(path, f"{data_name}_mcce_results_k_{k}_n_several_cpu.csv"), index_col=0).groupby(['n_test']).mean()
mcce = mcce.reset_index()

cchvae = pd.read_csv(os.path.join(path, f"{data_name}_carla_results_n_several_cuda.csv"), index_col=0).groupby(['n_test']).mean()
cchvae = cchvae.reset_index()

# IF you want log results, uncomment this

# mcce['time (seconds)'] = np.log(mcce['time (seconds)'])
# mcce['fit (seconds)'] = np.log(mcce['fit (seconds)'])
# mcce['n_test'] = np.log(mcce['n_test'])

# cchvae['time (seconds)'] = np.log(cchvae['time (seconds)'])
# cchvae['fitting (seconds)'] = np.log(cchvae['fitting (seconds)'])
# cchvae['n_test'] = np.log(cchvae['n_test'])

# plotting

par = {'axes.titlesize': 30}
plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams.update(par)

fig, ax = plt.subplots()

# plot lines
ax.plot(cchvae['n_test'], cchvae['time (seconds)'], label = "CCHVAE total", linestyle="-", marker='o', color='red', linewidth=3.0)
ax.plot(cchvae['n_test'], cchvae['fitting (seconds)'], label = "CCHVAE fit", linestyle="-.", marker='o', color='red', linewidth=3.0)
# ax.plot(cchvae.index, cchvae['sampling (seconds)'], label = "CCHVAE sampling", linestyle=":",  marker='o',color='red')

ax.plot(mcce['n_test'], mcce['time (seconds)'], label = "MCCE total", linestyle="-", marker='o', color='blue', linewidth=3.0)
ax.plot(mcce['n_test'], mcce['fit (seconds)'], label = "MCCE fit", linestyle="-.", marker='o', color='blue', linewidth=3.0)
# ax.plot(mcce.index, mcce['generate (seconds)'], label = "MCCE sampling", linestyle=":", marker='o', color='blue')
# ax.plot(mcce.index, mcce['postprocess (seconds)'], label = "MCCE postprocess", linestyle="--", marker='o', color='blue')

ax.legend(title_fontsize=100)
ax.legend(prop={'size': 20})
ax.set_ylabel("seconds", size=25)
ax.set_xlabel("number of test observations", size=25)
ax.set_facecolor('white')

# If you want to fix the log axes, uncomment below

# fig.canvas.draw()

# labels = ax.get_yticklabels()
# labels = [round(np.exp(float(label.get_text()))) for label in labels]
# ax.set_yticks(ax.get_yticks().tolist())
# ax.set_yticklabels(labels)

# labels = ax.get_xticklabels()
# labels = [round(np.exp(float(label.get_text()))) for label in labels]
# labels_new = []
# for ix, label in enumerate(labels): # remove some labels because they take too much room
#     if ix % 2 == 0:
#         labels_new.append(str(int(np.floor(label / 10) * 10))) # round to nearest 10
#     else:
#         labels_new.append("")
# ax.set_xticks(ax.get_xticks().tolist())
# ax.set_xticklabels(labels_new)

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

plt.show()

# plt.savefig(f'{data_name}_variation_in_ntest_logscale_k_{k}.png', bbox_inches='tight')
plt.savefig(f'{data_name}_variation_in_ntest_k_{k}.png', bbox_inches='tight')

