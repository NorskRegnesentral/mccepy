import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
  
style.use('ggplot')



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

cchvae = pd.read_csv(os.path.join(path, f"{data_name}_carla_results_n_several_cuda.csv"), index_col=0).groupby(['n_test']).mean()

# plotting

par = {'axes.titlesize':30}
plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams.update(par)

fig, ax = plt.subplots(1, sharex=True, sharey=True)

# plot lines
ax.plot(cchvae.index, cchvae['time (seconds)'], label = "CCHVAE total", linestyle="-", marker='o', color='red')
ax.plot(cchvae.index, cchvae['fitting (seconds)'], label = "CCHVAE fit", linestyle="-.", marker='o', color='red')
ax.plot(cchvae.index, cchvae['sampling (seconds)'], label = "CCHVAE sampling", linestyle=":",  marker='o',color='red')

ax.plot(mcce.index, mcce['time (seconds)'], label = "MCCE total", linestyle="-", marker='o', color='blue')
ax.plot(mcce.index, mcce['fit (seconds)'], label = "MCCE fit", linestyle="-.", marker='o', color='blue')
ax.plot(mcce.index, mcce['generate (seconds)'], label = "MCCE sampling", linestyle=":", marker='o', color='blue')
ax.plot(mcce.index, mcce['postprocess (seconds)'], label = "MCCE postprocess", linestyle="--", marker='o', color='blue')
ax.legend(title_fontsize=100)

ax.legend(prop={'size': 20})

ax.set_ylabel("seconds", size=35)
ax.set_xlabel("number of test observations", size=35)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

# plt.show()

plt.savefig(f'{data_name}_variation_in_ntest_k_{k}.png')