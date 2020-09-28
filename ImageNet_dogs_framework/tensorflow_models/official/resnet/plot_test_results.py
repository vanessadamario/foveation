import os
import numpy as np
from os.path import join
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

scenario_list = range(5)
n_train_list = [23, 45, 90]

test_accuracies = np.zeros((len(scenario_list), len(n_train_list)))

for id_s_, s_ in enumerate(scenario_list):
    for id_n_, n_ in enumerate(n_train_list):
        scenario_ = "scenario_%i/n_%i_0" % (s_, n_)
        path_ = join(scenario_, "eval")
        file_list = sorted(os.listdir(path_))

        validation_accuracy = np.array([])
        for file_ in file_list:
            tmp_vals = []
            for e_ in tf.train.summary_iterator(join(path_, file_)):
                for v_ in e_.summary.value:
                    if v_.tag == "accuracy":
                        tmp_vals.append(v_.simple_value)
            if len(tmp_vals) > 1:
                validation_accuracy = np.append(validation_accuracy, np.array(tmp_vals))
            else:
                test_accuracy = tmp_vals[0]
                test_accuracies[id_s_, id_n_] = test_accuracy
                print('test accuracy %f' % test_accuracy)

        validation_accuracy = validation_accuracy.reshape(-1, )

        # plt.plot(validation_accuracy)
        # plt.plot(len(validation_accuracy), test_accuracy, marker='o', ms=10)
        #plt.show()

count = 0
shift = 0

plt.figure(figsize=(15, 10))
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.titleweight"] = "bold"

plt.rc("xtick", labelsize=50)
plt.rc("ytick", labelsize=50)

rf_color = {0: "skyblue",
            1: "lightslategrey",
            2: "navy",
            3: "purple",
            4: "palevioletred"}

rf_legend = {0: "Case 1",
             1: "Case 2",
             2: "Case 3",
             3: "Case 4",
             4: "Case 5"}

n_axis = []
for id_s_, s_ in enumerate(scenario_list):
    for id_n_, n_ in enumerate(n_train_list):
        label = rf_legend[id_s_] if id_n_ == len(n_train_list) - 1 else None
        n_axis.append(id_n_ + id_s_ * len(n_train_list) + shift)
        plt.bar(id_n_ + id_s_ * len(n_train_list) + shift,
                test_accuracies[id_s_][len(n_train_list) - id_n_ - 1],
                label=label,
                color=rf_color[id_s_],
                alpha=1 / (len(n_train_list) - id_n_))
    shift += 1
plt.xlabel("Training examples per class", fontsize=55, labelpad=10)
plt.ylabel("Top-1 test accuracy", fontsize=55)
plt.xticks(n_axis, np.array([n_train_list[::-1]] * 5).reshape(-1, ),
           fontsize=40, rotation=90)
for jj in range(4):
    plt.axhline(0.1 * jj, color="gainsboro")

dct_font = dict(weight="bold")
for kk_ in range(5):
    plt.text(kk_ * 3 + kk_, 0.35, rf_legend[kk_], size=35, color=rf_color[kk_], fontdict=dct_font)

plt.axhline(1 / 120, color="grey", linestyle="--")
sns.set_style(style="white")

plt.subplots_adjust(top=0.85)
sns.despine()
plt.title("Stanford Dogs", fontsize=65, pad=100)
plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
plt.savefig("test_results.pdf")
