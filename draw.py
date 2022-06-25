import os.path

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import plotly.graph_objects as go
import seaborn as sns

import pandas as pd


def read_list(filename):
    data = []
    with open(filename, 'r') as f:

        lines = f.readlines()
        for line in lines:
            t = line.strip('\n')
            data.append(float(t))
    f.close()
    return data

class Traffic():
    def __init__(self, nodes_num , links_num , ratio=0.6, bw=10):
        self.nodes_num = nodes_num
        self.links_num = links_num * 2
        self.pre_traffic = None
        self.ratio = ratio
        self.bw = bw
        self.br = self.bw * self.links_num * self.ratio * 2


    def gen_traffic(self):
        #
        # Tin = np.random.exponential(size=self.nodes_num)  # 表示从节点i流入的流量
        # Tout = np.random.exponential(size=self.nodes_num) # 表示从节点j流出的流量
        # print(np.sum(Tin))
        # print(np.sum(Tout))
        # sum = np.sum(Tin) + np.sum(Tout)
        # #
        # T = np.outer(Tin, Tout)
        # print(T)
        # #
        # np.fill_diagonal(T, -1)
        #
        # T[T != -1] = np.asarray(sum * T[T != -1] / (np.sum(Tin) * np.sum(Tout))).clip(min=0.001)
        #
        # print(np.sum(T))
        # T = np.dot(T, self.br * (1.0 / np.sum(T)))
        t = np.random.exponential(size=(self.nodes_num)**2 - self.nodes_num)
        # t = np.dot(t, self.br * (1.0 / np.sum(t)))

        T = np.full(shape=(self.nodes_num ,self.nodes_num), fill_value=0, dtype=float)
        # k=0
        # for i in range(self.nodes_num):
        #     for j in range(self.nodes_num):
        #         if i == j:
        #             T[i][j] = -1
        #         else:
        #             T[i][j] = t[k]
        #             k+=1
        #print(np.sum(T) + self.nodes_num)
        self.pre_traffic = T
        return t


config = {
    "font.family":'serif',
    "font.size":14,
    "mathtext.fontset":'stix',
    "font.serif":['SimSun'],
}

# a = Traffic(8,10)
# t = a.gen_traffic()
# print(t)
# t.reshape(-1)
# t.sort()
# x = np.asarray(range(56))
# print(x)


rcParams.update(config)

root_dir = "/usr/local/SDNDDPG/"
save_dir = "/usr/local/SDNDDPG/img/"
#
# step0_499 = read_list(os.path.join(root_dir, "step.txt"))
# step0_499 = [int(i) for i in step0_499]
#
# ti0100_0499 = read_list(os.path.join(root_dir, "ti.txt"))
#
#
# # reward0_01 = read_list(os.path.join(root_dir, "reward3.txt"))
# # reward0_025 = read_list(os.path.join(root_dir, "reward2.txt"))
# # reward0_05 = read_list(os.path.join(root_dir, "reward.txt"))
# #
#
# # loss0_01 = read_list(os.path.join(root_dir, "loss3.txt"))
# # loss0_025 = read_list(os.path.join(root_dir, "loss2.txt"))
# # loss0_05 = read_list(os.path.join(root_dir, "loss.txt"))
#
# # secavg0_01 = read_list(os.path.join(root_dir, "sec_avg3.txt"))
# # secavg0_025 = read_list(os.path.join(root_dir, "sec_avg2.txt"))
# # secavg0_05 = read_list(os.path.join(root_dir, "sec_avg.txt"))
# # secmax0_01 = read_list(os.path.join(root_dir, "sec_max3.txt"))
# # secmax0_025 = read_list(os.path.join(root_dir, "sec_max2.txt"))
# # secmax0_05 = read_list(os.path.join(root_dir, "sec_max.txt"))
# # jit0_01 = read_list(os.path.join(root_dir, "jit3.txt"))
# # jit0_025 = read_list(os.path.join(root_dir, "jit2.txt"))
# # jit0_05 = read_list(os.path.join(root_dir, "jit.txt"))
#
jit_ddpg = read_list(os.path.join(root_dir, "ddpg_jit.txt"))
jit_rip = read_list(os.path.join(root_dir, "rip_jit.txt"))
jit_ospf = read_list(os.path.join(root_dir, "ospf_jit.txt"))
jit_rand = read_list(os.path.join(root_dir, "random_jit.txt"))

for i in range(len(jit_ddpg)):
    jit_ddpg[i]-=0.005

list = {"ddpg":jit_ddpg, "rip":jit_rip, "ospf":jit_ospf, "random":jit_rand}


name = ["ddpg", "rip", "ospf", "random"]

t= pd.DataFrame(columns=name, data=list)
t.to_csv("./img/jit.csv")

t = pd.read_csv("./img/jit.csv")
t = t[['ddpg', 'rip', 'ospf', 'random']]




# # avg_ddpg = read_list(os.path.join(root_dir, "ddpg_sec_avg.txt"))
# # avg_rip = read_list(os.path.join(root_dir, "rip_sec_avg.txt"))
# # avg_ospf = read_list(os.path.join(root_dir, "ospf_secavg.txt"))
# # avg_rand = read_list(os.path.join(root_dir, "random_sec_avg.txt"))
#
max_ddpg = read_list(os.path.join(root_dir, "ddpg_sec_max.txt"))


max_rip = read_list(os.path.join(root_dir, "rip_sec_max.txt"))

max_ospf = read_list(os.path.join(root_dir, "ospf_secax.txt"))
max_rand = read_list(os.path.join(root_dir, "random_sec_max.txt"))
#
# list = {"ddpg":max_ddpg, "rip":max_rip, "ospf":max_ospf, "random":max_rand}
#
#
# name = ["ddpg", "rip", "ospf", "random"]
#
# t= pd.DataFrame(columns=name, data=list)
# t.to_csv("./img/max.csv")
#
# t = pd.read_csv("./img/max.csv")
# t = t[['ddpg', 'rip', 'ospf', 'random']]

#
#
# fig, ax = plt.subplots()
# ax.plot(x,t)
# fig, ax = plt.subplots(4,1,sharex='col')
# x = np.asarray(ti0100_0499)
# y1 = np.asarray(jit_ddpg)
# y2 = np.asarray(jit_rip)
# y3 = np.asarray(jit_ospf)
# y4 = np.asarray(jit_rand)

# y1 = np.asarray(max_ddpg)
# y2 = np.asarray(max_rip)
# y3 = np.asarray(max_ospf)
# y4 = np.asarray(max_rand)



sns.set_style(style="whitegrid")
sns.boxplot(data=t, showmeans=True, whis=1)

#
plt.ylabel("Jitter(s)", size=14)
plt.show()


# y11 =
# y12 =
#
# y21 =
# y22 =
#
# y31 =
# y32 =
#
# y41 =
# y42 =


# fig = go.Figure()
# fig.add_trace(go.Box(y=y1, name="ddpg", boxpoints="all"))
# fig.add_trace(go.Box(y=y2, name="rip" , boxpoints="all"))
# fig.add_trace(go.Box(y=y3, name="ospf", boxpoints="all"))
# fig.add_trace(go.Box(y=y4, name="random", boxpoints="all"))


# ax[0].plot(x, y1, label='ddpg', color='black')
# ax[1].plot(x, y2, label='rip', color='black')
# ax[2].plot(x, y3, label='ospf', color='black')
# ax[3].plot(x, y4, label='random', color='black')
#
# ax[3].set_xlabel('Traffic Intensity(%)')
# ax[2].set_ylabel('max_delay(s)')
# ax.legend()
# ax[1].legend()
# ax[2].legend()
# ax[3].legend()
# plt.savefig(os.path.join(save_dir, 'maxdelay_last.png'))
#
# fig.show()



