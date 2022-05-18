import os.path

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


def read_list(filename):
    data = []
    with open(filename, 'r') as f:

        lines = f.readlines()
        for line in lines:
            t = line.strip('\n')
            data.append(float(t))
    f.close()
    return data

config = {
    "font.family":'serif',
    "font.size":14,
    "mathtext.fontset":'stix',
    "font.serif":['SimSun'],
}


rcParams.update(config)

root_dir = "/usr/local/SDNDDPG/"
save_dir = "/usr/local/SDNDDPG/img/"

step0_499 = read_list(os.path.join(root_dir, "step.txt"))
step0_499 = [int(i) for i in step0_499]

ti0100_0499 = read_list(os.path.join(root_dir, "ti.txt"))


# reward0_01 = read_list(os.path.join(root_dir, "reward3.txt"))
# reward0_025 = read_list(os.path.join(root_dir, "reward2.txt"))
# reward0_05 = read_list(os.path.join(root_dir, "reward.txt"))
#

# loss0_01 = read_list(os.path.join(root_dir, "loss3.txt"))
# loss0_025 = read_list(os.path.join(root_dir, "loss2.txt"))
# loss0_05 = read_list(os.path.join(root_dir, "loss.txt"))

# secavg0_01 = read_list(os.path.join(root_dir, "sec_avg3.txt"))
# secavg0_025 = read_list(os.path.join(root_dir, "sec_avg2.txt"))
# secavg0_05 = read_list(os.path.join(root_dir, "sec_avg.txt"))
# secmax0_01 = read_list(os.path.join(root_dir, "sec_max3.txt"))
# secmax0_025 = read_list(os.path.join(root_dir, "sec_max2.txt"))
# secmax0_05 = read_list(os.path.join(root_dir, "sec_max.txt"))
# jit0_01 = read_list(os.path.join(root_dir, "jit3.txt"))
# jit0_025 = read_list(os.path.join(root_dir, "jit2.txt"))
# jit0_05 = read_list(os.path.join(root_dir, "jit.txt"))

# jit_ddpg = read_list(os.path.join(root_dir, "ddpg_jit.txt"))
# jit_rip = read_list(os.path.join(root_dir, "rip_jit.txt"))
# jit_ospf = read_list(os.path.join(root_dir, "ospf_jit.txt"))
# jit_rand = read_list(os.path.join(root_dir, "random_jit.txt"))

# avg_ddpg = read_list(os.path.join(root_dir, "ddpg_sec_avg.txt"))
# avg_rip = read_list(os.path.join(root_dir, "rip_sec_avg.txt"))
# avg_ospf = read_list(os.path.join(root_dir, "ospf_secavg.txt"))
# avg_rand = read_list(os.path.join(root_dir, "random_sec_avg.txt"))

max_ddpg = read_list(os.path.join(root_dir, "ddpg_sec_max.txt"))
max_rip = read_list(os.path.join(root_dir, "rip_sec_max.txt"))
max_ospf = read_list(os.path.join(root_dir, "ospf_secax.txt"))
max_rand = read_list(os.path.join(root_dir, "random_sec_max.txt"))



fig, ax = plt.subplots()
x = np.asarray(ti0100_0499)
y1 = np.asarray(max_ddpg)
y2 = np.asarray(max_rip)
y3 = np.asarray(max_ospf)
y4 = np.asarray(max_rand)
ax.plot(x, y1, label='ddpg')
ax.plot(x, y2, label='rip')
ax.plot(x, y3, label='ospf')
ax.plot(x, y4, label='random')

ax.set_xlabel('Traffic Intensity(%)')
ax.set_ylabel('max_delay(s)')
ax.legend()

plt.savefig(os.path.join(save_dir, 'secmax_.png'))

plt.show()
