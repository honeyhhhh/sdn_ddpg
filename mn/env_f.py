#!/usr/bin/env python
import jsonlines
from mininet.net import Mininet
from mininet.node import Host, OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
from mininet.clean import cleanup

import tensorflow as tf
import numpy as np
import json
import jsonlines

from subprocess import Popen
from multiprocessing import Process
import os
import time


class Traffic():
    def __init__(self, nodes_num , links_num , ratio=0.6, bw=10):
        self.nodes_num = nodes_num
        self.links_num = links_num * 2
        self.pre_traffic = None
        self.ratio = ratio
        self.bw = bw
        self.br = self.bw * self.links_num * self.ratio


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
        t = np.dot(t, self.br * (1.0 / np.sum(t)))

        T = np.full(shape=(self.nodes_num ,self.nodes_num), fill_value=0, dtype=float)
        k=0
        for i in range(self.nodes_num):
            for j in range(self.nodes_num):
                if i == j:
                    T[i][j] = -1
                else:
                    T[i][j] = t[k]
                    k+=1
        #print(np.sum(T) + self.nodes_num)
        self.pre_traffic = T
        return T

def read_iperf():
    print("read iperf result.......")
    nodes_num = 8
    total_num = (nodes_num-1)**2
    cup_sum = 0.0
    cup_avg = 0.0
    sec_sum = 0.0
    sec_avg = 0.0
    jit_sum = 0.0
    jit_avg = 0.0

    lost_sum = 0
    ooo_sum = 0

    for i in range(nodes_num):

        fn = str(".././log/client") + str(i+1) + str(".out")

        while 1:
            with open(fn, 'r+', encoding="utf-8") as f:
                t = f.read()
                obj = t.replace('}\n{', '}@@{')
                objs = obj.split('@@')
                j_len = len(objs)

                #print(j_len)
                if j_len == nodes_num - 1:
                    #print(objs)

                    for item in objs:
                        data = json.loads(item)
                        cup_sum += data['end']['cpu_utilization_percent']['host_total']
                        sec_sum += data['end']['streams'][0]['udp']['seconds']
                        jit_sum += data['end']['sum']['jitter_ms']
                        lost_sum += data['end']['streams'][0]['udp']['lost_packets']
                        ooo_sum += data['end']['streams'][0]['udp']['out_of_order']

                        # print(data['end']['cpu_utilization_percent']['host_total'])
                        # print(data['end']['streams'][0]['udp']['seconds'], data['end']['streams'][0]['udp']['jitter_ms'], data['end']['streams'][0]['udp']['lost_packets'], data['end']['streams'][0]['udp']['out_of_order'])
                        # print(data['end']['sum']['seconds'], data['end']['sum']['jitter_ms'], data['end']['sum']['lost_packets'])
                    f.seek(0)
                    f.truncate()
                    f.close()
                    break
                f.close()
                time.sleep(.05)


    cup_avg = cup_sum / total_num
    sec_avg = sec_sum / total_num
    jit_avg = jit_sum / total_num

    # print("avg cpu_utilization_pec: {}".format(cup_avg))
    # print("avg:sec: {}".format(sec_avg))
    # print("avg jitter: {}".format(jit_avg))
    #
    # print("lost packet: {}".format(lost_sum))
    # print("out of order :{}".format(ooo_sum))


    return cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum


# server
ports = [
    [-1, 5002, 5003, 5004, 5005, 5006, 5007, 5008],
    [5011, -1, 5013, 5014, 5015, 5016, 5017, 5018],
    [5021, 5022, -1, 5024, 5025, 5026, 5027, 5028],
    [5031, 5032, 5033, -1, 5035, 5036, 5037, 5038],
    [5041, 5042, 5043, 5044, -1, 5046, 5047, 5048],
    [5051, 5052, 5053, 5054, 5055, -1, 5057, 5058],
    [5061, 5062, 5063, 5064, 5065, 5066, -1, 5068],
    [5071, 5072, 5073, 5074, 5075, 5076, 5077, -1]
]

def run_ryu():
    print('run ryu')
    proc = Popen("ryu-manager /usr/local/SDNDDPG/network-algo-lab/DijkstraController.py --observe-links", shell=True)
    #/usr/local/ryu-master/ryu/app/rest_conf_switch.py /usr/local/ryu-master/ryu/app/rest_topology.py /usr/local/ryu-master/ryu/app/ofctl_rest.py

def myNetwork():

    # os.system('mn -c')
    #


    net = Mininet( topo=None,
                   build=False,
                   ipBase='10.0.0.0/8')

    info( '*** Adding controller\n' )
    c0=net.addController(name='c0',
                      controller=RemoteController,
                      ip='127.0.0.1',
                      protocol='tcp',
                      port=6653)

    info( '*** Add switches\n')
    s1 = net.addSwitch('s1', cls=OVSKernelSwitch, protocols=['OpenFlow13'], dpid='0000000000000001')
    s2 = net.addSwitch('s2', cls=OVSKernelSwitch, protocols=['OpenFlow13'], dpid='0000000000000002')
    s3 = net.addSwitch('s3', cls=OVSKernelSwitch, protocols=['OpenFlow13'], dpid='0000000000000003')
    s4 = net.addSwitch('s4', cls=OVSKernelSwitch, protocols=['OpenFlow13'], dpid='0000000000000004')
    s5 = net.addSwitch('s5', cls=OVSKernelSwitch, protocols=['OpenFlow13'], dpid='0000000000000005')
    s6 = net.addSwitch('s6', cls=OVSKernelSwitch, protocols=['OpenFlow13'], dpid='0000000000000006')
    s7 = net.addSwitch('s7', cls=OVSKernelSwitch, protocols=['OpenFlow13'], dpid='0000000000000007')
    s8 = net.addSwitch('s8', cls=OVSKernelSwitch, protocols=['OpenFlow13'], dpid='0000000000000008')
    # s9 = net.addSwitch('s9', cls=OVSKernelSwitch, protocols=['OpenFlow13'], dpid='0000000000000009')
    # s10 = net.addSwitch('s10', cls=OVSKernelSwitch, protocols=['OpenFlow13'], dpid='000000000000000a')
    # s11 = net.addSwitch('s11', cls=OVSKernelSwitch, protocols=['OpenFlow13'], dpid='000000000000000b')
    # s12 = net.addSwitch('s12', cls=OVSKernelSwitch, protocols=['OpenFlow13'], dpid='000000000000000c')
    # s13 = net.addSwitch('s13', cls=OVSKernelSwitch, protocols=['OpenFlow13'], dpid='000000000000000d')
    # s14 = net.addSwitch('s14', cls=OVSKernelSwitch, protocols=['OpenFlow13'], dpid='000000000000000e')


    info( '*** Add hosts\n')

    h1 = net.addHost('h1', cls=Host, ip='10.0.0.1',
                     defaultRoute=None, mac='00:00:00:00:00:01')
    h2 = net.addHost('h2', cls=Host, ip='10.0.0.2',
                     defaultRoute=None, mac='00:00:00:00:00:02')
    h3 = net.addHost('h3', cls=Host, ip='10.0.0.3',
                     defaultRoute=None, mac='00:00:00:00:00:03')
    h4 = net.addHost('h4', cls=Host, ip='10.0.0.4',
                     defaultRoute=None, mac='00:00:00:00:00:04')
    h5 = net.addHost('h5', cls=Host, ip='10.0.0.5',
                     defaultRoute=None, mac='00:00:00:00:00:05')
    h6 = net.addHost('h6', cls=Host, ip='10.0.0.6',
                     defaultRoute=None, mac='00:00:00:00:00:06')
    h7 = net.addHost('h7', cls=Host, ip='10.0.0.7',
                     defaultRoute=None, mac='00:00:00:00:00:07')
    h8 = net.addHost('h8', cls=Host, ip='10.0.0.8',
                     defaultRoute=None, mac='00:00:00:00:00:08')




    # h9 = net.addHost('h9', cls=Host, ip='10.0.0.9',
    #                  defaultRoute=None, mac='00:00:00:00:00:09')
    # h10 = net.addHost('h10', cls=Host, ip='10.0.0.10',
    #                   defaultRoute=None, mac='00:00:00:00:00:0a')
    # h11 = net.addHost('h11', cls=Host, ip='10.0.0.11',
    #                   defaultRoute=None, mac='00:00:00:00:00:0b')
    # h12 = net.addHost('h12', cls=Host, ip='10.0.0.12',
    #                   defaultRoute=None, mac='00:00:00:00:00:0c')
    # h13 = net.addHost('h13', cls=Host, ip='10.0.0.13',
    #                   defaultRoute=None, mac='00:00:00:00:00:0d')
    # h14 = net.addHost('h14', cls=Host, ip='10.0.0.14',
    #                   defaultRoute=None, mac='00:00:00:00:00:0e')

    info( '*** Add links\n')
    # links between h s
    net.addLink(s1, h1)
    net.addLink(s2, h2)
    net.addLink(s3, h3)
    net.addLink(s4, h4)
    net.addLink(s5, h5)
    net.addLink(s6, h6)
    net.addLink(s7, h7)
    net.addLink(s8, h8)
    # net.addLink(s9, h9)
    # net.addLink(s10, h10)
    # net.addLink(s11, h11)
    # net.addLink(s12, h12)
    # net.addLink(s13, h13)
    # net.addLink(s14, h14)

    # links between s
    #ss = {'bw':10,'max_queue_size':500}

    net.addLink(s1, s2)
    net.addLink(s1, s3)
    net.addLink(s1, s8)
    net.addLink(s2, s3)
    net.addLink(s2, s4)
    net.addLink(s3, s6)
    net.addLink(s4, s5)
    # net.addLink(s4, s11)
    net.addLink(s5, s6)
    net.addLink(s5, s7)
    # net.addLink(s6, s10)
    # net.addLink(s6, s13)
    net.addLink(s7, s8)
    # net.addLink(s8, s9)
    # net.addLink(s9, s10)
    # net.addLink(s9, s12)
    # net.addLink(s9, s14)
    # net.addLink(s11, s14)
    # net.addLink(s11, s12)
    # net.addLink(s12, s14)
    # net.addLink(s13, s14)




    info( '*** Starting network\n')
    net.build()
    info( '*** Starting controllers\n')
    for controller in net.controllers:
        controller.start()


    info( '*** Starting switches\n')
    s1.start([c0])
    s2.start([c0])
    s3.start([c0])
    s4.start([c0])
    s5.start([c0])
    s6.start([c0])
    s7.start([c0])
    s8.start([c0])
    # s9.start([c0])
    # s10.start([c0])
    # s11.start([c0])
    # s12.start([c0])
    # s13.start([c0])
    # s14.start([c0])

    HostList = [{h1:ports[0]}, {h2:ports[1]}, {h3:ports[2]}, {h4:ports[3]}, {h5:ports[4]}, {h6:ports[5]}, {h7:ports[6]}, {h8:ports[7]}]
    for i in range(len(HostList)):
        for k, v in HostList[i].items():
            for j in v:
                if j != -1:
                    k.cmd(str('iperf3') + str(' -s ') + str(' -p ') + str(j)  + str(' &'))


    info( '*** Post configure switches and hosts\n')

    # time.sleep(2)
    # p = Process(target=run_ryu())
    # p.start()
    # p.join()

    #
    time.sleep(10)
    net.pingAll()
    # t = Traffic(nodes_num=8,links_num=10, ratio=0.1)
    # net.iperfMulti(hl=HostList, tm=t.gen_traffic())
    # cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum = read_iperf()
    # print(cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum)
    # print("done1")
    # net.iperfMulti(hl=HostList, tm=t.gen_traffic())
    # cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum = read_iperf()
    # print(cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum)
    # print("done2")
    # t = Traffic(nodes_num=8,links_num=10, ratio=0.3)
    # net.iperfMulti(hl=HostList, tm=t.gen_traffic())
    # cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum = read_iperf()
    # print(cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum)
    # print("done3")
    # net.iperfMulti(hl=HostList, tm=t.gen_traffic())
    # cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum = read_iperf()
    # print(cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum)
    # print("done4")
    # t = Traffic(nodes_num=8,links_num=10, ratio=0.6)
    # net.iperfMulti(hl=HostList, tm=t.gen_traffic())
    # cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum = read_iperf()
    # print(cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum)
    # print("done5")
    # net.iperfMulti(hl=HostList, tm=t.gen_traffic())
    # cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum = read_iperf()
    # print(cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum)
    # print("done6")
    # t = Traffic(nodes_num=8,links_num=10, ratio=0.9)
    # net.iperfMulti(hl=HostList, tm=t.gen_traffic())
    # cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum = read_iperf()
    # print(cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum)
    # print("done7")
    # net.iperfMulti(hl=HostList, tm=t.gen_traffic())
    # cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum = read_iperf()
    # print(cup_avg, sec_avg, jit_avg, lost_sum, ooo_sum)
    # print("done8")


    # CLI(net)
    #net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    myNetwork()
    # cleanup()
    #
    # nodes_num = 8
    # total_num = (nodes_num-1)**2
    # cup_sum = 0.0
    # cup_avg = 0.0
    # sec_sum = 0.0
    # sec_avg = 0.0
    # jit_sum = 0.0
    # jit_avg = 0.0
    #
    # lost_sum = 0
    # ooo_sum = 0
    #
    # for i in range(nodes_num):
    #
    #     fn = str(".././log/client") + str(i+1) + str(".out")
    #     j_len = 0
    #     while j_len != nodes_num - 1:
    #         #print("dead?")
    #         with open(fn, 'r+', encoding="utf-8") as f:
    #             t = f.read()
    #             objs = t.replace('}\n{', '}@@{')
    #             objs = objs.split('@@')
    #             j_len = len(objs)
    #             if j_len == nodes_num - 1:
    #                 for item in objs:
    #                     data = json.loads(item)
    #                     cup_sum += data['end']['cpu_utilization_percent']['host_total']
    #                     sec_sum += data['end']['streams'][0]['udp']['seconds']
    #                     jit_sum += data['end']['sum']['jitter_ms']
    #                     lost_sum += data['end']['streams'][0]['udp']['lost_packets']
    #                     ooo_sum += data['end']['streams'][0]['udp']['out_of_order']
    #
    #                     # print(data['end']['cpu_utilization_percent']['host_total'])
    #                     # print(data['end']['streams'][0]['udp']['seconds'], data['end']['streams'][0]['udp']['jitter_ms'], data['end']['streams'][0]['udp']['lost_packets'], data['end']['streams'][0]['udp']['out_of_order'])
    #                     # print(data['end']['sum']['seconds'], data['end']['sum']['jitter_ms'], data['end']['sum']['lost_packets'])
    #                 f.seek(0)
    #                 f.truncate()
    #             f.close()
    #
    #
    # cup_avg = cup_sum / total_num
    # sec_avg = sec_sum / total_num
    # jit_avg = jit_sum / total_num
    #
    # print("avg cpu_utilization_pec: {}".format(cup_avg))
    # print("avg:sec: {}".format(sec_avg))
    # print("avg jitter: {}".format(jit_avg))
    #
    # print("lost packet: {}".format(lost_sum))
    # print("out of order :{}".format(ooo_sum))
    #
    # #
    # #
    #
    #





        # tt = t.replace('}\n{', '}{')
        # print(tt)
        # for it in jsonlines.Reader(tt):
        #     print(it)


# dpctl dump-flows -O OpenFlow13 查看流表
# dpctl del-flows -O OpenFlow13 "priority=1"
# cookie=0x0, duration=2580.176s, table=0, n_packets=8, n_bytes=480, priority=65535,dl_dst=01:80:c2:00:00:0e,dl_type=0x88cc actions=CONTROLLER:65535
#  cookie=0x0, duration=2560.859s, table=0, n_packets=172468, n_bytes=11382824, priority=1,in_port="s1-eth2",dl_src=00:00:00:00:00:02,dl_dst=00:00:00:00:00:01 actions=output:"s1-eth1"
#  cookie=0x0, duration=2556.187s, table=0, n_packets=253909, n_bytes=11260638394, priority=1,in_port="s1-eth1",dl_src=00:00:00:00:00:01,dl_dst=00:00:00:00:00:02 actions=output:"s1-eth2"
#  cookie=0x0, duration=2553.291s, table=0, n_packets=7, n_bytes=406, priority=1,in_port="s1-eth3",dl_src=00:00:00:00:00:03,dl_dst=00:00:00:00:00:01 actions=output:"s1-eth1"
#  cookie=0x0, duration=2550.491s, table=0, n_packets=4, n_bytes=224, priority=1,in_port="s1-eth1",dl_src=00:00:00:00:00:01,dl_dst=00:00:00:00:00:03 actions=output:"s1-eth3"
#  cookie=0x0, duration=2547.863s, table=0, n_packets=8, n_bytes=448, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:04,dl_dst=00:00:00:00:00:01 actions=output:"s1-eth1"
#  cookie=0x0, duration=2545.446s, table=0, n_packets=4, n_bytes=224, priority=1,in_port="s1-eth1",dl_src=00:00:00:00:00:01,dl_dst=00:00:00:00:00:04 actions=output:"s1-eth4"
#  cookie=0x0, duration=2544.179s, table=0, n_packets=7, n_bytes=406, priority=1,in_port="s1-eth3",dl_src=00:00:00:00:00:05,dl_dst=00:00:00:00:00:01 actions=output:"s1-eth1"
#  cookie=0x0, duration=2541.776s, table=0, n_packets=4, n_bytes=224, priority=1,in_port="s1-eth1",dl_src=00:00:00:00:00:01,dl_dst=00:00:00:00:00:05 actions=output:"s1-eth4"
#  cookie=0x0, duration=2539.317s, table=0, n_packets=8, n_bytes=448, priority=1,in_port="s1-eth3",dl_src=00:00:00:00:00:06,dl_dst=00:00:00:00:00:01 actions=output:"s1-eth1"
#  cookie=0x0, duration=2536.794s, table=0, n_packets=4, n_bytes=224, priority=1,in_port="s1-eth1",dl_src=00:00:00:00:00:01,dl_dst=00:00:00:00:00:06 actions=output:"s1-eth3"
#  cookie=0x0, duration=2534.275s, table=0, n_packets=8, n_bytes=448, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:07,dl_dst=00:00:00:00:00:01 actions=output:"s1-eth1"
#  cookie=0x0, duration=2531.865s, table=0, n_packets=4, n_bytes=224, priority=1,in_port="s1-eth1",dl_src=00:00:00:00:00:01,dl_dst=00:00:00:00:00:07 actions=output:"s1-eth4"
#  cookie=0x0, duration=2530.600s, table=0, n_packets=7, n_bytes=406, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:08,dl_dst=00:00:00:00:00:01 actions=output:"s1-eth1"
#  cookie=0x0, duration=2528.160s, table=0, n_packets=4, n_bytes=224, priority=1,in_port="s1-eth1",dl_src=00:00:00:00:00:01,dl_dst=00:00:00:00:00:08 actions=output:"s1-eth4"
#  cookie=0x0, duration=2525.593s, table=0, n_packets=7, n_bytes=406, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:09,dl_dst=00:00:00:00:00:01 actions=output:"s1-eth1"
#  cookie=0x0, duration=2523.174s, table=0, n_packets=4, n_bytes=224, priority=1,in_port="s1-eth1",dl_src=00:00:00:00:00:01,dl_dst=00:00:00:00:00:09 actions=output:"s1-eth4"
#  cookie=0x0, duration=2520.683s, table=0, n_packets=8, n_bytes=448, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:0a,dl_dst=00:00:00:00:00:01 actions=output:"s1-eth1"
#  cookie=0x0, duration=2518.265s, table=0, n_packets=4, n_bytes=224, priority=1,in_port="s1-eth1",dl_src=00:00:00:00:00:01,dl_dst=00:00:00:00:00:0a actions=output:"s1-eth4"
#  cookie=0x0, duration=2515.802s, table=0, n_packets=20, n_bytes=1512, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:0b,dl_dst=00:00:00:00:00:01 actions=output:"s1-eth1"
#  cookie=0x0, duration=2513.249s, table=0, n_packets=16, n_bytes=1288, priority=1,in_port="s1-eth1",dl_src=00:00:00:00:00:01,dl_dst=00:00:00:00:00:0b actions=output:"s1-eth4"
#  cookie=0x0, duration=2510.782s, table=0, n_packets=11, n_bytes=686, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:0c,dl_dst=00:00:00:00:00:01 actions=output:"s1-eth1"
#  cookie=0x0, duration=2508.371s, table=0, n_packets=8, n_bytes=504, priority=1,in_port="s1-eth1",dl_src=00:00:00:00:00:01,dl_dst=00:00:00:00:00:0c actions=output:"s1-eth4"
#  cookie=0x0, duration=2507.089s, table=0, n_packets=7, n_bytes=406, priority=1,in_port="s1-eth3",dl_src=00:00:00:00:00:0d,dl_dst=00:00:00:00:00:01 actions=output:"s1-eth1"
#  cookie=0x0, duration=2504.606s, table=0, n_packets=4, n_bytes=224, priority=1,in_port="s1-eth1",dl_src=00:00:00:00:00:01,dl_dst=00:00:00:00:00:0d actions=output:"s1-eth3"
#  cookie=0x0, duration=2502.063s, table=0, n_packets=11, n_bytes=2044, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:0e,dl_dst=00:00:00:00:00:01 actions=output:"s1-eth1"
#  cookie=0x0, duration=2499.652s, table=0, n_packets=4467, n_bytes=6745340, priority=1,in_port="s1-eth1",dl_src=00:00:00:00:00:01,dl_dst=00:00:00:00:00:0e actions=output:"s1-eth4"
#  cookie=0x0, duration=2481.142s, table=0, n_packets=7, n_bytes=406, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:07,dl_dst=00:00:00:00:00:02 actions=output:"s1-eth2"
#  cookie=0x0, duration=2478.620s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth2",dl_src=00:00:00:00:00:02,dl_dst=00:00:00:00:00:07 actions=output:"s1-eth4"
#  cookie=0x0, duration=2476.170s, table=0, n_packets=8, n_bytes=448, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:08,dl_dst=00:00:00:00:00:02 actions=output:"s1-eth2"
#  cookie=0x0, duration=2474.963s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth2",dl_src=00:00:00:00:00:02,dl_dst=00:00:00:00:00:08 actions=output:"s1-eth4"
#  cookie=0x0, duration=2473.683s, table=0, n_packets=7, n_bytes=406, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:09,dl_dst=00:00:00:00:00:02 actions=output:"s1-eth2"
#  cookie=0x0, duration=2471.255s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth2",dl_src=00:00:00:00:00:02,dl_dst=00:00:00:00:00:09 actions=output:"s1-eth4"
#  cookie=0x0, duration=2468.679s, table=0, n_packets=8, n_bytes=448, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:0a,dl_dst=00:00:00:00:00:02 actions=output:"s1-eth2"
#  cookie=0x0, duration=2466.203s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth2",dl_src=00:00:00:00:00:02,dl_dst=00:00:00:00:00:0a actions=output:"s1-eth4"
#  cookie=0x0, duration=2458.850s, table=0, n_packets=7, n_bytes=406, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:0c,dl_dst=00:00:00:00:00:02 actions=output:"s1-eth2"
#  cookie=0x0, duration=2456.316s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth2",dl_src=00:00:00:00:00:02,dl_dst=00:00:00:00:00:0c actions=output:"s1-eth4"
#  cookie=0x0, duration=2451.328s, table=0, n_packets=8, n_bytes=448, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:0e,dl_dst=00:00:00:00:00:02 actions=output:"s1-eth2"
#  cookie=0x0, duration=2448.896s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth2",dl_src=00:00:00:00:00:02,dl_dst=00:00:00:00:00:0e actions=output:"s1-eth4"
#  cookie=0x0, duration=2435.312s, table=0, n_packets=7, n_bytes=406, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:07,dl_dst=00:00:00:00:00:03 actions=output:"s1-eth3"
#  cookie=0x0, duration=2432.809s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth3",dl_src=00:00:00:00:00:03,dl_dst=00:00:00:00:00:07 actions=output:"s1-eth4"
#  cookie=0x0, duration=2430.306s, table=0, n_packets=7, n_bytes=406, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:08,dl_dst=00:00:00:00:00:03 actions=output:"s1-eth3"
#  cookie=0x0, duration=2427.896s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth3",dl_src=00:00:00:00:00:03,dl_dst=00:00:00:00:00:08 actions=output:"s1-eth4"
#  cookie=0x0, duration=2425.370s, table=0, n_packets=8, n_bytes=448, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:09,dl_dst=00:00:00:00:00:03 actions=output:"s1-eth3"
#  cookie=0x0, duration=2422.963s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth3",dl_src=00:00:00:00:00:03,dl_dst=00:00:00:00:00:09 actions=output:"s1-eth4"
#  cookie=0x0, duration=2421.619s, table=0, n_packets=7, n_bytes=406, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:0a,dl_dst=00:00:00:00:00:03 actions=output:"s1-eth3"
#  cookie=0x0, duration=2419.194s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth3",dl_src=00:00:00:00:00:03,dl_dst=00:00:00:00:00:0a actions=output:"s1-eth4"
#  cookie=0x0, duration=2411.861s, table=0, n_packets=7, n_bytes=406, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:0c,dl_dst=00:00:00:00:00:03 actions=output:"s1-eth3"
#  cookie=0x0, duration=2409.321s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth3",dl_src=00:00:00:00:00:03,dl_dst=00:00:00:00:00:0c actions=output:"s1-eth4"
#  cookie=0x0, duration=2401.983s, table=0, n_packets=8, n_bytes=448, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:0e,dl_dst=00:00:00:00:00:03 actions=output:"s1-eth3"
#  cookie=0x0, duration=2399.557s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth3",dl_src=00:00:00:00:00:03,dl_dst=00:00:00:00:00:0e actions=output:"s1-eth4"
#  cookie=0x0, duration=2317.773s, table=0, n_packets=8, n_bytes=448, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:08,dl_dst=00:00:00:00:00:06 actions=output:"s1-eth3"
#  cookie=0x0, duration=2315.341s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth3",dl_src=00:00:00:00:00:06,dl_dst=00:00:00:00:00:08 actions=output:"s1-eth4"
#  cookie=0x0, duration=2312.872s, table=0, n_packets=8, n_bytes=448, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:09,dl_dst=00:00:00:00:00:06 actions=output:"s1-eth3"
#  cookie=0x0, duration=2310.457s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth3",dl_src=00:00:00:00:00:06,dl_dst=00:00:00:00:00:09 actions=output:"s1-eth4"
#  cookie=0x0, duration=2233.531s, table=0, n_packets=7, n_bytes=406, priority=1,in_port="s1-eth3",dl_src=00:00:00:00:00:0d,dl_dst=00:00:00:00:00:08 actions=output:"s1-eth4"
#  cookie=0x0, duration=2231.111s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:08,dl_dst=00:00:00:00:00:0d actions=output:"s1-eth3"
#  cookie=0x0, duration=2212.380s, table=0, n_packets=8, n_bytes=448, priority=1,in_port="s1-eth3",dl_src=00:00:00:00:00:0d,dl_dst=00:00:00:00:00:09 actions=output:"s1-eth4"
#  cookie=0x0, duration=2209.946s, table=0, n_packets=5, n_bytes=322, priority=1,in_port="s1-eth4",dl_src=00:00:00:00:00:09,dl_dst=00:00:00:00:00:0d actions=output:"s1-eth3"
#  cookie=0x0, duration=2580.186s, table=0, n_packets=576, n_bytes=27199, priority=0 actions=CONTROLLER:65535