import copy
import warnings
from random import randint

import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations

from ryu.base import app_manager
from ryu.lib import hub
from ryu.controller import ofp_event
from ryu.controller.handler import (CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls)
from ryu.lib.packet import arp, ether_types, ethernet, packet
from ryu.ofproto import ofproto_v1_3
from ryu.topology import event
from ryu.topology.api import get_link, get_switch



import Actor
import Critic
import ReplayBuffer
import OUNoise


from mininet.net import Mininet
from mininet.node import Host, OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import info


import tensorflow as tf
import numpy as np
import json


from subprocess import Popen
from multiprocessing import Process
import os
import time


tmd = 0

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
    # print("read iperf result.......")
    nodes_num = 8
    total_num = (nodes_num-1)**2
    cup_sum = 0.0
    sec_sum = 0.0
    sec_max = 0.0
    jit_sum = 0.0


    lost_sum = 0
    ooo_sum = 0

    for i in range(nodes_num):

        fn = str("./log/client") + str(i+1) + str(".out")

        while 1:
            with open(fn, 'r+', encoding="utf-8") as f:
                t = f.read()
                obj = t.replace('}\n{', '}@@{')
                objs = obj.split('@@')
                j_len = len(objs)

                # print(j_len)
                if j_len == nodes_num - 1:
                    #print(objs)

                    for item in objs:
                        data = json.loads(item)
                        cup_sum += data['end']['cpu_utilization_percent']['host_total']
                        sec = data['end']['streams'][0]['udp']['seconds']
                        sec_max = max(sec_max, sec)
                        sec_sum += sec
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
                time.sleep(.01)


    cup_avg = cup_sum / total_num
    sec_avg = sec_sum / total_num
    jit_avg = jit_sum / total_num

    # print("avg cpu_utilization_pec: {}".format(cup_avg))
    # print("avg:sec: {}".format(sec_avg))
    # print("avg jitter: {}".format(jit_avg))
    #
    # print("lost packet: {}".format(lost_sum))
    # print("out of order :{}".format(ooo_sum))


    return cup_avg, sec_avg, jit_avg, lost_sum, sec_max


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
    # time.sleep(10)
    # net.pingAll()
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
    time.sleep(3)

    return net, HostList


def txt_save(fn, data):
    file = open(fn, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']','')
        s = s.replace("'",'').replace(',','') + "\n"
        file.write(s)
    file.close()


class UnionFindSet():
    """
    并查集
    """

    def __init__(self, n: int) -> None:
        self.fa = [i for i in range(0, n + 1)]
        self.size = n

    def find(self, x: int) -> int:
        if self.fa[x] != x:
            self.fa[x] = self.find(self.fa[x])
        return self.fa[x]

    def union(self, x: int, y: int) -> None:
        x = self.find(x)
        y = self.find(y)
        if x != y:
            self.fa[y] = x


class BucketSet(object):
    """
    循环桶类
    """
    __header_location = 0  # 桶头位置
    __thing_amount = 0     # 所有桶中节点数量

    def __init__(self, bucket_num):
        self.__bucket_num = bucket_num     # 桶数量
        self.__buckets = [None]*bucket_num  # 建桶
        self.__init_buckets(bucket_num)     # 初始化桶
        self.__header = self.__buckets[0]   # 桶头是第一个桶

    def __move_header_to_next(self):
        """
        头节点下移一个位置
        :return: None
        """
        self.__header_location = (self.__header_location+1) % self.__bucket_num
        self.__header = self.__buckets[self.__header_location]

    def __init_buckets(self, bucket_num):
        """
        初始化桶
        :param bucket_num: 桶数量
        :return: None
        """
        for i in range(bucket_num):
            self.__buckets[i] = BucketSet.__Bucket(i)

    def __hash(self, length):
        """
        根据距离标记计算哈希值，决定放入的桶
        :param length: 距离标记
        :return: 返回桶编号
        """
        return length % self.__bucket_num

    def add_thing(self, thing):
        """
        节点添加到桶中
        :param thing:
        :return:
        """
        length = int(thing[0]*100)   # 距离标记

        bucket_id = self.__hash(length)  # 计算桶编号
        self.__buckets[bucket_id].add_thing(thing)  # 放入桶中
        self.__thing_amount += 1   # 更新节点数量

    def is_empty(self):
        """
        判断循环桶是否空
        :return: Boolean  所有桶空返回TRUE，否者FALSE
        """
        return self.__thing_amount == 0

    def pop_min(self):
        """
        取最小距离标记的的节点
        :return: 返回距离标记最小的桶的集合
        """
        while self.__header.is_empty():  # 该桶空就往后查询
            self.__move_header_to_next()
        min_things = self.__header.pop()  # 取桶中所有节点
        self.__thing_amount = self.__thing_amount - \
            len(min_things)  # 更新循环桶中节点数量
        self.__move_header_to_next()     # 头桶往后移动
        return min_things.copy()

    class __Bucket(object):
        """
        桶类
        """
        __thing_amount = 0  # 桶中的节点

        def __init__(self, bucket_id):
            """
            初始化
            :param bucket_id: 桶id
            """
            self.list_thing = list()  # 节点容器list
            self.id = bucket_id

        def add_thing(self, thing):
            """
            往桶里放节点
            :param thing:
            :return:
            """
            self.list_thing.append(thing)
            self.__thing_amount += 1  # 更新节点数量

        def pop(self):
            """
            取出桶内节点
            :return: 返回节点集合
            """
            things = self.list_thing.copy()
            self.list_thing.clear()
            self.__thing_amount = 0  # 桶内节点数量更新为0
            return things

        def is_empty(self):
            """
            判断桶是否为空
            :return: 桶空返回true,否则返回FALSE
            """
            return self.__thing_amount == 0


class Topo(nx.DiGraph): # 有向图
    """
    网络拓扑，继承自 networkx.DiGraph
    """

    def __init__(self):
        super().__init__()
        warnings.filterwarnings("ignore", category=UserWarning)
        self.plot_options = {
            "font_size": 20,
            "node_size": 1500,
            "node_color": "white",  # 节点背景颜色
            "linewidths": 3,
            "width": 3,
            "with_labels": True
        }
        self.pos = nx.spring_layout(self)
        plt.figure(1, figsize=(18, 14))
        plt.ion()

    def dijkstra(self, src: int, dst: int, first_port: int, last_port: int, flag: int):
        # print("dijkstra....")
        """
        获取单源最短路并返回最短路
        """
        #print("Calculating all the path of {}:{} -> {}:{}...".format( src, first_port, dst, last_port))

        shortest_path = []

        # 桶实现的 Dijkstra
        pre = {}

        buckets = BucketSet(10+1)           # 创建循环桶对象
        node_src = (0, src)                 # 源节点二元组（距离标记，节点）
        visited = {}                        # 永久标记集合key是标记节点，所有value是1
        buckets.add_thing(node_src)         # 添加源节点到桶
        path_length = {}                    # 路径长度字典
        while visited.get(dst) is None:     # 没有到终点
            min_list = buckets.pop_min()    # 取最小距离标记集合 返回列表
            while not len(min_list) == 0:
                min_node = min_list.pop()
                if visited.get(min_node[1]) is None:        # 该节点没有永久标记
                    visited[min_node[1]] = 1                # 永久标记该节点
                    path_length[min_node[1]] = min_node[0]  # 记下路径长度
                    
                    # 计算前驱
                    for n in self.predecessors(min_node[1]):
                        if min_node[1] != src and n in visited:
                            if path_length[min_node[1]] - path_length[n] == self.edges[min_node[1], n]['weight']:
                                pre[min_node[1]] = n
                        # print(min_node[1], n, path_length)

                    # 将所有邻接节点加入桶中
                    for v in self.neighbors(min_node[1]):
                        if visited.get(v) is None:
                            buckets.add_thing(
                                (self.edges[min_node[1], v]['weight'] + min_node[0], v))

        i = dst
        shortest_path += [dst]
        while i != src:
            shortest_path += [pre[i]]
            i = pre[i]

        shortest_path.reverse()

        #print("Shortest path:", shortest_path, "length:", len(shortest_path), sep=' ')

        # 绘图
        # if flag % 500 == 0 and flag !=0:
        #     self.draw_path(shortest_path, flag)

        # 生成路径：(src, in_port, out_port)->(s2, in_port, out_port)->...->(dst, in_port, out_port)
        ryu_path = []
        in_port = first_port
        for s1, s2 in zip(shortest_path[:-1], shortest_path[1:]):
            out_port = self.edges[s1, s2]["src_port"]
            ryu_path.append((s1, in_port, out_port))
            in_port = self.edges[s2, s1]["src_port"]
        ryu_path.append((dst, in_port, last_port))

        return ryu_path

    def draw_path(self, path: "list[int]", flag: int):

        edge_to_display = []

        for s1, s2 in zip(path[:-1], path[1:]):
            edge_to_display.append((s1, s2))
            edge_to_display.append((s2, s1))

        edge_colors = [
            "red" if e in edge_to_display else 'black' for e in list(self.edges)]
        node_edge_colors = [
            "red" if n in path else "black" for n in list(self.nodes())]

        plt.clf()
        plt.title("Step{}: Shortest Path from {} to {}".format(flag, path[0], path[-1]))
        nx.draw(self, pos=self.pos, edge_color=edge_colors,
                edgecolors=node_edge_colors, **self.plot_options)

        edge_labels = {}
        for e in list(self.edges(data=True)):
            edge_labels[e[0:2]] = e[2]["weight"]
        nx.draw_networkx_edge_labels(
            self, pos=self.pos, edge_labels=edge_labels)

        plt.savefig("topo"+ str(flag)+"_" + str(path[0]) + "_"+ str(path[-1]) +".png")



class DijkstraController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DijkstraController, self).__init__(*args, **kwargs)
        self.net, self.HostList = myNetwork()


        self.datapaths = []  # 保存受控制的交换机
        self.host_mac_to = {}  # 记录与主机直接相连的交换机 ID 与端口
        self.topo = Topo()  # 控制器发现的拓扑
        self.arp_history = {}  # ARP 历史记录
        self.scount = 0   # 记录交换机进入数量

        # env
        self.ACTIVE_NODES = 8
        self.LINKS_NUM = 10
        self.a_dim = self.LINKS_NUM
        self.s_dim = self.ACTIVE_NODES**2 - self.ACTIVE_NODES
        self.tgen = Traffic(self.ACTIVE_NODES, self.LINKS_NUM, ratio=0.105)
        self.env_T = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)
        self.env_W = np.full([self.a_dim], -1.0, dtype=float)
        self.env_edge = []  # 保存无向边

        self.train_thread = hub.spawn(self._main)
        self.topo_done = 0

        self.avg_cpup = 0.0
        self.avg_sec = 0.0
        self.avg_jit = 0.0
        self.loss_p = 0
        self.sec_max = 0.0


        self.total_step_count = 0


    def _main(self):
        print("Initializing....")

        # print(self.net.pingAll())
        time.sleep(8)
        # for dp in self.datapaths:
        #     self.delete_flow(dp)
        # print("del dlows done !")
        # assert self.topo_done == 1
        self.play()



    def find_datapath_by_id(self, dpid: int):
        for datapath in self.datapaths:
            if datapath.id == dpid:
                return datapath
        return None

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """
        处理 Feature-request 消息，下发默认流表
        """
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    # 实现路径切换：用新流表替换旧流表，从而改变数据包的转发端口
    # 控制器感知 some change
    def delete_flow(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        actions = [parser.OFPActionOutput(ofproto.OFPP_NORMAL, 0)]
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        for dst in self.host_mac_to.keys():
            #print(dst)
            match = parser.OFPMatch(eth_dst=dst)
            mod = parser.OFPFlowMod(
                datapath=datapath,cookie=0, cookie_mask=0, table_id=0, command=ofproto.OFPFC_DELETE,idle_timeout=0, hard_timeout=0, buffer_id=ofproto.OFP_NO_BUFFER, flags=ofproto.OFPFF_SEND_FLOW_REM,
                out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY,
                priority=1, match=match, instructions=inst)
            datapath.send_msg(mod)
            time.sleep(.01)

    def add_flow(self, datapath, priority, match, actions):
        """
        下发流表，datapath表示属于那个交换机，priority优先级
        """
        ofproto = datapath.ofproto #描述OpenFlow协议的属性
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]

        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)  #控制器向交换机发数据包

    def configure_path(self, path: list, src_mac: str, dst_mac: str):
        """
        将计算的最短路径配置到交换机上
        """
        # self.logger.info("Configuring related switches...")

        path_str = src_mac
        # 枚举路径上涉及的交换机和端口并打印
        for switch, in_port, out_port in path:
            datapath = self.find_datapath_by_id(int(switch))
            assert datapath
            parser = datapath.ofproto_parser
            match = parser.OFPMatch(
                in_port=in_port, eth_src=src_mac, eth_dst=dst_mac)
            actions = [parser.OFPActionOutput(out_port)]
            self.add_flow(datapath, 1, match, actions)
            path_str += "--{}-{}-{}".format(in_port, switch, out_port)

        path_str += "--" + dst_mac
        # self.logger.info("Path: {} has been configured.".format(path_str))

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, event):
        """
        异步处理 Packet-In 消息, 流表匹配失败
        """
        msg = event.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        dpid = datapath.id
        in_port = msg.match['in_port']

        # 从事件消息中解析数据帧
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        assert isinstance(eth, ethernet.ethernet)

        # 丢弃 LLDP 帧
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        # 丢弃 IPV6
        if eth.ethertype == ether_types.ETH_TYPE_IPV6:
            return

        # 获取源 MAC 地址和目的 MAC 地址
        dst_mac = eth.dst
        src_mac = eth.src

        # self.logger.info(
        #         "From {}:{} {} packet in ({} -> {})".format(dpid, in_port, eth.ethertype, src_mac, dst_mac))

        # host_mac_to 记录与主机(src_mac)直接相连的交换机的 ID 与端口
        if src_mac not in self.host_mac_to.keys():
            self.host_mac_to[src_mac] = (dpid, in_port)

        # 处理 ARP 数据包
        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            arp_pkt = pkt.get_protocol(arp.arp)
            assert isinstance(arp_pkt, arp.arp)
            if arp_pkt.opcode == arp.ARP_REQUEST:
                # 这里处理的是 ARP 请求消息，因为 ARP 回复时 src 和 dst 必定已经加入拓扑
                if (datapath.id, arp_pkt.src_mac, arp_pkt.dst_ip) in self.arp_history and self.arp_history[(datapath.id, arp_pkt.src_mac, arp_pkt.dst_ip)] != in_port:
                    # 打破 ARP 循环，避免引发 ARP 风暴
                    return
                else:
                    # 记录 ARP request 历史信息
                    self.arp_history[(
                        datapath.id, arp_pkt.src_mac, arp_pkt.dst_ip)] = in_port

        # 检测 host_mac_to，判断目的主机的 MAC 是否已经进入拓扑
        if dst_mac in self.host_mac_to.keys():

            # 找到和源主机直接相连的交换机
            src_switch = self.host_mac_to[src_mac][0]

            # 找到和源主机直接相连的端口
            first_port = self.host_mac_to[src_mac][1]

            # 找到和目的主机直接相连的交换机
            dst_switch = self.host_mac_to[dst_mac][0]

            # 找到与目的主机直接相连的交换机的端口
            final_port = self.host_mac_to[dst_mac][1]

            # 计算路径
            path = self.topo.dijkstra(
                src_switch, dst_switch, first_port, final_port, self.total_step_count)
            assert len(path) > 0

            # 配置路径上的交换机
            self.configure_path(path, src_mac, dst_mac)

            out_port = None
            # 设置 Packet-Out 为路径上当前交换机应该转发到的端口，避免丢包
            for switch, _, op in path:
                if switch == dpid:
                    out_port = op
            assert out_port
            # self.logger.info(
            #     "From {}:{}, a {} packet in ({} -> {}), send to {}".format(dpid, in_port, eth.ethertype, src_mac, dst_mac, out_port))

        else:
            # 目的 MAC 地址尚未收入拓扑或本身就是广播，设置为泛洪
            out_port = ofproto.OFPP_FLOOD
            # self.logger.info(
            #     "Unknown/Broadcast MAC address {}, flooding...".format(dst_mac))

        # 发送 Packet-Out，避免丢包
        actions = [parser.OFPActionOutput(out_port)]
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=msg.data)
        datapath.send_msg(out)

   # 拓扑发现
    @set_ev_cls(event.EventSwitchLeave)
    def switch_leave_handler(self, event):
        self.logger.info("A SwitchLeave..")
    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, event):
        """
        处理交换机进入消息，依赖 LLDP，用于发现拓扑
        """
        #print(event)
        self.logger.info("A SwitchEnterEvent...")
        self.scount += 1
        if self.scount != self.ACTIVE_NODES:
            return

        # 重新发现拓扑时清除历史数据
        self.topo = Topo()
        self.topo.clear()
        plt.clf()

        # 保存交换机信息
        all_switches = copy.copy(get_switch(self))
        self.datapaths = [s.dp for s in all_switches]
        self.topo.add_nodes_from(
            [(s.dp.id, {"ports": [p.port_no for p in s.ports]}) for s in all_switches])
        self.topo.set = UnionFindSet(len(all_switches))

        self.logger.info("Total {} switches:".format(len(self.topo.nodes)))
        self.logger.info(self.topo.nodes(data=True))

        # 保存链路信息，加入双向边（两条单向边），边权随机指定
        all_links = copy.copy(get_link(self))
        for link in all_links:
            u = link.src.dpid
            v = link.dst.dpid
            weight = self.topo.edges[v, u]['weight'] if self.topo.has_edge(
                v, u) else randint(1, 10)
            self.topo.add_edge(
                u, v, **{"src_port": link.src.port_no, "dst_port": link.dst.port_no, "weight": weight})
            if (u , v) not in self.env_edge and (v , u) not in self.env_edge:
                self.env_edge.append((u, v))


        self.logger.info("Total {} links: ".format(len(all_links)))
        self.logger.info(self.topo.edges(data=True))

        if len(all_links) != self.LINKS_NUM*2:
            os.system('mn -c')

        self.logger.info('Topology discovery succeeded.')

        plt.title('Discovered Topology')
        self.topo.pos = nx.spring_layout(self.topo)
        edge_labels = {e[0:2]: e[2]["weight"]
                       for e in list(self.topo.edges(data=True))}
        nx.draw(self.topo, pos=self.topo.pos,
                edgecolors="black", **self.topo.plot_options)
        nx.draw_networkx_edge_labels(
            self.topo, pos=self.topo.pos, edge_labels=edge_labels)
        plt.show()
        plt.savefig("Topo.png")
        plt.pause(1)
        self.topo_done = 1
        self.logger.info('Topology image saved.')


    def rl_state(self):
        self.env_T = np.asarray(self.tgen.gen_traffic())
        # print(self.env_T)
        return self.env_T[(self.env_T != -1)]

    def softmax(self, action):
        return np.exp(action) / np.sum(np.exp(action), axis=0)

    def rl_reward(self):

        return -(self.avg_sec*5+ self.sec_max + self.avg_jit*50)

    def upd_weights(self):
        # print("update weights.....")
        #print(self.topo.edges(data=True))   #(1, 8, {'src_port': 4, 'dst_port': 2, 'weight': 2}) 列表
        # print(self.topo.nodes(data=True))   #(1, {'ports': [4, 1, 2, 3]}) 列表

        weights = {}
        for e, w in zip(self.env_edge, self.env_W):
            weights[e] = w
        # 双向更新
        for (u, v), value in weights.items():
            #print(u, v, value)
            try:
                self.topo[u][v]['weight'] = value
                self.topo[v][u]['weight'] = value
            except KeyError:
                pass


        # plt.clf()
        #
        # self.topo.pos = nx.spring_layout(self.topo)
        # edge_labels = {e[0:2]: e[2]["weight"]
        #                for e in list(self.topo.edges(data=True))}
        # nx.draw(self.topo, pos=self.topo.pos,
        #         edgecolors="black", **self.topo.plot_options)
        # nx.draw_networkx_edge_labels(
        #     self.topo, pos=self.topo.pos, edge_labels=edge_labels)
        # plt.show()
        # plt.savefig("Topo"+str(self.total_step_count)+".png")
        # plt.pause(.5)

        #print(self.topo.edges(data=True))


    def pingall(self):
        # print("pingall...")
        ok = self.net.pingAll()
        time.sleep(.05)
        # print("pingdone! {}".format(ok))



    def step(self, action):

        # self.total_step_count += 1

        #print(action)

        self.env_W = np.asarray(action)
        #print(self.env_W)


        # update weights and flowtable
        self.upd_weights()
        for dp in self.datapaths:
            self.delete_flow(dp)
        # print("del dlows done !")
        self.pingall()

        # excute
        # print("excute mininet....")
        self.net.iperfMulti(hl=self.HostList, tm=self.env_T)
        # read output
        self.avg_cpup, self.avg_sec, self.avg_jit, self.loss_p, self.sec_max = read_iperf()
        # print("delay :{}".format(self.avg_sec))

        reward = self.rl_reward()

        new_state = self.rl_state()

        return new_state,  reward


    def play(self):
        # print(self.env_edge)
        # 相关常量的定义
        print("play")
        BUFFER_SIZE = 100  # 缓冲池的大小
        BATCH_SIZE = 16  # batch_size的大小
        GAMMA = 0.99  # 折扣系数
        TAU = 0.01  # target网络软更新的速度
        LR_A = 0.01  # Actor网络的学习率
        LR_C = 0.01  # Critic网络的学习率

        # 相关变量的定义

        action_dim = self.a_dim  # 动作的维度---w
        state_dim = self.s_dim  # 状态的维度---流量在链路上的分配信息

        episode = 50  # 迭代的次数
        step = 10  # 每次需要与环境交互的步数
        # total_step = 0  # 总共运行了多少步


        # 可视化集合定义
        reward_list = []  # 记录所有的rewards进行可视化展示
        loss_list = []  # 记录损失函数进行可视化展示
        avgsec_list = []
        maxsec_list = []
        jit_list = []
        cpu_list = []
        lossp_list = []

        step_list2 = []
        step_list = []  # 记录每一步的结果

        # 神经网络相关操作定义
        sess = tf.Session()
        from keras import backend as K
        K.set_session(sess)
        OU = OUNoise.OU()  # 引入噪声

        # 初始化四个个网络
        actor = Actor.ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LR_A)
        critic = Critic.CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LR_C)

        # 创建缓冲区
        buff = ReplayBuffer.ReplayBuffer(BUFFER_SIZE)

        # 加载训练数据
        print("Now we load the weight")
        try:
            actor.model.load_weights("src/actormodel.h5")
            critic.model.load_weights("src/criticmodel.h5")
            actor.target_model.load_weights("src/actormodel.h5")
            critic.target_model.load_weights("src/criticmodel.h5")
            print("Weight load successfully")
        except:
            print("Cannot find the weight")

        s_t = self.rl_state()
        # print(s_t)
        # print(s_t.shape)
        # 开始迭代
        print("Experiment Start.")
        for i in range(episode):
            # 输出当前信息
            if i == 5:
                self.tgen = Traffic(self.ACTIVE_NODES, self.LINKS_NUM, ratio=0.1)
            print("Episode : " + str(i) + " Replay Buffer " + str(buff.getCount()))
            total_reward =  0
            total_loss = 0


            # 开始执行step步
            for t in range(step):
                # print(total_step)
                loss = 0

                a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
                # print(a_t_original)  #[[........]]
                # print(a_t_original.shape) #(1,8)


                # 添加噪声和探索
                explore = 20
                if i <= explore:
                    a_t_original = OU.function(a_t_original, 1.0, (i / explore), 1.0 - (i / explore))
                else:
                    a_t_original = OU.function(a_t_original, 1.0, 0.8, 0.2)
                # print(a_t_original)
                a_t = np.fabs(a_t_original[0])
                # print(a_t.shape) #(8,)


                # 环境交互,
                s_t1, r_t = self.step(a_t)


                # 将该状态转移存储到缓冲池中
                buff.add(s_t, a_t, r_t, s_t1)

                # 选取batch_size个样本
                batch = buff.getBatch(BATCH_SIZE)
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                y_t = np.asarray([e[2] for e in batch])

                # 目标网络的预测q值---相当于y_label
                target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

                for k in range(len(batch)):
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

                # 训练网络
                states = states.reshape(len(states), -1)
                actions = actions.reshape(len(actions), -1)
                y_t = y_t.reshape(len(y_t), -1)
                loss = critic.model.train_on_batch([states, actions], y_t)  # 计算当前target网络和eval网络的损失值
                a_for_grad = actor.model.predict(states)  # 当前状态下eval网络产生的动作
                grads = critic.tarin(states, a_for_grad)  # 产生的梯度
                actor.train(states, grads)  # 更新eval网络
                actor.target_train()  # 更新target网络
                critic.target_train()  # 更新target网络

                total_reward += r_t
                total_loss += loss

                s_t = s_t1   # 转移到下一个状态

                print("step : {}  avg_sec : {} max_sec: {} jit: {}".format(self.total_step_count, self.avg_sec, self.sec_max, self.avg_jit))

                avgsec_list.append(self.avg_sec)
                maxsec_list.append(self.sec_max)
                jit_list.append(self.avg_jit)
                cpu_list.append(self.avg_cpup)
                lossp_list.append(self.loss_p)

                step_list2.append(self.total_step_count)

                self.total_step_count += 1


                # 绘图数据添加
                reward_list.append(r_t)
                loss_list.append(loss)

            # # 每隔100次保存一次参数
            # print("Now we save model")
            # actor.model.save_weights("src/actormodel.h5", overwrite=True)
            # # with open("src/actormodel.json", "w") as outfile:
            # #     json.dump(actor.model.to_json(), outfile)
            #
            # critic.model.save_weights("src/criticmodel.h5", overwrite=True)
            # with open("src/criticmodel.json", "w") as outfile:
            #     json.dump(critic.model.to_json(), outfile)

            # 打印相关信息
            print("")
            print("-" * 50)
            print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
            print("TOTAL LOSS @ " + str(i) + "-th Episode  : LOSS " + str(total_loss / step))
            print("Total Step: " + str(self.total_step_count))
            print("-" * 50)
            print("")

            # # 绘制图像，并保存
            # if i != 0 and i % 10 == 0:
            #     plt.cla()
            #     plt.plot(step_list, reward_list)
            #     print(step_list, reward_list)
            #     plt.xlabel("step")
            #     plt.ylabel("reward")
            #     plt.title("reward-step")
            #     img_name = "img/reward/" + str(i) + "-th Episode"
            #     plt.savefig(img_name)
            # if i != 0 and i % 10 == 0:
            #     plt.cla()  # 清除
            #     plt.plot(step_list, loss_list)
            #     plt.xlabel("step")
            #     plt.ylabel("loss")
            #     plt.title("loss-step")
            #     img_name = "img/loss/" + str(i) + "-th Episode"
            #     plt.savefig(img_name)

        # 训练完成之后最后保存信息
        print("Now we save model")
        actor.model.save_weights("src/actormodel.h5", overwrite=True)
        # # with open("src/actormodel.json", "w") as outfile:
        # #     json.dump(actor.model.to_json(), outfile)
        #
        critic.model.save_weights("src/criticmodel.h5", overwrite=True)
        # with open("src/criticmodel.json", "w") as outfile:
        #     json.dump(critic.model.to_json(), outfile)
        txt_save("sec_avg2.txt", avgsec_list)
        txt_save("sec_max2.txt", maxsec_list)
        txt_save("cpu2.txt", cpu_list)
        txt_save("jit2.txt", jit_list)
        txt_save("lossp2.txt", lossp_list)

        txt_save("reward2.txt", reward_list)
        txt_save("loss2.txt", loss_list)
        txt_save("step.txt", step_list2)

        print("Finish.")




if __name__ == '__main__':
    os.system('mn -c')
    os.system('ryu-manager /usr/local/SDNDDPG/DijkstraController.py --observe-links')


    # print([i for i in range(5001, 5001+(13*13))])
