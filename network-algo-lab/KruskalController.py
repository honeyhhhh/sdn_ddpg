import copy
import warnings
from random import randint

import matplotlib.pyplot as plt
import networkx as nx
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (CONFIG_DISPATCHER, MAIN_DISPATCHER,
                                    set_ev_cls)
from ryu.lib.packet import ether_types, ethernet, packet
from ryu.ofproto import ofproto_v1_3
from ryu.topology import event
from ryu.topology.api import get_link, get_switch
import time


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


class Topo(nx.DiGraph):
    """
    网络拓扑，继承自 networkx.DiGraph
    """

    def __init__(self):
        super().__init__()
        warnings.filterwarnings("ignore", category=UserWarning)
        self.set = UnionFindSet(0)  # 默认并查集，稍后会在拓扑发现时替换
        self.MST_exist = False
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

    def Kruskal(self) -> list:
        """
        Kruskal 算法计算最小生成树
        """
        tree_edges = []  # 保存最小生成树中的边
        for e in sorted(list(self.edges(data=True)), key=lambda e: e[2]["weight"]):
            # 按边权从小到大遍历所有边，相连两点不在并查集内就加入 MST
            if self.set.find(e[0]) != self.set.find(e[1]):
                self.set.union(e[0], e[1])
                # 将边添加进 MST
                tree_edges.append((e[0], e[1]))

        # 绘制最小生成树
        self.draw_tree(tree_edges)

        return tree_edges

    def draw_tree(self, tree: "list"):
        """
        绘制树
        """
        plt.clf()
        plt.title("Minimum Spanning Tree")

        edge_colors = []
        edge_labels = {}
        for e in list(self.edges(data=True)):
            edge_colors.append("red" if (e[0], e[1]) in tree or (
                e[1], e[0]) in tree else 'black')
            edge_labels[e[0:2]] = e[2]["weight"]

        node_edge_colors = ["red" if len(
            tree) else "black"] * len(list(self.nodes()))

        nx.draw(self, pos=self.pos, edge_color=edge_colors,
                edgecolors=node_edge_colors, **self.plot_options)

        nx.draw_networkx_edge_labels(
            self, pos=self.pos, edge_labels=edge_labels)
        plt.show()
        plt.pause(1)


class KruskalController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(KruskalController, self).__init__(*args, **kwargs)
        self.datapaths = []  # 保存受控制的交换机
        self.mac_to_port = {}  # MAC 表
        self.topo = Topo()  # 控制器发现的拓扑
        self.scount = 0   # 记录交换机进入数量
        self.myscount = 14

    def find_datapath_by_id(self, dpid: int):
        for datapath in self.datapaths:
            if datapath.id == dpid:
                return datapath
        return None

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        """
        下发流表
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    def send_port_mod(self, datapath, port_no, opt):
        """
        下发 port_mod，更改端口状态
        """
        ofp = datapath.ofproto
        ofp_parser = datapath.ofproto_parser

        hw_addr = None
        switch = get_switch(self, dpid=datapath.id)[0]
        for p in switch.ports:
            if p.port_no == port_no:
                hw_addr = p.hw_addr
                break
        assert hw_addr

        config = opt
        mask_all = (ofp.OFPPC_PORT_DOWN | ofp.OFPPC_NO_RECV |
                    ofp.OFPPC_NO_FWD | ofp.OFPPC_NO_PACKET_IN)

        req = ofp_parser.OFPPortMod(datapath=datapath, port_no=port_no, hw_addr=hw_addr, config=config,
                                    mask=mask_all)
        datapath.send_msg(req)

    def block_links(self, excepts: "list[tuple]"):
        """
        屏蔽除 except 外的链路
        """
        for e in list(self.topo.edges(data=True)):
            if (e[0], e[1]) in excepts or (e[1], e[0]) in excepts:
                self.logger.info("{} -- {} Linked!".format(e[0], e[1]))
                continue

            src_dp = self.find_datapath_by_id(e[0])
            assert src_dp
            self.send_port_mod(
                src_dp, e[2]["src_port"], src_dp.ofproto.OFPPC_PORT_DOWN)

            dst_dp = self.find_datapath_by_id(e[1])
            assert dst_dp
            self.send_port_mod(
                dst_dp, e[2]["dst_port"], dst_dp.ofproto.OFPPC_PORT_DOWN)

            self.logger.info("{} -- {} Blocked!".format(e[0], e[1]))

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

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, event):
        """
        处理 Packet-In 消息
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

        # 丢弃 NDP 帧
        if eth.ethertype == ether_types.ETH_TYPE_IPV6:
            return

        # 获取源 MAC 地址和目的 MAC 地址
        dst_mac = eth.dst
        src_mac = eth.src

        # 学习 MAC 地址
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src_mac] = in_port

        # self.logger.info(
        #         "From {}:{} {} packet in ({} -> {})".format(dpid, in_port, eth.ethertype, src_mac, dst_mac))

        if dst_mac in self.mac_to_port[dpid]:
            # 目的 MAC 地址已经在 MAC 表中
            out_port = self.mac_to_port[dpid][dst_mac]
        elif self.topo.MST_exist:
            # 目的 MAC 地址尚未收入 MAC 表或本身就是广播，设置为泛洪
            out_port = ofproto.OFPP_FLOOD
            # self.logger.info(
            #     "Unknown/Broadcast MAC address {}, flooding...".format(dst_mac))
        else:
            # MST 不存在时，需要计算 MST
            self.logger.info(
                "MST is not exist, generating...".format(dst_mac))

            tree_edges = self.topo.Kruskal()

            # 屏蔽 MST 之外的链路
            self.block_links(tree_edges)

            # MST 标志置为 True
            self.topo.MST_exist = True
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # 下发流表，避免下次 Packet-In
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(
                in_port=in_port, eth_dst=dst_mac, eth_src=src_mac)
            # 验证是否存在有效的缓冲区ID，避免同时发送flow_mod和packet_out
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                return
            else:
                self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        # 发送 Packet-Out
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
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
        print(event)
        self.logger.info("A SwitchEnterEvent...")
        self.scount += 1
        if self.scount != self.myscount:
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

        self.logger.info("Total {} links: ".format(len(all_links)))
        self.logger.info(self.topo.edges(data=True))


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

        self.logger.info('Topology image saved.')
