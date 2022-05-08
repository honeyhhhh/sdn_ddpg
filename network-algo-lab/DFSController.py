import copy
import warnings

import matplotlib.pyplot as plt
import networkx as nx
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (CONFIG_DISPATCHER, MAIN_DISPATCHER,
                                    set_ev_cls)
from ryu.lib.packet import arp, ether_types, ethernet, packet
from ryu.ofproto import ofproto_v1_3
from ryu.topology import event
from ryu.topology.api import get_link, get_switch


class Topo(nx.DiGraph):
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

    def dfs(self, u: int, dst: int, vis: 'dict[int, bool]', cur_path: 'list[int]', all_paths: 'list[list[int]]'):
        """
        DFS 递归遍历所有路径
        """
        if u == dst:
            all_paths.append(cur_path.copy())
            return
        for v in self.neighbors(u):
            if not vis[v]:
                vis[v] = True
                cur_path.append(v)
                self.dfs(v, dst, vis, cur_path, all_paths)
                cur_path.pop()  # 回溯
                vis[v] = False

    def search_path(self, src: int, dst: int, first_port: int, last_port: int):
        """
        获取所有路径并根据需求返回最长路或最短路
        """
        print("Calculating all the path of {}:{} -> {}:{}...".format(
            src, first_port, dst, last_port))

        vis = {}  # 标记路径上已经经过的交换机 type: dict[int, bool]
        for s in list(self.nodes):
            vis[s] = False  # 除 src 外，初始化为 False，即都未经过
        vis[src] = True

        cur_path = []
        cur_path.append(src)
        all_paths = []

        self.dfs(src, dst, vis, cur_path, all_paths)
        print("Found {} paths:".format(len(all_paths)))

        shortest_path = all_paths[0]
        longest_path = all_paths[0]
        for path in all_paths:
            if(len(path) > len(longest_path)):
                longest_path = path
            if(len(path) < len(shortest_path)):
                shortest_path = path
            print(path)

        print("Shortest path:", shortest_path,
              "length:", len(shortest_path), sep=' ')
        print("Longest path: ", longest_path,
              "length:", len(longest_path), sep=' ')

        # 选择最长路
        if src == dst:
            path = [src]
        else:
            path = longest_path

        # 绘图
        self.draw_path(path)

        # 生成路径：(src, in_port, out_port)->(s2, in_port, out_port)->...->(dst, in_port, out_port)
        ryu_path = []
        in_port = first_port
        for s1, s2 in zip(path[:-1], path[1:]):
            out_port = self.edges[s1, s2]["port"]
            ryu_path.append((s1, in_port, out_port))
            in_port = self.edges[s2, s1]["port"]
        ryu_path.append((dst, in_port, last_port))

        return ryu_path

    def draw_path(self, path: "list[int]"):
        edge_to_display = []

        for s1, s2 in zip(path[:-1], path[1:]):
            edge_to_display.append((s1, s2))
            edge_to_display.append((s2, s1))

        edge_colors = [
            "red" if e in edge_to_display else 'black' for e in list(self.edges)]
        node_edge_colors = [
            "red" if n in path else "black" for n in list(self.nodes())]

        plt.clf()
        plt.title("Longest Path from {} to {}".format(path[0], path[-1]))
        nx.draw(self, pos=self.pos, edge_color=edge_colors,
                edgecolors=node_edge_colors, **self.plot_options)
        plt.show()
        plt.pause(1)


class DFSController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DFSController, self).__init__(*args, **kwargs)
        self.datapaths = []  # 保存受控制的交换机
        self.host_mac_to = {}  # 记录与主机直接相连的交换机 ID 与端口
        self.topo = Topo()  # 控制器发现的拓扑
        self.arp_history = {}  # ARP 历史记录

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

    def configure_path(self, path: list, src_mac: str, dst_mac: str):
        """
        将计算的最短/最长路径配置到交换机上
        """
        self.logger.info("Configuring related switches...")

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
        self.logger.info("Path: {} has been configured.".format(path_str))

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
            path = self.topo.search_path(
                src_switch, dst_switch, first_port, final_port)
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
    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, event):
        """
        处理交换机进入消息，依赖 LLDP，用于发现拓扑
        """
        self.logger.info(
            "SwitchEnterEvent received, start topology discovery...")

        # 重新发现拓扑时清除历史数据
        self.topo.clear()
        plt.clf()

        # 保存交换机信息
        all_switches = copy.copy(get_switch(self))
        self.topo.add_nodes_from([s.dp.id for s in all_switches])
        self.datapaths = [s.dp for s in all_switches]
        self.logger.info("Total {} switches:".format(len(self.topo.nodes)))
        self.logger.info(self.topo.nodes)

        # 保存链路信息，加入双向边（两条单向边），port 为 src 的发送端口
        all_links = copy.copy(get_link(self))
        self.topo.add_edges_from(
            [(l.src.dpid, l.dst.dpid, {"port": l.src.port_no}) for l in all_links])
        self.topo.add_edges_from(
            [(l.dst.dpid, l.src.dpid, {"port": l.dst.port_no}) for l in all_links])
        self.logger.info("Total {} links: ".format(len(all_links)))
        self.logger.info(self.topo.edges())
        self.logger.info('Topology discovery succeeded.')

        plt.title('Discovered Topology')
        self.topo.pos = nx.spring_layout(self.topo)
        nx.draw(self.topo, pos=self.topo.pos,
                edgecolors="black", **self.topo.plot_options)
        plt.show()
        plt.savefig("Topo.png")
        plt.pause(1)
        self.logger.info('Topology image saved.')
