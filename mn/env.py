#!/usr/bin/env python

from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSController
from mininet.node import CPULimitedHost, Host, Node
from mininet.node import OVSKernelSwitch, UserSwitch
from mininet.node import IVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink, Intf
from subprocess import call



def myNetwork():

    HostIPList = []
    HostList = []

    net = Mininet( topo=None,
                   build=False,
                   ipBase='10.0.0.0/8',
                   autoSetMacs=True)

    info( '*** Adding controller\n' )
    c0=net.addController(name='c0',
                      controller=RemoteController,
                      protocol='tcp',
                      port=6653)  #RYU 6653 ryu-manager rest_conf_switch.py rest_topology.py rest_qos.py ofctl_rest.py --observe-links


    info( '*** Add switches\n')
    s7 = net.addSwitch('s7', cls=OVSKernelSwitch)
    s11 = net.addSwitch('s11', cls=OVSKernelSwitch)
    s10 = net.addSwitch('s10', cls=OVSKernelSwitch)
    s6 = net.addSwitch('s6', cls=OVSKernelSwitch)
    s3 = net.addSwitch('s3', cls=OVSKernelSwitch)
    s12 = net.addSwitch('s12', cls=OVSKernelSwitch)
    s9 = net.addSwitch('s9', cls=OVSKernelSwitch)
    s5 = net.addSwitch('s5', cls=OVSKernelSwitch)
    s2 = net.addSwitch('s2', cls=OVSKernelSwitch)
    s8 = net.addSwitch('s8', cls=OVSKernelSwitch)
    s1 = net.addSwitch('s1', cls=OVSKernelSwitch)
    s4 = net.addSwitch('s4', cls=OVSKernelSwitch)
    s0 = net.addSwitch('s0', cls=OVSKernelSwitch)
    s13 = net.addSwitch('s13', cls=OVSKernelSwitch)



    info( '*** Add hosts\n')
    sh = {'bw':100,'delay':'0ms', 'max_queue_size':500}





    info( '*** Add links\n')
    ss = {'bw':100,'delay':'0ms', 'max_queue_size':500}
    net.addLink(s1, s2, cls=TCLink , **ss)
    net.addLink(s1, s0, cls=TCLink , **ss)
    net.addLink(s0, s2, cls=TCLink , **ss)
    net.addLink(s1, s3, cls=TCLink , **ss)
    net.addLink(s3, s4, cls=TCLink , **ss)
    net.addLink(s4, s5, cls=TCLink , **ss)
    net.addLink(s5, s9, cls=TCLink , **ss)
    net.addLink(s9, s8, cls=TCLink , **ss)
    net.addLink(s8, s13, cls=TCLink , **ss)
    net.addLink(s13, s12, cls=TCLink , **ss)
    net.addLink(s12, s5, cls=TCLink , **ss)
    net.addLink(s11, s8, cls=TCLink , **ss)
    net.addLink(s11, s13, cls=TCLink , **ss)
    net.addLink(s11, s10, cls=TCLink , **ss)
    net.addLink(s10, s3, cls=TCLink , **ss)
    net.addLink(s4, s6, cls=TCLink , **ss)
    net.addLink(s6, s7, cls=TCLink , **ss)
    net.addLink(s7, s0, cls=TCLink , **ss)
    net.addLink(s7, s8, cls=TCLink , **ss)
    net.addLink(s10, s13, cls=TCLink , **ss)
    net.addLink(s2, s5, cls=TCLink , **ss)

    info( '*** Starting network\n')
    net.build()
    info( '*** Starting controllers\n')  # 开启控制器
    for controller in net.controllers:
        controller.start()

    info( '*** Starting switches\n')  # 交换机发起和控制器的连接请求
    net.get('s7').start([c0])
    net.get('s11').start([c0])
    net.get('s10').start([c0])
    net.get('s6').start([c0])
    net.get('s3').start([c0])
    net.get('s12').start([c0])
    net.get('s9').start([c0])
    net.get('s5').start([c0])
    net.get('s2').start([c0])
    net.get('s8').start([c0])
    net.get('s1').start([c0])
    net.get('s4').start([c0])
    net.get('s0').start([c0])
    net.get('s13').start([c0])

    for hoststr in net.keys():
        if "h" in hoststr:
            host = net.get(hoststr)
            HostList.append(host)
            HostIPList.append(host.IP())

     #print(HostIPList)

    info( '*** Post configure switches and hosts\n')

    CLI(net)  # 需要quit 或者 mn -c 清除拓扑
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    myNetwork()

