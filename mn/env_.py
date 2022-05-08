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

    net = Mininet( topo=None,
                   build=False,
                   ipBase='10.0.0.0/8')

    info( '*** Adding controller\n' )
    c0=net.addController(name='c0',
                      controller=RemoteController,
                      protocol='tcp',
                      port=6653)

    info( '*** Add switches\n')
    s7 = net.addSwitch('s7', cls=OVSKernelSwitch, dpid='7')
    s11 = net.addSwitch('s11', cls=OVSKernelSwitch, dpid='11')
    s10 = net.addSwitch('s10', cls=OVSKernelSwitch, dpid='10')
    s6 = net.addSwitch('s6', cls=OVSKernelSwitch, dpid='6')
    s3 = net.addSwitch('s3', cls=OVSKernelSwitch, dpid='3')
    s12 = net.addSwitch('s12', cls=OVSKernelSwitch, dpid='12')
    s9 = net.addSwitch('s9', cls=OVSKernelSwitch, dpid='9')
    s5 = net.addSwitch('s5', cls=OVSKernelSwitch, dpid='5')
    s2 = net.addSwitch('s2', cls=OVSKernelSwitch, dpid='2')
    s8 = net.addSwitch('s8', cls=OVSKernelSwitch, dpid='8')
    s1 = net.addSwitch('s1', cls=OVSKernelSwitch, dpid='1')
    s4 = net.addSwitch('s4', cls=OVSKernelSwitch, dpid='4')
    s0 = net.addSwitch('s0', cls=OVSKernelSwitch, dpid='0')
    s13 = net.addSwitch('s13', cls=OVSKernelSwitch, dpid='13')

    info( '*** Add hosts\n')
    h0 = net.addHost('h0', cls=Host, ip='10.0.0.20')
    h11 = net.addHost('h11', cls=Host, ip='10.0.0.21')
    net.addLink(h0, s0)
    net.addLink(h11,s11)

    info( '*** Add links\n')
    s1s2 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s1, s2, cls=TCLink , **s1s2)
    s1s0 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s1, s0, cls=TCLink , **s1s0)
    s0s2 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s0, s2, cls=TCLink , **s0s2)
    s1s3 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s1, s3, cls=TCLink , **s1s3)
    s3s4 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s3, s4, cls=TCLink , **s3s4)
    s4s5 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s4, s5, cls=TCLink , **s4s5)
    s5s9 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s5, s9, cls=TCLink , **s5s9)
    s9s8 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s9, s8, cls=TCLink , **s9s8)
    s8s13 = {'bw':100, 'delay':'0','max_queue_size':500}
    net.addLink(s8, s13, cls=TCLink , **s8s13)
    s13s12 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s13, s12, cls=TCLink , **s13s12)
    s12s5 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s12, s5, cls=TCLink , **s12s5)
    s11s8 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s11, s8, cls=TCLink , **s11s8)
    s11s13 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s11, s13, cls=TCLink , **s11s13)
    s11s10 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s11, s10, cls=TCLink , **s11s10)
    s10s3 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s10, s3, cls=TCLink , **s10s3)
    s4s6 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s4, s6, cls=TCLink , **s4s6)
    s6s7 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s6, s7, cls=TCLink , **s6s7)
    s7s0 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s7, s0, cls=TCLink , **s7s0)
    s7s8 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s7, s8, cls=TCLink , **s7s8)
    s10s13 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s10, s13, cls=TCLink , **s10s13)
    s2s5 = {'bw':100,'delay':'0','max_queue_size':500}
    net.addLink(s2, s5, cls=TCLink , **s2s5)

    info( '*** Starting network\n')
    net.build()
    info( '*** Starting controllers\n')
    for controller in net.controllers:
        controller.start()

    info( '*** Starting switches\n')
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

    info( '*** Post configure switches and hosts\n')
    s7.cmd('ifconfig s7 10.0.0.8')
    s11.cmd('ifconfig s11 10.0.0.12')
    s10.cmd('ifconfig s10 10.0.0.11')
    s6.cmd('ifconfig s6 10.0.0.7')
    s3.cmd('ifconfig s3 10.0.0.4')
    s12.cmd('ifconfig s12 10.0.0.13')
    s9.cmd('ifconfig s9 10.0.0.10')
    s5.cmd('ifconfig s5 10.0.0.6')
    s2.cmd('ifconfig s2 10.0.0.3')
    s8.cmd('ifconfig s8 10.0.0.9')
    s1.cmd('ifconfig s1 10.0.0.2')
    s4.cmd('ifconfig s4 10.0.0.5')
    s0.cmd('ifconfig s0 10.0.0.1')
    s13.cmd('ifconfig s13 10.0.0.14')

    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    myNetwork()

