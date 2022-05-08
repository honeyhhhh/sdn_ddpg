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

    info( '*** Add hosts\n')

    info( '*** Add links\n')
    s1s4 = {'bw':10,'delay':'10ms'}
    net.addLink(s1, s4, cls=TCLink , **s1s4)
    s1s2 = {'bw':10,'delay':'10ms'}
    net.addLink(s1, s2, cls=TCLink , **s1s2)
    s2s3 = {'bw':10,'delay':'10ms'}
    net.addLink(s2, s3, cls=TCLink , **s2s3)
    s3s7 = {'bw':10,'delay':'10ms'}
    net.addLink(s3, s7, cls=TCLink , **s3s7)
    s7s12 = {'bw':10,'delay':'10ms'}
    net.addLink(s7, s12, cls=TCLink , **s7s12)
    s4s6 = {'bw':10,'delay':'10ms'}
    net.addLink(s4, s6, cls=TCLink , **s4s6)
    s5s6 = {'bw':10,'delay':'10ms'}
    net.addLink(s5, s6, cls=TCLink , **s5s6)
    s6s7 = {'bw':10,'delay':'10ms'}
    net.addLink(s6, s7, cls=TCLink , **s6s7)
    s5s8 = {'bw':10,'delay':'10ms'}
    net.addLink(s5, s8, cls=TCLink , **s5s8)
    s8s9 = {'bw':10,'delay':'10ms'}
    net.addLink(s8, s9, cls=TCLink , **s8s9)
    s9s10 = {'bw':10,'delay':'10ms'}
    net.addLink(s9, s10, cls=TCLink , **s9s10)
    s10s12 = {'bw':10,'delay':'10ms'}
    net.addLink(s10, s12, cls=TCLink , **s10s12)
    s11s12 = {'bw':10,'delay':'10ms'}
    net.addLink(s11, s12, cls=TCLink , **s11s12)
    s9s11 = {'bw':10,'delay':'10ms'}
    net.addLink(s9, s11, cls=TCLink , **s9s11)
    s2s5 = {'bw':10,'delay':'10ms'}
    net.addLink(s2, s5, cls=TCLink , **s2s5)

    info( '*** Starting network\n')
    net.build()
    info( '*** Starting controllers\n')
    for controller in net.controllers:
        controller.start()

    info( '*** Starting switches\n')
    net.get('s7').start([])
    net.get('s11').start([])
    net.get('s10').start([])
    net.get('s6').start([])
    net.get('s3').start([])
    net.get('s12').start([])
    net.get('s9').start([])
    net.get('s5').start([])
    net.get('s2').start([])
    net.get('s8').start([])
    net.get('s1').start([])
    net.get('s4').start([])

    info( '*** Post configure switches and hosts\n')

    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    myNetwork()

