from mininet.net import Mininet
from mininet.node import RemoteController, Host, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.clean import cleanup
from subprocess import Popen
from multiprocessing import Process
import os
import time

def run_ryu():
    print('run ryu')
    proc = Popen("ryu-manager /usr/local/SDNDDPG/network-algo-lab/KruskalController.py --observe-links", shell=True)
    #/usr/local/ryu-master/ryu/app/rest_conf_switch.py /usr/local/ryu-master/ryu/app/rest_topology.py /usr/local/ryu-master/ryu/app/ofctl_rest.py


def myNetwork():
    os.system('mn -c')

    net = Mininet(topo=None,
                  build=False,
                  ipBase='10.0.0.0/8')

    info('*** Adding controller\n')
    c0 = net.addController(name='c0',
                           controller=RemoteController,
                           protocol='tcp',
                           port=6633)

    info('*** Add switches\n')
    s1 = net.addSwitch('s1', cls=OVSKernelSwitch, protocols=['OpenFlow13'])
    s2 = net.addSwitch('s2', cls=OVSKernelSwitch, protocols=['OpenFlow13'])
    s3 = net.addSwitch('s3', cls=OVSKernelSwitch, protocols=['OpenFlow13'])
    s4 = net.addSwitch('s4', cls=OVSKernelSwitch, protocols=['OpenFlow13'])
    s5 = net.addSwitch('s5', cls=OVSKernelSwitch, protocols=['OpenFlow13'])
    s6 = net.addSwitch('s6', cls=OVSKernelSwitch, protocols=['OpenFlow13'])
    s7 = net.addSwitch('s7', cls=OVSKernelSwitch, protocols=['OpenFlow13'])
    s8 = net.addSwitch('s8', cls=OVSKernelSwitch, protocols=['OpenFlow13'])
    s9 = net.addSwitch('s9', cls=OVSKernelSwitch, protocols=['OpenFlow13'])
    s10 = net.addSwitch('s10', cls=OVSKernelSwitch, protocols=['OpenFlow13'])
    s11 = net.addSwitch('s11', cls=OVSKernelSwitch, protocols=['OpenFlow13'])
    s12 = net.addSwitch('s12', cls=OVSKernelSwitch, protocols=['OpenFlow13'])
    s13 = net.addSwitch('s13', cls=OVSKernelSwitch, protocols=['OpenFlow13'])

    info('*** Add hosts\n')
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
    h9 = net.addHost('h9', cls=Host, ip='10.0.0.9',
                     defaultRoute=None, mac='00:00:00:00:00:09')
    h10 = net.addHost('h10', cls=Host, ip='10.0.0.10',
                      defaultRoute=None, mac='00:00:00:00:00:10')
    h11 = net.addHost('h11', cls=Host, ip='10.0.0.11',
                      defaultRoute=None, mac='00:00:00:00:00:11')
    h12 = net.addHost('h12', cls=Host, ip='10.0.0.12',
                      defaultRoute=None, mac='00:00:00:00:00:12')
    h13 = net.addHost('h13', cls=Host, ip='10.0.0.13',
                      defaultRoute=None, mac='00:00:00:00:00:13')

    info('*** Add links\n')
    # Every switch links one host:
    net.addLink(s1, h1)
    net.addLink(s2, h2)
    net.addLink(s3, h3)
    net.addLink(s4, h4)
    net.addLink(s5, h5)
    net.addLink(s6, h6)
    net.addLink(s7, h7)
    net.addLink(s8, h8)
    net.addLink(s9, h9)
    net.addLink(s10, h10)
    net.addLink(s11, h11)
    net.addLink(s12, h12)
    net.addLink(s13, h13)
    # Links between switches:
    net.addLink(s1, s3)
    net.addLink(s3, s2)
    net.addLink(s2, s6)
    net.addLink(s3, s6)
    net.addLink(s3, s4)
    net.addLink(s3, s7)
    net.addLink(s7, s5)
    net.addLink(s7, s10)
    net.addLink(s6, s8)
    net.addLink(s8, s13)
    net.addLink(s8, s9)
    net.addLink(s9, s10)
    net.addLink(s10, s12)
    net.addLink(s6, s9)
    net.addLink(s10, s11)
    net.addLink(s9, s11)
    net.addLink(s11, s13)

    info('*** Starting network\n')
    net.build()
    info('*** Starting controllers\n')
    for controller in net.controllers:
        controller.start()

    info('*** Starting switches\n')
    s1.start([c0])
    s2.start([c0])
    s3.start([c0])
    s4.start([c0])
    s5.start([c0])
    s6.start([c0])
    s7.start([c0])
    s8.start([c0])
    s9.start([c0])
    s10.start([c0])
    s11.start([c0])
    s12.start([c0])
    s13.start([c0])

    info('*** Post configure switches and hosts\n')
    time.sleep(5)
    p = Process(target=run_ryu())
    p.start()
    p.join()


    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    myNetwork()
    cleanup()
