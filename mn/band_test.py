from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import CPULimitedHost
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel


# 一台控制器，一台交换机，六台主机，h1与h2之间10Mb/s， h3与h4之间30，h5与h6之间50
class SingleSwitchTopo(Topo):
    def __init__(self, **opt):
        Topo.__init__(self, **opt)
        switch = self.addSwitch('s1')
        h1 = self.addHost('h1', cpu=0.5)
        h2 = self.addHost('h2', cpu=0.5)

        self.addLink(h1, switch, bw=100, delay='5ms', loss=0, max_queue_size=1000, use_htb=True)
        self.addLink(h2, switch, bw=100, delay='5ms', loss=0, max_queue_size=1000, use_htb=True)



def perfTest():
    topo = SingleSwitchTopo()
    net = Mininet(topo=topo, host=CPULimitedHost, link=TCLink)

    net.start()

    dumpNodeConnections(net.hosts)

    net.pingAll()

    host_pool = net.get('h1', 'h2')
    res = net.iperf((host_pool[0], host_pool[1]))
    print(res)

    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    perfTest()
