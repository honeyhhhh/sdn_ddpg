from mininet.net import Mininet
from mininet.topo import LinearTopo
from mininet.topo import SingleSwitchTopo
from mininet.topolib import TreeTopo

from mininet.util import dumpNodeConnections


Linear4 = LinearTopo(k=4)  # 4个交换机，分别挂一个主机
Single3 = SingleSwitchTopo(k=3) # 1个交换机，挂3个主机
Tree22 = TreeTopo(depth=2, fanout=2)


# net = Mininet(topo=Tree22)

net = Mininet()
c0 = net.addController()
s0 = net.addSwitch('s0')
h0 = net.addHost('h0')
h1 = net.addHost('h1', cpu=0.5)
h2 = net.addHost('h2', cpu=0.5)

net.addLink(s0, h0, bw=10, delay='5ms', max_queue_size=1000, loss=10, use_htb=True)  # 带宽bw，延迟delay， 最大队列的大小max_queue_size，损耗率loss

net.addLink(h1, s0)
net.addLink(h2, s0)
h0.setIP('192.168.1.3', 24)
h1.setIP('192.168.1.1', 24)  # 网络前缀
h2.setIP('192.168.1.2', 24)



net.start()
dumpNodeConnections(net.hosts)
net.pingAll()
net.stop()

