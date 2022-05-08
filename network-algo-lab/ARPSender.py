import argparse

import netifaces
from netifaces import AF_INET, AF_INET6, ifaddresses
from scapy.layers.l2 import ARP
from scapy.sendrecv import sr1


def get_mac_address(ifname):
    """
    Get MAC address of interface.
    """
    return netifaces.ifaddresses(ifname)[netifaces.AF_LINK][0]['addr']


def get_ip_address(ifname):
    """
    Get ipv4 address of interface.
    """
    return ifaddresses(ifname)[AF_INET][0]['addr']


def get_ipv6_address(ifname):
    """
    Get ipv6 address of interface.
    """
    return ifaddresses(ifname)[AF_INET6][0]['addr']


def arp_request(dst, ifname):
    """
    Send ARP request.
    """
    hwsrc = get_mac_address(ifname)
    psrc = get_ip_address(ifname)
    try:
        arp_pkt = sr1(ARP(op=1, hwsrc=hwsrc, psrc=psrc,
                      pdst=dst), timeout=5, verbose=False)
        assert arp_pkt
        return dst, arp_pkt.getlayer(ARP).fields['hwsrc']
    except:
        return dst, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Send Address Resolution Protocol(ARP) packet.")
    parser.add_argument("IFNAME", type=str, help="interface to send ARP")
    parser.add_argument("HOSTNAME", type=str, help="host to respond ARP")

    args = parser.parse_args()
    iface = args.IFNAME
    hostname = args.HOSTNAME
    print("Requesting MAC address of {} from {}...".format(hostname, iface))

    arp_result = arp_request(hostname, iface)
    if arp_result[1] != None:
        print("MAC address of {} is {}.".format(arp_result[0], arp_result[1]))
    else:
        print("Unable to receive response.")
