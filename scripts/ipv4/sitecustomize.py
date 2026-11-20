# Loaded by the Kaggle CLI processes kg.py starts (10 Nov 2026): when this machine's IPv6 route to Google storage is dead
# (connections stay in SYN_SENT), uploads hang, because the CLI tries IPv6 first without a connect timeout. Only processes
# started with this folder on PYTHONPATH resolve names to IPv4 addresses.
import socket

_getaddrinfo = socket.getaddrinfo


def _ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return _getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _ipv4
