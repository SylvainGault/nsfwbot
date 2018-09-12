#!/usr/bin/env python3

import logging
import socket
import ssl
import itertools as it
import irc.bot
import irc.connection



################################ Configuration #################################
# List of addresses (for the same network) to try to connect to.
hosts = ["irc.server.net"]
# List of ports to try to connect to. SSL ports should be first.
ports = [6697, 6667]
# Nick to use.
nick = "nsfwbot"
# List of channels to join.
channels = ["#channel"]
# Real name
realname = "If you post NSFW images, I will tell!"
# Logging level
loglevel = logging.INFO
############################# End of configuration #############################



class ConnectionFactory(object):
    """This connection factory is automatically compatible with IPv6."""

    def __init__(self, bind=None, tryssl=True):
        self._bind = bind
        self._tryssl = tryssl

    def connect(self, addr):
        if self._tryssl:
            sock = socket.create_connection(addr, source_address=self._bind)
            try:
                return ssl.wrap_socket(sock)
            except ssl.SSLError:
                pass

            logging.debug("Couldn't connect to %s %d using SSL", *addr)
            sock.close()

        return socket.create_connection(addr, source_address=self._bind)

    __call__ = connect



class NSFWBot(irc.bot.SingleServerIRCBot):
    def on_ready(self, cnx, event):
        for c in channels:
            cnx.join(c)

    on_nomotd = on_ready
    on_endofmotd = on_ready



def main():
    logfmt = "%(asctime)s %(levelname)s: %(message)s"
    logging.basicConfig(format=logfmt, level=loglevel)

    specs = [irc.bot.ServerSpec(h, p) for h, p in it.product(hosts, ports)]
    bot = NSFWBot(specs, nick, realname, reconnection_interval=10, connect_factory=ConnectionFactory())
    bot.start()



if __name__ == '__main__':
    main()
