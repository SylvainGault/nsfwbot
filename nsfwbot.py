#!/usr/bin/env python3

import logging
import traceback as tb
import queue
import re
import socket
import ssl
import itertools as it
import humanize
import irc.bot
import irc.connection
import asyncworkflow



################################ Configuration #################################
# List of addresses (for the same network) to try to connect to.
hosts = ["irc.server.net"]
# List of ports to try to connect to. SSL ports should be first.
ports = [6697, 6667]
# List of nicks to try to use.
nicks = ["nsfwbot"]
# List of channels to join.
channels = ["#channel"]
# Real name
realname = "If you post NSFW images, I will tell!"
# Logging level
loglevel = logging.INFO
# Max size of images to download
max_download_size = 20 * 1024 * 1024
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
    nickre = re.compile(nicks[0] + r'(\d+)')
    urlre = re.compile(r'https?://[^\s<>()]*')

    def __init__(self, *args, **kwargs):
        super(NSFWBot, self).__init__(*args, **kwargs)
        self.ready = False
        self._workflow = asyncworkflow.AsyncWorkflow(maxdlsize=max_download_size)
        self._ircloopq = queue.Queue()
        self.ircobj.execute_every(0.2, self._runqueue)

    def _runqueue(self):
        while True:
            try:
                fun, args = self._ircloopq.get_nowait()
            except queue.Empty:
                break

            fun(*args)

    def on_ready(self, cnx, event):
        self.ready = True
        for c in channels:
            cnx.join(c)

    on_nomotd = on_ready
    on_endofmotd = on_ready

    def on_disconnect(self, cnx, event):
        self.ready = False

    def choose_initial_nick(self, cnx, nick, msg):
        # nick is the last one we tried. So try the next one in the list
        try:
            idx = nicks.index(nick)
        except ValueError:
            idx = None

        if idx is not None and idx < len(nicks) - 1:
            cnx.nick(nicks[idx + 1])
            return

        # Generate a new nickname based on nicks[0]
        matches = self.nickre.match(nick)
        if not matches:
            newnum = 0
        else:
            newnum = int(matches.group(1)) + 1

        newnick = nicks[0] + str(newnum)
        while newnick in nicks:
            newnum += 1
            newnick = nicks[0] + str(newnum)

        cnx.nick(newnick)

    def on_nicknameinuse(self, cnx, event):
        if not self.ready:
            self.choose_initial_nick(cnx, *event.arguments)

    def on_erroneusnickname(self, cnx, event):
        if not self.ready:
            self.choose_initial_nick(cnx, *event.arguments)

    def on_pubmsg(self, cnx, event):
        chan = event.target
        if chan not in channels:
            return

        msg = event.arguments[0]
        urls = self.urlre.findall(msg)

        for url in urls:
            def cb(*args):
                logging.debug("Scheduling _scorecb with arguments %s", args)
                self._ircloopq.put((self._scorecb, (cnx, chan, url, *args)))
            def errcb(*args):
                logging.debug("Scheduling _errcb with arguments %s", args)
                self._ircloopq.put((self._errcb, (cnx, chan, url, *args)))

            self._workflow.addurl(url, cb, errcb)

    def _scorecb(self, cnx, chan, url, totalsize, istrunc, scores):
        msg = "<%s> " % url
        if totalsize:
            msg += "(%s) " % humanize.naturalsize(totalsize, binary=True)

        if len(scores) == 0:
            msg += "Can't read as an image."
        else:
            score = scores[0] * 100
            msg += "NSFW score: %.2f%%. " % score

            if score < 10:
                msg += "Certainly safe."
            elif 10 <= score < 50:
                msg += "Probably safe."
            elif 50 <= score < 90:
                msg += "Probably sexy."
            else:
                msg += "Most likely porn."

        if istrunc:
            mds = humanize.naturalsize(max_download_size, binary=True)
            msg += " The %s download limit was reached." % mds

        cnx.privmsg(chan, msg)

    def _errcb(self, cnx, chan, url, exc):
        cnx.privmsg(chan, "Error while processing <%s>" % url)

        lines = tb.format_exception_only(type(exc), exc)
        for l in lines:
            logging.info(l)
            cnx.privmsg(chan, l.rstrip())



def main():
    logfmt = "%(asctime)s %(levelname)s: %(message)s"
    logging.basicConfig(format=logfmt, level=loglevel)

    specs = [irc.bot.ServerSpec(h, p) for h, p in it.product(hosts, ports)]
    bot = NSFWBot(specs, nicks[0], realname, reconnection_interval=10, connect_factory=ConnectionFactory())

    try:
        bot.start()
    except KeyboardInterrupt:
        bot.die()



if __name__ == '__main__':
    main()
