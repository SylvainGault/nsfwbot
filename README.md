# NSFW bot for IRC
This bot lurks in some channels, grab links and check if they contain NSFW links.

## Configuration
Configuration block at the top of `nsfwbot.py`.

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
    # NickServ password
    nspass = None
    # Logging level
    loglevel = logging.INFO
    # Max size of images to download
    max_download_size = 20 * 1024 * 1024
    
    # NickServ nickname
    ns_nick = "NickServ"
    # NickServ message
    nsmsgre = {}
    nsmsgre["nick_is_registered"] = r"Ce pseudo est enregistr� et prot�g�.*"
    nsmsgre["accepted_password"] = r"Mot de passe accept�.*"
    nsmsgre["ghosted"] = r"L'utilisateur .* a �t� d�connect�."
    ############################# End of configuration #############################

All the configuration options should be pretty explicit.

- The `hosts` list is tried in order.
- The `ports` list is tried in order. Favoring SSL connections if possible.
- If the first nick is already in use, the others are tried in order.
- All the channels are joined and monitored for links.
- If `nspass` is not `None`, it will be tried as `NickServ` password.
	- The nick should be already registered.
	- The bot try to use the `GHOST` command if it was unable to use its first nickname.
- `max_download_size` is the maximum size of image to download.
- Depending on the services default language, you might have to configure the regular expressions said by NickServ.


## Usage
    ./nsfwbot.py
