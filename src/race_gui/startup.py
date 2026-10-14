"""Find this workspace's GUI or bind a free local port without stopping apps."""
import errno
import json
import os
from urllib.error import HTTPError,URLError
from urllib.request import ProxyHandler,build_opener


def existing_gui_url(port,workspace):
    url=f'http://127.0.0.1:{port}'
    try:
        # Local discovery must not go through a configured HTTP proxy.
        with build_opener(ProxyHandler({})).open(url+'/api/identity',timeout=.2) as response:
            data=json.loads(response.read(4097).decode('utf-8'))
        if (data.get('application')=='race-rl-gui' and data.get('protocol')==2 and
                isinstance(data.get('workspace'),str) and
                os.path.normcase(data['workspace'])==os.path.normcase(str(workspace.resolve()))):
            return url
    except (HTTPError,URLError,OSError,ValueError,AttributeError):pass
    return None


def prepare_server(manager,port,*,auto_port,server_factory,port_attempts=20):
    if port==0:
        server=server_factory(('127.0.0.1',0),manager)
        return server,f'http://127.0.0.1:{server.server_address[1]}'
    ports=range(port,min(65536,port+(port_attempts if auto_port else 1)))
    # Repeated launch should reopen an existing GUI even if a lower port is free.
    for candidate in ports:
        url=existing_gui_url(candidate,manager.root)
        if url:return None,url
    for candidate in ports:
        try:
            server=server_factory(('127.0.0.1',candidate),manager)
            return server,f'http://127.0.0.1:{server.server_address[1]}'
        except OSError as exc:
            # Winsock also reports access denied for reserved/exclusive ports.
            unavailable=(exc.errno in (errno.EADDRINUSE,10048,10013) or
                getattr(exc,'winerror',None) in (10048,10013))
            if not auto_port or not unavailable:raise
            # Another launcher may have started the same GUI during discovery.
            url=existing_gui_url(candidate,manager.root)
            if url:return None,url
    # If all nearby ports are used, let the OS choose an available one.
    server=server_factory(('127.0.0.1',0),manager)
    return server,f'http://127.0.0.1:{server.server_address[1]}'
