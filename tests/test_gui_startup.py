"""Launch/reopen the correct workspace and avoid occupied loopback ports."""
from contextlib import contextmanager,redirect_stdout
import errno
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from race_gui.server import JobManager,LocalServer,main
from race_gui.startup import existing_gui_url,prepare_server


@contextmanager
def serving(server):
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:yield server
    finally:server.shutdown();server.server_close();thread.join(timeout=5)


class OtherApp(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        data=b'{"application":"another-app","protocol":1}'
        self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)


class GuiStartupTests(unittest.TestCase):
    def test_reuses_same_workspace_without_creating_or_stopping_server(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=JobManager(folder)
            with serving(LocalServer(('127.0.0.1',0),manager)) as running:
                with patch.object(manager,'stop_all') as stop,redirect_stdout(io.StringIO()):
                    with patch('race_gui.server.JobManager',return_value=manager),patch('race_gui.server.webbrowser.open') as browser:
                        main(['--port',str(running.server_port)])
                    browser.assert_called_once_with(f'http://127.0.0.1:{running.server_port}')
                    stop.assert_not_called()
                    with patch('race_gui.server.LocalServer',side_effect=AssertionError('duplicate server')) as factory:
                        server,url=prepare_server(manager,running.server_port,auto_port=True,server_factory=factory)
                    self.assertIsNone(server)
                    self.assertEqual(url,f'http://127.0.0.1:{running.server_port}')

    def test_other_app_is_left_running_and_free_port_is_selected(self):
        with tempfile.TemporaryDirectory() as folder,serving(ThreadingHTTPServer(('127.0.0.1',0),OtherApp)) as occupied:
            manager=JobManager(folder)
            server,url=prepare_server(manager,occupied.server_port,auto_port=True,port_attempts=1,server_factory=LocalServer)
            try:
                self.assertNotEqual(server.server_port,occupied.server_port)
                self.assertEqual(server.server_address[0],'127.0.0.1')
                self.assertEqual(url,f'http://127.0.0.1:{server.server_port}')
                self.assertIsNone(existing_gui_url(occupied.server_port,manager.root))
            finally:server.server_close()

    def test_never_reuses_race_gui_from_another_workspace(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            manager=JobManager(b)
            with serving(LocalServer(('127.0.0.1',0),JobManager(a))) as other:
                self.assertIsNone(existing_gui_url(other.server_port,manager.root))
                server,_=prepare_server(manager,other.server_port,auto_port=True,port_attempts=1,server_factory=LocalServer)
                try:self.assertNotEqual(server.server_port,other.server_port)
                finally:server.server_close()

    def test_explicit_fixed_port_reports_collision(self):
        with tempfile.TemporaryDirectory() as folder,serving(ThreadingHTTPServer(('127.0.0.1',0),OtherApp)) as occupied:
            with self.assertRaises(OSError) as caught:
                prepare_server(JobManager(folder),occupied.server_port,auto_port=False,server_factory=LocalServer)
            self.assertIn(caught.exception.errno,(errno.EADDRINUSE,10048))

    def test_unrelated_bind_errors_are_not_hidden(self):
        with tempfile.TemporaryDirectory() as folder,patch('race_gui.startup.existing_gui_url',return_value=None):
            with self.assertRaises(OSError) as caught:
                prepare_server(JobManager(folder),8765,auto_port=True,server_factory=lambda *_:(_ for _ in ()).throw(PermissionError(errno.EACCES,'denied')))
            self.assertEqual(caught.exception.errno,errno.EACCES)

    def test_port_zero_uses_os_assigned_port(self):
        with tempfile.TemporaryDirectory() as folder:
            server,url=prepare_server(JobManager(folder),0,auto_port=True,server_factory=LocalServer)
            try:
                self.assertGreater(server.server_port,0)
                self.assertEqual(url,f'http://127.0.0.1:{server.server_port}')
            finally:server.server_close()

    def test_windows_reserved_port_uses_a_free_port(self):
        from types import SimpleNamespace
        error=PermissionError(errno.EACCES,'Windows reserved port')
        error.winerror=10013
        assigned=SimpleNamespace(server_address=('127.0.0.1',54321))
        with tempfile.TemporaryDirectory() as folder,patch('race_gui.startup.existing_gui_url',return_value=None):
            factory=unittest.mock.Mock(side_effect=[error,assigned])
            server,url=prepare_server(JobManager(folder),8765,auto_port=True,port_attempts=1,server_factory=factory)
            self.assertIs(server,assigned)
            self.assertEqual(url,'http://127.0.0.1:54321')
            self.assertEqual(factory.call_args_list[1].args[0],('127.0.0.1',0))
