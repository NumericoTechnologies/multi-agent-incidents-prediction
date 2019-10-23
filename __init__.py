""""""
import os
import sys

this_dir = os.path.abspath(os.path.dirname(__file__))
projects_dir = os.path.dirname(this_dir)
if projects_dir not in sys.path: sys.path.append(projects_dir)

from common_utils.argparser import Argparser, get_dropbox_path

_db_path = get_dropbox_path(
    '2019 - NEC RWS 8 Exploration Agent Based Incident Prediction',
    'python_out')
dropbox_join = lambda *x: os.path.join(_db_path, *x)

class MultiAgentArgparser(Argparser):
    """Specialized Argparser subclass for the needs of deployment project."""
    _db_path = _db_path
