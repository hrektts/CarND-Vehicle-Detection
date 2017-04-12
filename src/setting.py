#!/usr/bin/env python
""" Settings
"""

import logging

logger = logging.getLogger("")
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(module)s: %(funcName)s: %(message)s'))
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
