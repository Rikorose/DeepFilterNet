#!/usr/bin/env python3

import os
import sys
from configparser import ConfigParser

USAGE = f"Usage: {sys.argv[0]} <file> <section> <parameter>"
assert len(sys.argv) == 4, USAGE
assert os.path.isfile(sys.argv[1]), USAGE

parser = ConfigParser()
parser.read_file(open(sys.argv[1]))
assert parser.has_section(sys.argv[2])
print(parser.get(sys.argv[2], sys.argv[3]))
