#!/bin/env python3

import os
import sys


def main():
    if len(sys.argv) <= 1:
        exit("No base dir provided.")
    basedir = sys.argv[1]
    if not os.path.isdir(basedir):
        exit("Base dir not found at {}".format(basedir))
    continue_file = os.path.join(basedir, "continue")
    if os.path.isfile(continue_file):
        os.remove(continue_file)
        exit(0)
    else:
        exit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Exception during has_continue_file:", e)
