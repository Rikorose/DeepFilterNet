#!/bin/bash

grep version DeepFilterNet/pyproject.toml -m 1 | cut -d '=' -f2 | tr -d ' "'
