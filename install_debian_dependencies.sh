#!/bin/bash
sudo apt-get install -y python python-pip libllvm-3.7-ocaml-dev libz-dev libedit-dev
sudo ln -s /usr/bin/llvm-config-3.7 /usr/bin/llvm-config
sudo pip install -r requirements.txt
