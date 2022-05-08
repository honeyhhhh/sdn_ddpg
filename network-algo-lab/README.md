# Network_AlgoLab

## Description
This repository holds the lab code of UESTC's Network Algorithm Basis course.

For environment setup and design explanation, visit [Nativus' Space](https://naiv.fun/Dev/41.html).

## Installation

1. Install Mininet.

2. Install RYU and other depend python packages. Or simply execute:

   ```sh
   sudo pip3 install -r requirements.txt
   ```

3. Clone this repo or download the source code to anywhere.

## Instructions

1. Launch Mininet with given topo:
    ```sh
    $ sudo python3 topo.py
    ```
    
2. Run xxxController.py with ryu-manager, e.g.,
    ```sh
    $ sudo ryu-manager DFSController.py  --observe-links 
    ```
    
3. Wait till the topology discovery finished, then execute `pingall` to test connectivity or any other commands in the Mininet CLI.

### Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request
