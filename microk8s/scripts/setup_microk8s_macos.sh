# setup microk8s on macos (using VM)
brew install ubuntu/microk8s/microk8s
# setup cluster
# microk8s install --channel 1.24 -y
# microk8s status --wait-ready
# join a node as a worker - need to regenerate token on the main node
microk8s join 10.128.0.2:25000/0205c3d06c2bdcf88f469a19080aed9d/fd2aa8615b4c --worker