# install & create new user group
sudo snap install microk8s --classic --channel=1.24
sudo usermod -a -G microk8s $USER
sudo chown -f -R $USER ~/.kube
newgrp microk8s
# check status to verify that microk8s is running
microk8s status --wait-ready
# add some plugins
microk8s enable dns storage ingress prometheus dashboard
# should have open the right port from gcp control plane
sudo ufw enable

# don't drop traffic
sudo iptables -P FORWARD ACCEPT
sudo ufw default allow routed