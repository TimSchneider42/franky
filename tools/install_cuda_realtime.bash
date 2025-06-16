#!/bin/bash
# Author: Tim Schneider <tim@robot-learning.de>

# Installs CUDA drivers on a fresh Ubuntu with realtime kernel. If other CUDA drivers are already installed, please remove them prior
# to running this script, e.g. with
# sudo apt purge 'nvidia-*'
# sudo apt autoremove
# sudo apt autoclean

# Install cuda drivers following https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu
# Install linux headers for your kernel version
sudo apt-get -y install linux-headers-$(uname -r) &&

# Delete old Nvidia key
sudo apt-key del 7fa2af80;

# Install new Nvidia key
ubuntu_version=$(lsb_release -r | awk '{print $2}' | tr -d '.')
arch=$(uname -m)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${ubuntu_version}/$arch/cuda-keyring_1.1-1_all.deb -P /tmp/ &&
if sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb; then
	rm /tmp/cuda-keyring_1.1-1_all.deb
else
	rm /tmp/cuda-keyring_1.1-1_all.deb
	exit 1
fi

# Set IGNORE_PREEMPT_RT_PRESENCE persistently
grep -qxF 'IGNORE_PREEMPT_RT_PRESENCE=1' /etc/environment || echo 'IGNORE_PREEMPT_RT_PRESENCE=1' | sudo tee -a /etc/environment > /dev/null &&

# Install cuda drivers
sudo apt-get update &&

# Here we deviate from the online guide by telling the installer to ignore the presence of the realtime kernel
sudo IGNORE_PREEMPT_RT_PRESENCE=1 apt-get -y install cuda-drivers &&

echo "Please reboot the system now."
