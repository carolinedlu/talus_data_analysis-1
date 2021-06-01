#!/bin/bash -ex
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
sudo yum update -y;
sudo yum install git tmux -y;
sudo amazon-linux-extras install docker -y;
sudo service docker start;
sudo usermod -a -G docker ec2-user;
sudo docker pull 622568582929.dkr.ecr.us-west-2.amazonaws.com/talus_streamlit:latest;