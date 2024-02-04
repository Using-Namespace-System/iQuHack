FROM mcr.microsoft.com/vscode/devcontainers/anaconda:latest as vscode

FROM nvcr.io/nvidia/cuquantum-appliance:23.10 as base

USER root

ARG UNAME=cuquantum

RUN export UID=$(id -u ${UNAME}) \
    && export GID=$(id -g ${UNAME}) \
    && echo $UNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$UNAME \
    && chmod 0440 /etc/sudoers.d/$UNAME \
    && passwd -d $UNAME


ARG USERNAME=vscode
ARG USER_UID=1001
ARG USER_GID=1000

# Configure
RUN useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

#conda env export > environment.yml


USER cuquantum

COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN sudo apt-get -y update \
    && sudo apt-get -y install libjpeg-dev zlib1g-dev \ 
    && if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && conda env update -p "/home/cuquantum/conda/envs/cuquantum-23.10" -f /tmp/conda-tmp/environment.yml; fi 

##/home/cuquantum/conda/envs/cuquantum-23.10