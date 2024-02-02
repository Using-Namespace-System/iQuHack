FROM docker.io/nvidia/cuda:12.3.1-runtime-ubuntu22.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Configure
RUN apt-get -y update \
    # install git
    && apt-get -y install git openssh-client less iproute2 procps lsb-release \
    #install wget
    && apt-get -y install wget \
    # Create user
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh -O ~/Anaconda3-2023.09-0-Linux-x86_64.sh \
    && /bin/bash ~/Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /opt/conda \
    && /opt/conda/bin/conda update -n base conda \
    #add fast solver
    && /opt/conda/bin/conda install -n base conda-libmamba-solver \
    && /opt/conda/bin/conda config --set solver libmamba

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

#to build yml
#add write access to conda environment using: chown -R $USER_UID:USER_GID /opt/conda
#configure as needed
#then run: conda env export > environment.yml