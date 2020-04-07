FROM nvidia/cudagl:10.1-devel-ubuntu18.04

RUN apt-get update && apt-get upgrade -y

# Install packages
RUN apt-get install -y wget git vim libsm6 libxext6 libxrender-dev ffmpeg python-opengl

# install anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
RUN bash Anaconda3-2019.10-Linux-x86_64.sh -b
RUN rm Anaconda3-2019.10-Linux-x86_64.sh
ENV PATH /root/anaconda3/bin:$PATH
RUN conda update conda
RUN yes | conda update anaconda
RUN yes | conda update --all
RUN conda init

# Install packages
RUN conda install -y -c pytorch pytorch torchvision
RUN conda install -y tensorflow-gpu==1.14.0
RUN pip install gin-config
RUN pip install gym[atari]
RUN pip install gym[box2d]

WORKDIR /root
RUN git clone https://github.com/openai/baselines.git
WORKDIR /root/baselines
RUN git checkout adba88b218edbb3f3bd8044dc963f8ca0901d6c6
RUN pip install .
WORKDIR /root

# Add a directory for python packages to be mounted
ENV PYTHONPATH /root/pkgs:$PYTHONPATH

# Bash entrypoint
ENTRYPOINT /bin/bash
