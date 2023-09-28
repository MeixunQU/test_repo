ARG BASE_IMAGE=nvcr.io/nvidia/l4t-ml:r35.2.1-py3

FROM ${BASE_IMAGE}

ARG ROS_PKG=desktop-full
ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}

ENV PATH="/usr/local/cuda-11.4/bin:$PATH"

ENV LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y  curl \
						git \
						g++ \
                		nano \
                		vim \
                		python3-pip \
                		libeigen3-dev \
                		tmux \
                		tmuxinator \
						build-essential \
						wget \
						gnupg2 \
						lsb-release \
						ca-certificates \
						&& rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -


RUN apt-get purge -y '*opencv*'

RUN apt-get update -y && \
	apt-get upgrade -y && \
    apt-get install -y --no-install-recommends 	ros-noetic-`echo "${ROS_PKG}" | tr '_' '-'` \
												ros-noetic-image-transport \
        										python3-matplotlib \
												python3-tk

RUN apt-get upgrade -y

WORKDIR /app

COPY . .

RUN wget https://paddle-inference-lib.bj.bcebos.com/2.4.2/python/Jetson/jetpack5.0.2_gcc9.4/xavier/paddlepaddle_gpu-2.4.2-cp38-cp38-linux_aarch64.whl

RUN python3 -m pip install -r requirements.txt

RUN echo 'source ${ROS_ROOT}/setup.bash' >> /root/.bashrc


# CMD ["bash"]
