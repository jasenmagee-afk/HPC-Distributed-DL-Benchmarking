# Use a recent Ubuntu image
FROM ubuntu:22.04

# Set up the environment
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    openmpi-bin libopenmpi-dev \
    python3.10 python3-pip python3-dev \
    libomp-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up environment variables
ENV PATH="/usr/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
ENV CPLUS_INCLUDE_PATH="/usr/include/openmpi:${CPLUS_INCLUDE_PATH}"

# Install Python packages
RUN pip3 install --no-cache-dir torch numpy pandas mpi4py horovod deepspeed

# Set the working directory
WORKDIR /app

# --- CRITICAL NEW STEP: Copy source files for compilation ---
COPY . /app/

# --- CRITICAL NEW STEP: Compile the C++ module into the image ---
# The --user flag ensures it installs to a user directory that is always on the path
RUN python3 setup.py install


CMD ["/bin/bash"]
