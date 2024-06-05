FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages (do i need libgl1-mesa-glx and libglib2.0-0?)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \ 
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Set Python3 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Clone the GitHub repository  (uses personal access token)
RUN git clone https://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@github.com/NamanSharma5/svd-unisim.git /workspace/repo

# Set the working directory to the repository
WORKDIR /workspace/repo

# Copy the requirements file into the image
COPY requirements.txt /workspace/repo/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint

ENTRYPOINT ["python", "train_svd_multi_img_text.py"] 