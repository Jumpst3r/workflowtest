FROM continuumio/miniconda3
LABEL maintainer="nicolas.dutly@unifr.ch"
COPY ./code/environment.yml /input/
RUN apt update && apt install --no-install-recommends -y libgl1-mesa-glx
RUN apt-get autoremove -y && apt-get clean && \
    conda clean -i -l -t -y && \
    rm -rf /usr/local/src/*
RUN conda env create --force -f /input/environment.yml && conda clean -afy
COPY ./code/* /input/
RUN cd /input && mkdir models && mv fcnn_bin.h5 models/
RUN sed -i '$d' /root/.bashrc
RUN echo "conda activate myenv" >> /root/.bashrc

WORKDIR /input/
