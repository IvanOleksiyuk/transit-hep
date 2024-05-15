FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt -y update && \
    apt -y upgrade && \
    apt -y install \
        build-essential \
        git \
        texlive \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-fonts-extra \
        cm-super-minimal # needed for latex-support in matplotlib

COPY requirements.txt .

RUN conda update conda
RUN conda install pip

RUN pip install --upgrade pip
RUN pip install --upgrade -r requirements.txt
