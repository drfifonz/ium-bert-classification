FROM condaforge/mambaforge:latest

# seting env variables
ENV KAGGLE_USERNAME=filippatyk
ENV KAGGLE_KEY=""
ENV RUN_TYPE=""

# create working direcotyry
WORKDIR /app
ENV KAGGLE_CONFIG_DIR="./" 

# create mamba env and activate it
COPY environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml && mamba clean -ya
RUN echo "mamba activate ium" >> ~/.bashrc
ENV PATH /opt/conda/envs/ium/bin:$PATH
# RUN pip install 'dvc[ssh]' paramiko

RUN useradd -r -u 111 jenkins

COPY src ./src
COPY data ./data
#make dir for data
RUN mkdir -p ./data; mkdir -p ./results


CMD kaggle datasets download -p data --unzip clmentbisaillon/fake-and-real-news-dataset && python ./dataset.py --dataset && python ./src/main.py "--$RUN_TYPE"