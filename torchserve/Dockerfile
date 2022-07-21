FROM python:3

RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6  openjdk-11-jdk ant \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*\
    && rm -rf /var/cache/oracle-jdk11-installer;

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY ./configs.properties ./configs.properties
COPY ./serve_alldet ./serve_alldet

CMD ["torchserve","--start", "--foreground", "--ts-config", "./configs.properties"]