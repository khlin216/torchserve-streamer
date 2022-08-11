FROM python:3.9.13-buster

WORKDIR /app
ENV EKS=True

COPY . .
USER root
RUN pip install -r requirements.txt

CMD [ "python","-u",  "swarm.py" ]