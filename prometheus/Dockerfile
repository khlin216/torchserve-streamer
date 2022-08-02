FROM prom/prometheus
COPY prometheus.${ARCH:-dev}.yml /etc/prometheus/prometheus.yml
RUN echo $(ls /etc/prometheus)