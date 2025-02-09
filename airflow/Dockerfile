FROM apache/airflow:2.9.0-python3.9

ARG CURRENT_USER=$USER

USER root
# Install Python dependencies to be able to process the wheels from the private PyPI server.
RUN apt-get -y update && ACCEPT_EULA=Y apt-get -y upgrade && apt -y install build-essential

USER ${CURRENT_USER}
