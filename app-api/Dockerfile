FROM python:3.9.4

WORKDIR /app/src

RUN apt-get update && apt-get upgrade -y
RUN pip install --no-cache -U pip
RUN pip install --no-cache poetry==1.8.5

# Configuring poetry.
RUN poetry config virtualenvs.create false

# First copy & install requirements to speed up the build process in case only the code changes.
COPY pyproject.toml /app/src/
COPY poetry.lock /app/src/

RUN poetry install --no-interaction --no-root -vvv

# Copy the rest of the files.
ADD . /app/src

# Give access to run the run.sh script.
RUN chmod +x run.sh

CMD ["bash", "./run.sh"]