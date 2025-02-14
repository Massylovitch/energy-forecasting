FROM python:3.9.8

WORKDIR /app/src

RUN apt-get update && apt-get upgrade -y
RUN pip install --no-cache -U pip
RUN pip install --no-cache poetry==1.4.2

RUN poetry config virtualenvs.create false

COPY ./app-monitoring/pyproject.toml /app/src/
COPY ./app-monitoring/poetry.lock /app/src/

RUN poetry install --no-interaction --no-root -vvv

ADD ./app-monitoring /app/src

CMD ["streamlit", "run", "monitoring/main.py", "--server.port", "8502"]