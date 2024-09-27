FROM python:3-slim

USER root

RUN pip3 install poetry
RUN poetry install

