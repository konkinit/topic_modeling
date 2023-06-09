FROM python:3.10-slim

COPY . ./topic_modeling

WORKDIR /topic_modeling

RUN apt-get update && \
    apt-get -y install gcc

RUN bash package_installing.sh

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "./src/frontend/Onboarding.py"]
