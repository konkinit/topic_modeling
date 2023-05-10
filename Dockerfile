FROM python:3.10-slim

COPY . /topic_modeling

WORKDIR /topic_modeling

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "./src/frontend/Onboarding.py"]
