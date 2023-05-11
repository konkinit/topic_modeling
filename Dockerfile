FROM python:3.10-slim

COPY . /topic_modeling

WORKDIR /topic_modeling

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python3 -m spacy download fr_core_news_md && \
    python3 -m spacy download en_core_web_sm

ENTRYPOINT ["streamlit", "run", "./src/frontend/Onboarding.py"]
