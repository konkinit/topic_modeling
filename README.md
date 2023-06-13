<h1 align="center">
  Topic Modeling with on BERTopic
</br>
</h1>

<p align="center">
  <img alt="GitHub Workflow Status" src="https://img.shields.io/github/actions/workflow/status/konkinit/topic_modeling/topic_app_ci.yaml?label=Test%20%26%20Build%20Image&style=for-the-badge">
</br>
  <img alt="GitHub License" src="https://img.shields.io/github/license/konkinit/topic_modeling?style=for-the-badge">
  <a href="https://www.python.org/downloads/release/python-3100/" target="_blank">
    <img src="https://img.shields.io/badge/python-3.10-blue.svg?style=for-the-badge" alt="Python Version"/>
  </a>
  <img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-black?style=for-the-badge">
</br>
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/konkinit/topic_modeling?style=for-the-badge">
  <img alt="Docker Image Size (latest by date)" src="https://img.shields.io/docker/image-size/kidrissa/bertopicapp?style=for-the-badge">
</p>

## Description

The project consists of a packaging of BERTopic modeling with Streamlit framework. Instructions and more
details are provided in the app ...

## Getting Started

```bash
docker pull kidrissa/bertopicapp:latest
```

```bash
docker run -p 8501:8501 -d kidrissa/bertopicapp:latest
```

## Citation

```bib
@article{grootendorst2022bertopic,
  title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},
  author={Grootendorst, Maarten},
  journal={arXiv preprint arXiv:2203.05794},
  year={2022}
}
```
