<h1 align="center">
  Topic Modeling with BERTopic
</br>
</h1>

<p align="center">
  The project consists of a packaging of BERTopic modeling with Streamlit framework. Instructions and more
  details are provided in the app ...
</p>

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


## ToDo

- Fix Docker CI
- Tune BERTopic hyperparams
- Write Unit Tests ...


## Getting Started

- Through Docker Image:
```bash
docker pull kidrissa/bertopicapp:latest
```

```bash
docker run -p 8501:8501 -d kidrissa/bertopicapp:latest
```

- Through Repo cloning (to be performed preferable on a Linux-based OS):
```bash
git clone https://github.com/konkinit/topic_modeling.git
```

```bash
cd topic_modeling/
```

```bash
bash package_installing.sh
```

```bash
streamlit run ./src/frontend/Onboarding.py
```


## Continious Integration

One continous integration (CI) procedure with 2 jobs mainly is crafted and launched at 
every push to the main branch:

- Pytest collects the test from the tests folder and executes them
- if Testing passes, a Docker Image is built and pushed onto the docker hub

## Citation

```bib
@article{grootendorst2022bertopic,
  title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},
  author={Grootendorst, Maarten},
  journal={arXiv preprint arXiv:2203.05794},
  year={2022}
}
```
