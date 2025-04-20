# CSPC 477 Final Project

**Authors:** Calder Katyal, Jess Yatvitskiy  
**Affiliation:** Yale University  
**Emails:** calder.katyal@yale.edu, jess.yatvitskiy@yale.edu

---

## Setup

Note: GPU is not required. We have provided pre-generated embeddings, and any computer with decent RAM should be able to run our code.

First download Docker at https://www.docker.com/products/docker-desktop and run the following to ensure it is set up correctly:

```
docker version
docker compose version
```

Clone this directory, and donwnload the requirements:

```
pip install -r requirements.txt
```

Ensure you have a valid Kaggle API token at `~/.kaggle/`, and then run:

```
python main.py
```
