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

Spin up the docker container: 

```
docker compose up -d
```

Ensure you have a valid Kaggle API token at `~/.kaggle/`.

## Launch the Email Search Engine

To run the email search engine, run the following command in the root directory of this repository:

```
python main.py
```

## Evaluations

To recreate our evaluations, run the following command in the root directory of this repository:

```
python main.py --is_test --seed=42
```
