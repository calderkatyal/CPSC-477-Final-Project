# CSPC 477 Final Project

**Authors:** Calder Katyal, Jess Yatvitskiy  
**Affiliation:** Yale University  
**Emails:** calder.katyal@yale.edu, jess.yatvitskiy@yale.edu

---

## Setup

Download `faiss-gpu`

```
conda install -c pytorch faiss-gpu
```

Download Docker at https://www.docker.com/products/docker-desktop and run the following to ensure it is set up correctly.

```
docker version
docker compose version
```

Donwload requirements

```
pip install -r requirements.txt
```

## Temporary Pipeline (run from ROOT)

Clone this directory

Download dataset

```
python setup.py
```

Preprocess the data

```
python src/preprocessing/preprocess.py
```

Spin up the docker container

```
docker compose up -d 
```

Store metadate in PostgreSQL

```
python src/database/insert_metadata.py
```

To view the data

```
docker exec -it email_postgres psql -U postgres -d emails_db
```