Simple example for cocoindex: build embedding index based on local files.

## Prerequisite
Follow [Setup Postgres](../../#setup-postgres) section on the root directory to setup Postgres database.

## Run

Install dependencies:

```bash
pip install -e .
```

Setup:

```bash
python pdf_embedding.py cocoindex setup
```

Update index:

```bash
python text_embedding.py cocoindex update
```

Run:

```bash
python pdf_embedding.py
```
