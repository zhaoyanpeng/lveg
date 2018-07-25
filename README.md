## Latent Vector Grammars (LVeGs)

Code for [Gaussian Mixture Latent Vector Grammars](https://arxiv.org/abs/1805.04688).

## How to Run

Specify parameter values for learning (param.in) and for inference (param.f1).

### Learning

```
$ sbt "run-main edu.shanghaitech.ai.nlp.lveg.LVeGTrainerImp param.in"
```

### Inference

```
$ sbt "run-main edu.shanghaitech.ai.nlp.lveg.LVeGTesterImp param.f1"
```

## Data

Parsing data available at [Google Drive](https://drive.google.com/open?id=1sSwTaVgKJe-oA7jsoM6Mbij_j-xDUdH7).

## Models

Parsing models available at [Google Drive](https://drive.google.com/open?id=1CqWOMn7xWfax5Sj5lP_ypKYieasjviKd).

## Dependencies

sbt-0.13.10, java-1.8.0, and scala-2.12.0.