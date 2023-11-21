# FDDF

## How to get the code

```
git clone https://github.com/401bteamFL/FDDF.git
```

## Training Setting

In the training code, the adversary rate, trigger value, and trigger size are modifiable.

| variable       | code                                           |
|----------------|------------------------------------------------|
| adversary rate | ```ATTACK_NUM```                               |
| trigger value  | ```features[j][k][position] = trigger value``` |
| trigger size   | ```features[j][k][position] = trigger value``` |


In the framework code, the K ranges is modifiable.

| variable       | code                                           |
|----------------|------------------------------------------------|
| K ranges       | ```vote = [-k]```                              |


In both the training and framework code, testing can be performed on binary and multi-class labeled data by adjusting certain settings.

| file                | variable   | code                         |
|---------------------|------------|------------------------------|
| 'model_training.py' | num_class  | ```num_class= class number```|
| 'framework.py'      | num_class  | ```num_class= class number```|