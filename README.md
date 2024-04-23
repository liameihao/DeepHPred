# DeepHPred

# Installation

python>3.7

```
pip install selene-sdk
pip install torch
```

# Use

## example

```
from selene_sdk.utils import load_path
from selene_sdk.utils import parse_configs_and_run
# load config 
configs = load_path("config_yml/run.yml")
# run
parse_configs_and_run(configs, lr=0.01)
```

The file address in the run.yml file is best given as an absolute path.

More detailed results files will be made public after we publish the article.

## model_name

- DeepHPred-Type1: model/DeepHPred-Type1.py
- DeepHPred-Type2: model/DeepHPred-Type2.py
- Basset: model/basset.py
- DeepSEA: model/deepsea.py
- DeepSEA Beluga: model/deeper_deepsea_arch.py
