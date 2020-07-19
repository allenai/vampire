## Installing API

VAMPIRE can be trained and loaded for predictions using a new API detailed in `vampire/api`.

First, make sure you install the library by cloning and pip installing: 

```bash
git clone https://github.com/allenai/vampire
cd vampire
pip install --editable .
```

### Training and Using VAMPIRE via API

```python
from vampire.api import preprocess_data
from vampire.api import VampireModel
from vampire.api import VampireTokenizer
from pathlib import Path



train_path = Path("examples/ag/train.jsonl")
tokenized_train_path = Path("examples/ag/train.tok.jsonl")
dev_path = Path("examples/ag/dev.jsonl")
tokenized_dev_path = Path("examples/ag/dev.tok.jsonl")
data_dir = Path("examples/ag/world")
vocab_size = 10000

# Tokenize data with spacy and 20 workers
tokenizer = VampireTokenizer(tokenizer='spacy')
tokenizer.pretokenize(num_workers=20,
                      input_file=train_path,
                      output_file=tokenized_train_path,
                      is_json=True,
                      lower=True,
                      silent=True)
tokenizer.pretokenize(num_workers=20,
                      input_file=dev_path,
                      output_file=tokenized_dev_path,
                      is_json=True,
                      lower=True,
                      silent=True)

# Preprocess data into bag-of-words vectors
preprocess_data(train_path=tokenized_train_path,
                dev_path=tokenized_dev_path,
                vocab_size=10000,
                serialization_dir=data_dir,
                tfidf=True)

# Instantiate VAMPIRE
vampire = VampireModel.from_params(data_dir=data_dir, hidden_dim=81)

# Fit VAMPIRE to data directory on CPU
# Output VAMPIRE model in serialization_dir path
vampire.fit(data_dir=data_dir, serialization_dir=Path("model_logs/vampire"), cuda_device=-1)

# Extract scalar mix features of a document
vampire.extract_features({"text": "This is a sentence."})

```