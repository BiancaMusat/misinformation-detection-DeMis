# DeMis: Data-efficient Misinformation Detection using RL
Reinforcement Learning based models for misinformation detection on Twitter. This repo is the official resource of the following paper.
- [DeMis: Data-efficient Misinformation Detection using Reinforcement Learning](https://drive.google.com/file/d/1oQL5R5YiaO3Wdj6o7Nqd7BVAN2kSMxN8/view?usp=sharing), ECML-PKDD 2022.

## 📚 Data Sets
The data sets about COVID-19 misinformation on Twitter presented in [our paper](https://drive.google.com/file/d/1oQL5R5YiaO3Wdj6o7Nqd7BVAN2kSMxN8/view?usp=sharing) are available below.

- COMYTH (weather & home-remedies) - [[Download](XXX)]
- COVIDLies - [[Paper](XXX)]

## 🚀 Pre-trained Models
We release our RL-based models (DeMis) for misinformation detection on Twitter trained on three COVID-19 misinformation data sets separately. All models are uploaded to my [Huggingface](https://huggingface.co/kornosk) 🤗 so you can load model with **just three lines of code**!!!

- [DeMis-COMYTH-W](XXX) (trained on COVID-weather data)
- [DeMis-COMYTH-H](XXX) (trained on COVID-home-remedies data)
- [DeMis-COVIDLies](XXX) (trained on COVIDLies data)

## ⚙️ Usage
We tested in `pytorch v1.10.1` and `transformers v4.18.0`.

### 1. Choose and load the model and tokenizer for misinformation detection
```python
from transformers import AutoModel, AutoTokenizer, pipeline
import torch

# Choose GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select mode path here
pretrained_LM_path = "XXX"

# Load model
tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
model = AutoModel.from_pretrained(pretrained_LM_path)
```

### 2. Get a prediction (see more in `sample_predict.py`)
```python
id2label = {
    0: "legitimate",
    1: "misinfo"
}

##### Prediction #####
sentence = "Heat in the summer kills COVID!!!"
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)
predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()

print("Sentence:", sentence)
print("Prediction:", id2label[np.argmax(predicted_probability)])
print("Legitimate:", predicted_probability[0])
print("Misinfo:", predicted_probability[1])

# please consider citing our paper if you feel this is useful :)
```

## ✏️ Citation
If you feel our paper and resources are useful, please consider citing our work! 🙏
```bibtex
@inproceedings{kawintiranon2022demis,
  title     = {DeMis: Data-efficient Misinformation Detection using Reinforcement Learning},
  author    = {Kawintiranon, Kornraphop and Singh, Lisa},
  booktitle = {Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD)},
  year      = {2022},
  publisher = {Springer}
}
```

##  🛠 Throubleshoots
[Create an issue here](https://github.com/GU-DataLab/misinformation-detection-DeMis/issues) if you have any issues loading models or data sets.
