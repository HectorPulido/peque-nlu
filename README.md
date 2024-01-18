# Peque NLU - Natural Language Understanding with Machine Learning
Peque-NLU (Natural Language Understanding) is a Python library that allows to parse sentences written in natural language and extract intents, features and information.

For example: `quiero conocer el ultimo blogpost de unity` 
Result: Timing -> latest, Technology -> unity, Intention -> search

## Table of Contents

- [Features](#features)
- [Use Cases](#use-cases)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features
- Feature extraction from text
Agnostic algorithm: you can use SGD, MLNN, LLMs, Word2Vec, etc.
- 100% Free and Open source

## Use cases
- Chatbots, to get intention and extract features
- Search engines, get keywords and intention from a semantic info
- Data mining, classifying text and unstructured data without boilerplate


## Getting Started

### Prerequisites

- Python 3.6+

### Installation

- ⚠️ Pip installation coming soon

1. Clone this repo
```
git clone git@github.com:HectorPulido/peque-nlu.git
```
2. Install the requirements
```
pip install -r requirements.txt
```
3. Use the library
```py
from peque_nlu.intent_engines import SGDIntentEngine
from peque_nlu.intent_classifiers import ModelIntentClassifier


intent_engine = SGDIntentEngine("spanish")
model = ModelIntentClassifier("spanish", intent_engine)
model.fit(DATASET_PATH)

prediction = model.multiple_predict(
    [
        "Hola como te encuentras?",
        "Quiero aprender sobre lo último de python",
        "describeme usando un meme",
    ]
)

assert len(prediction) == 3
first_prediction = prediction[0]
assert "intent" in first_prediction
assert "probability" in first_prediction
assert "text" in first_prediction
assert "features" not in first_prediction

assert first_prediction["intent"] == "small_talk"

```

## Usage
You need to provide to the algorithm before start, you [can check this](https://github.com/HectorPulido/peque-nlu/blob/main/intents_example.json) as base
```json
{
    "intents": {
        "small_talk": [
            "hola",
            ...

        ],
        "fun_phrases": [
            "eres gracioso",
            ...
        ],
        "meme": [
            "¿conoces algun buen meme?",
            ...
        ],
        "thanks": [
            "gracias",
            ...
        ]
    },
    "entities": {
        "technology": [
            "python",
            ...
        ],
        "timing": [
            "recient",
            ...
        ]
    }
}
```

When you have your format ready, you can load and fit your dataset.
```py
intent_engine = SGDIntentEngine("spanish")
model = ModelIntentClassifier("spanish", intent_engine)
model.fit(DATASET_PATH)
```

You can also save and load your models to reduce time and resources.
```py
# Save
saver = PickleSaver()
saver.save(intent_engine, PICKLE_PATH)

# Load
intent_engine_loaded = SGDIntentEngine("spanish")
intent_engine_loaded = saver.load(PICKLE_PATH)
```

Then you can start to predict or extract features from a text
```py
prediction = model.predict("quiero conocer el ultimo blogpost de unity")
```

Response:
```
{
    "intent": "search",
    "features": [
      {
        "word": "ultimo",
        "entity": "timing",
        "similarities": 1
      },
      {
        "word": "otro_ejemplo",
        "entity": "otra_entidad",
        "similarities": 0.9
      }
    ]
  }
```

## Contributing

Your contributions are greatly appreciated! Please follow these steps:

1. Fork the project
2. Create your feature branch `git checkout -b feature/MyFeature`
3. Commit your changes `git commit -m "my cool feature"`
4. Push to the branch `git push origin feature/MyFeature`
5. Open a Pull Request

## License

Every base code made by me is under the MIT license

## Contact

<hr>
<div align="center">
<h3 align="center">Let's connect 😋</h3>
</div>
<p align="center">
<a href="https://www.linkedin.com/in/hector-pulido-17547369/" target="blank">
<img align="center" width="30px" alt="Hector's LinkedIn" src="https://www.vectorlogo.zone/logos/linkedin/linkedin-icon.svg"/></a> &nbsp; &nbsp;
<a href="https://twitter.com/Hector_Pulido_" target="blank">
<img align="center" width="30px" alt="Hector's Twitter" src="https://www.vectorlogo.zone/logos/twitter/twitter-official.svg"/></a> &nbsp; &nbsp;
<a href="https://www.twitch.tv/hector_pulido_" target="blank">
<img align="center" width="30px" alt="Hector's Twitch" src="https://www.vectorlogo.zone/logos/twitch/twitch-icon.svg"/></a> &nbsp; &nbsp;
<a href="https://www.youtube.com/channel/UCS_iMeH0P0nsIDPvBaJckOw" target="blank">
<img align="center" width="30px" alt="Hector's Youtube" src="https://www.vectorlogo.zone/logos/youtube/youtube-icon.svg"/></a> &nbsp; &nbsp;
<a href="https://pequesoft.net/" target="blank">
<img align="center" width="30px" alt="Pequesoft website" src="https://github.com/HectorPulido/HectorPulido/blob/master/img/pequesoft-favicon.png?raw=true"/></a> &nbsp; &nbsp;
