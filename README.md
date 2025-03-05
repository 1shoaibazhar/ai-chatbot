# Chatbot

This project trains a chatbot using Natural Language Processing (NLP) and a neural network.

## Features
- Uses **Bag of Words (BoW)** for text representation.
- Implements a **TensorFlow-based neural network**.
- Classifies user input into predefined **intents**.
- Uses **SGD optimizer with momentum** for training.

## Installation
### Prerequisites
- Python 3.x
- Required Libraries:
  ```bash
  pip install numpy tensorflow nltk pickle5 
  ```

## Usage
1. **Prepare `intents.json`**
   - Define intents and training patterns.
2. **Train the Model**
   ```bash
   python train.py
   ```
   - This script tokenizes text, creates a BoW model, and trains a neural network.
3. **Save & Use the Model**
   - The trained model is saved as `chatbot_model.keras`.
   - Use it for chatbot predictions.

## File Structure
- `intents.json` – Contains chatbot responses and training data.
- `train.py` – Main script to train the chatbot model.
- `words.pkl` / `classes.pkl` – Saved tokenized words and intent classes.
- `chatbot_model.keras` – Trained model.

## How It Works
1. **Preprocessing**
   - Tokenizes text and applies **lemmatization**.
   - Converts text to a **Bag of Words (BoW)**.
2. **Neural Network Training**
   - A multi-layer neural network learns to classify intents.
3. **Prediction**
   - User input is transformed into BoW and passed through the trained model.

## Example Intents Format (`intents.json`)
```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hello", "Hi", "Hey"],
      "responses": ["Hello!", "Hi there!"]
    }
  ]
}
```

## Credits
Developed using **Python, TensorFlow, and NLTK**.


