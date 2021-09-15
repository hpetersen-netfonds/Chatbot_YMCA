import training
import chatbot
import nltk

# check for required resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('stem/wordnet')
except LookupError:
    nltk.download('wordnet')

training.train()

chatbot.run_chatbot()
