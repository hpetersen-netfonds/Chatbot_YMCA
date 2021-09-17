import training
import chatbot
import nltk

# check for required resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

training.train()

chatbot.run_chatbot()
