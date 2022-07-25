import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()


def load_parsed_csv(datasets_path='datasets/records_4k.csv', encoding='ISO-8859-1'):
    data = pd.read_csv(datasets_path, encoding=encoding)
    data = data[data.title != '[none]']

    return data


def basic_clean(title):
    lowercase_sentence = re.sub("[^a-zA-Z]", " ", str(title))
    lowercase_sentence = lowercase_sentence.lower()

    lowercase_sentence = lowercase_sentence.translate(str.maketrans("", "", string.punctuation))

    lowercase_sentence = lowercase_sentence.strip()

    lowercase_sentence = re.sub('\s+', ' ', lowercase_sentence)
    tokens = nltk.tokenize.word_tokenize(lowercase_sentence)
    stops = set(stopwords.words("english"))

    meaningful_words = [lemmatizer.lemmatize(w) for w in tokens if not w in stops]
    return ",".join(meaningful_words)


def split_dataset():
    data = load_parsed_csv('../../datasets/all/records_4k.csv')
    train, test = train_test_split(data, test_size=0.2, random_state=0)
    val, test = train_test_split(test, test_size=0.5, random_state=1)

    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    val = pd.DataFrame(val)

    train.to_csv('../../datasets/all/training.csv')
    test.to_csv('../../datasets/all/testing.csv')
    val.to_csv('../../datasets/all/validation.csv')

if __name__ == "__main__":
    split_dataset()