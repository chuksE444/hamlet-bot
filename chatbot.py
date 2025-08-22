import nltk
import string
import re

# Download necessary resources (only needs to run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def preprocess(file_path):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text = re.sub(r"\[.*?\]", "", text)

    # Split into sentences
    sentences = sent_tokenize(text)

    # Lowercase + remove punctuation + remove stopwords
    stop_words = set(stopwords.words("english"))
    cleaned_sentences = []
    for sent in sentences:
        words = word_tokenize(sent.lower())
        words = [w for w in words if w not in string.punctuation]
        words = [w for w in words if w not in stopwords.words("english")]
        cleaned_sentences.append(" ".join(words))

    return sentences, cleaned_sentences


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_most_relevant_sentence(query, sentences, cleaned_sentences):
    # Add the user query to the cleaned sentences
    all_sentences = cleaned_sentences + [query]

    # Vectorize
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_sentences)

    # Compute cosine similarity between query (last one) and all sentences
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Find the index of the most similar sentence
    best_index = np.argmax(cosine_similarities)

    return sentences[best_index], cosine_similarities[best_index]

def chatbot(query, sentences, cleaned_sentences, threshold=0.1):
    best_sentence, score = get_most_relevant_sentence(query, sentences, cleaned_sentences)

    # If similarity score is too low, return a fallback
    if score < threshold:
        return "I'm not sure about that. Can you try asking in a different way?"
    else:
        return best_sentence



if __name__ == "__main__":
    sentences, cleaned_sentences = preprocess("hamlet.txt")

    print("Original sentence example:")
    print(sentences[50])   # show the 51st sentence

    print("\nCleaned version:")
    print(cleaned_sentences[50])

    print("\nTotal sentences:", len(sentences))


while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Bot: Goodbye!")
            break
        response = chatbot(user_input, sentences, cleaned_sentences)
        print("Bot:", response)



