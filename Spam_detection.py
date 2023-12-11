import tkinter as tk
from tkinter import messagebox
import pickle
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def expand_abbreviations(senti):
    abbreviation_dict = {
        "dun": "do not",
        "don": "do not",
        "cant": "can not",
        "pl": "please",
        "dont": "do not"
    }
    for abbr, expanded in abbreviation_dict.items():
        senti = senti.replace(abbr, expanded)
    return senti


def preprocess_message(message):
    senti = re.sub('[^A-Za-z]', ' ', message)
    senti = senti.lower()
    words = word_tokenize(senti)
    words = [stem.stem(word) for word in words if word not in stop_words]
    senti = ' '.join(words)
    senti = expand_abbreviations(senti)
    return senti


def predict_spam():
    new_message = text_entry.get("1.0", "end-1c")
    processed_message = preprocess_message(new_message)
    vectorized_message = loaded_cv.transform([processed_message])
    dense_vectorized_message = vectorized_message.toarray()
    prediction = loaded_model.predict(dense_vectorized_message)
    if prediction[0] == 'ham':
        messagebox.showinfo("Prediction", "This message is not Spam.")
    else:
        messagebox.showinfo("Prediction", "This message is Spam.")
    text_entry.delete("1.0", "end")

# Tải mô hình và CountVectorizer
with open('svm_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('count_vectorizer.pkl', 'rb') as file:
    loaded_cv = pickle.load(file)

# Thiết lập giao diện tkinter
root = tk.Tk()
root.title("SMS Spam Detector")

label = tk.Label(root, text="Enter a message:", padx=10, pady=10)
label.pack()

text_entry = tk.Text(root, height=10, width=50)
text_entry.pack()

predict_button = tk.Button(root, text="Predict", command=predict_spam)
predict_button.pack()

root.mainloop()
