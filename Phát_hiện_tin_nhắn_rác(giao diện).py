import tkinter as tk
from tkinter import messagebox
import pickle
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

with open('svm_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('count_vectorizer.pkl', 'rb') as file:
    loaded_cv = pickle.load(file)

stem = PorterStemmer()
stop_words = set(stopwords.words())

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
        messagebox.showinfo("Nhận diện", "Đây không phải tin nhắn rác.")
    else:
        messagebox.showinfo("Nhận diện", "Đây là tin nhắn rác.")
    text_entry.delete("1.0", "end")



# Thiết lập giao diện tkinter
root = tk.Tk()
root.title("Nhận diện tin nhắn rác")

label = tk.Label(root, text="Nhập nội dung tin nhắn:", padx=10, pady=10)
label.pack()

text_entry = tk.Text(root, height=10, width=50)
text_entry.pack()

predict_button = tk.Button(root, text="Nhận diện", command=predict_spam)
predict_button.pack()

root.mainloop()
