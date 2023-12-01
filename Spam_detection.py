import tkinter as tk
from tkinter import messagebox
import pickle
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Khởi tạo PorterStemmer và stop words
stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Hàm mở rộng các từ viết tắt
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

# Hàm tiền xử lý tin nhắn
def preprocess_message(message):
    senti = re.sub('[^A-Za-z]', ' ', message)
    senti = senti.lower()
    words = word_tokenize(senti)
    words = [stem.stem(word) for word in words if word not in stop_words]
    senti = ' '.join(words)
    senti = expand_abbreviations(senti)
    return senti

# Hàm để dự đoán tin nhắn
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
    text_entry.delete("1.0", "end")  # Xóa nội dung đã nhập sau khi dự đoán

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
