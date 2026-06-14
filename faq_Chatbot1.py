import tkinter as tk
from tkinter import scrolledtext

import nltk
import string

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt', quiet=True)

# FAQ Dataset
faq_data = {
    "What is AI?":
        "Artificial Intelligence is the simulation of human intelligence by machines.",

    "What is Machine Learning?":
        "Machine Learning is a subset of Artificial Intelligence that learns from data.",

    "What is Deep Learning?":
        "Deep Learning uses neural networks with multiple layers.",

    "What is Python?":
        "Python is a popular programming language used in AI and Data Science.",

    "What is NLP?":
        "Natural Language Processing helps computers understand human language."
}

questions = list(faq_data.keys())

# Text Preprocessing
def preprocess(text):
    text = text.lower()

    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()

    tokens = [
        word for word in tokens
        if word not in string.punctuation
    ]

    return " ".join(tokens)

processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

# Chatbot Response Function
def chatbot_response():

    user_question = entry.get().strip()

    if user_question == "":
        return

    chat_area.insert(
        tk.END,
        "🧑 You: " + user_question + "\n"
    )

    user_processed = preprocess(user_question)

    user_vector = vectorizer.transform(
        [user_processed]
    )

    similarity = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match_index = similarity.argmax()
    best_score = similarity[0][best_match_index]

    if best_score > 0.6:
        answer = faq_data[
            questions[best_match_index]
        ]
    else:
        answer = "Sorry, I don't know the answer."

    chat_area.insert(
        tk.END,
        "🤖 Bot: " + answer + "\n\n"
    )

    chat_area.see(tk.END)

    entry.delete(0, tk.END)

# =========================
# GUI WINDOW
# =========================
root = tk.Tk()
root.title("🤖 AI FAQ Chatbot")
root.geometry("900x700")
root.configure(bg="#1E1E2E")
root.resizable(False, False)

# =========================
# TITLE
# =========================
title = tk.Label(
    root,
    text="🤖 AI FAQ Chatbot",
    font=("Segoe UI", 24, "bold"),
    bg="#1E1E2E",
    fg="#00E5FF"
)
title.pack(pady=15)

# =========================
# SUBTITLE
# =========================
subtitle = tk.Label(
    root,
    text="Ask questions about AI, ML, Python, NLP and more",
    font=("Segoe UI", 11),
    bg="#1E1E2E",
    fg="white"
)
subtitle.pack()

# =========================
# MAIN FRAME
# =========================
main_frame = tk.Frame(
    root,
    bg="#2A2A40",
    bd=0
)
main_frame.pack(
    padx=20,
    pady=15,
    fill="both",
    expand=True
)

# =========================
# CHAT AREA
# =========================
chat_area = scrolledtext.ScrolledText(
    main_frame,
    width=85,
    height=22,
    font=("Segoe UI", 11),
    bg="#121212",
    fg="white",
    insertbackground="white",
    relief="flat"
)
chat_area.pack(
    padx=15,
    pady=15
)

# Welcome Message
chat_area.insert(
    tk.END,
    "🤖 Bot: Hello! Ask me anything from my FAQ database.\n\n"
)

# =========================
# INPUT FRAME
# =========================
input_frame = tk.Frame(
    main_frame,
    bg="#2A2A40"
)
input_frame.pack(pady=10)

entry = tk.Entry(
    input_frame,
    width=50,
    font=("Segoe UI", 13),
    bg="#F5F5F5",
    relief="flat"
)
entry.grid(
    row=0,
    column=0,
    padx=10,
    ipady=8
)

# =========================
# SEND BUTTON
# =========================
send_btn = tk.Button(
    input_frame,
    text="🚀 Send",
    command=chatbot_response,
    font=("Segoe UI", 11, "bold"),
    bg="#00C853",
    fg="white",
    activebackground="#00E676",
    activeforeground="white",
    width=12,
    relief="flat",
    cursor="hand2"
)
send_btn.grid(
    row=0,
    column=1,
    padx=5
)

# =========================
# CLEAR CHAT BUTTON
# =========================
clear_btn = tk.Button(
    input_frame,
    text="🗑 Clear",
    command=lambda: chat_area.delete(1.0, tk.END),
    font=("Segoe UI", 11, "bold"),
    bg="#FF5252",
    fg="white",
    activeforeground="white",
    width=12,
    relief="flat",
    cursor="hand2"
)
clear_btn.grid(
    row=0,
    column=2,
    padx=5
)

# Press Enter to Send
entry.bind(
    "<Return>",
    lambda event: chatbot_response()
)


# =========================
# RUN APPLICATION
# =========================
root.mainloop()