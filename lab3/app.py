import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rapidfuzz import process

# === Модели ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# === Данные и индекс ===
df = pd.read_parquet("arxiv_metadata.parquet.gzip")
index = faiss.read_index("arxiv_index.faiss")

# === Вокабуляры для исправления опечаток ===
vocab = set(df["topic"]).union(*df["keywords"].apply(lambda x: x.split(", ")))
authors_vocab = set(df["authors"])

# === Функции ===
def correct_query(query, choices):
    match, score, _ = process.extractOne(query, choices)
    return match if score > 70 else query

def ask_flan(prompt):
    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = flan_model.generate(**inputs, max_new_tokens=256)
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

def search_similar_chunks(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return df.iloc[indices[0]]

# === Streamlit UI ===
st.title("🔍 arXiv Semantic Search + FLAN-T5")
query = st.text_input("Введите поисковый запрос:")

top_k = st.slider("Сколько результатов показать", min_value=1, max_value=20, value=5)

col1, col2, col3 = st.columns(3)
with col1:
    filter_topic = st.text_input("Фильтр по теме (опционально):")
with col2:
    filter_author = st.text_input("Фильтр по автору (опционально):")
with col3:
    filter_kw = st.text_input("Фильтр по ключевым словам (опционально):")

use_llm = st.checkbox("Показать краткие содержания через FLAN-T5")
summarize_all = st.checkbox("Суммировать все статьи (может занять время)")

if st.button("Искать"):
    if not query:
        st.warning("Введите поисковый запрос")
    else:
        corrected_query = correct_query(query, vocab.union(authors_vocab))
        st.write(f"🔎 Интерпретируем как: **{corrected_query}**")

        results = search_similar_chunks(corrected_query, top_k=top_k)

        if filter_topic:
            filter_topic = correct_query(filter_topic, vocab)
            results = results[results["topic"].str.lower() == filter_topic.lower()]
        if filter_author:
            filter_author = correct_query(filter_author, authors_vocab)
            results = results[results["authors"].str.contains(filter_author, case=False)]
        if filter_kw:
            filter_kw = correct_query(filter_kw, vocab)
            results = results[results["keywords"].str.contains(filter_kw, case=False)]

        if results.empty:
            st.warning("Ничего не найдено.")
        else:
            for _, row in results.iterrows():
                with st.expander(f"📄 {row['title']}"):
                    st.markdown(f"**Авторы**: {row['authors']}")
                    st.markdown(f"**Ключевые слова**: {row['keywords']}")
                    st.markdown(f"**Тема**: {row['topic']}")
                    st.text_area("Текст чанка", row["chunk"], height=150)

                    if use_llm and not summarize_all:
                        with st.spinner("FLAN-T5 думает..."):
                            prompt = f"Summarize the key ideas of this paper:\n{row['chunk']}"
                            summary = ask_flan(prompt)
                            st.success("Краткое содержание:")
                            st.write(summary)

            if summarize_all:
                st.subheader("📚 FLAN-T5: краткое содержание всех результатов")
                for i, row in results.iterrows():
                    with st.spinner(f"FLAN-T5 обрабатывает: {row['title']}"):
                        prompt = f"Summarize the key ideas of this paper:\n{row['chunk']}"
                        summary = ask_flan(prompt)
                        st.markdown(f"**📝 {row['title']}**")
                        st.write(summary)
                        st.markdown("---")