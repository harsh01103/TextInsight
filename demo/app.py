"""TextInsight demo: paste English or Hinglish text and see every analysis.

Run locally with: streamlit run demo/app.py
"""

from __future__ import annotations

import streamlit as st

import textinsight as ti

EXAMPLES = {
    "Hinglish phone review": (
        "Is phone ki battery life bahut acchi hai. Camera quality bhi mast hai "
        "lekin display thoda dim hai."
    ),
    "Mixed feelings": "camera mast hai but battery bekaar hai",
    "English review": "The food was excellent but the service was terrible.",
    "Personal details": "Mail a.person@example.com or call 9876543210 about the order.",
}

st.set_page_config(page_title="TextInsight demo", layout="centered")
st.title("TextInsight")
st.write("NLP for English and Hinglish, with no model downloads or API calls.")

choice = st.selectbox("Try an example", list(EXAMPLES))
text = st.text_area("Text", value=EXAMPLES[choice], height=120, max_chars=5000)

if text.strip():
    language = ti.detect_language(text).language
    st.caption(f"Detected language: {language}")
    if language == "hi":
        st.warning("Hindi in Devanagari script is not supported yet. Try romanized Hinglish.")
    else:
        mood = ti.sentiment(text)
        first, second = st.columns(2)
        first.metric("Sentiment", mood.label.capitalize())
        second.metric("Score (-1 to +1)", f"{mood.score:+.2f}")

        st.subheader("Aspects")
        found = ti.aspects(text)
        if len(found):
            st.table([{"Aspect": a.name, "Sentiment": a.label} for a in found])
        else:
            st.write("No opinions about specific things were found.")

        st.subheader("Keywords")
        st.write(", ".join(ti.keywords(text, top_n=6).texts) or "None found.")

        st.subheader("Summary")
        st.write(ti.summarize(text, max_sentences=2).summary)

    st.subheader("Personal information hidden")
    st.write(ti.redact(text).text)

    numbers = ti.stats(text)
    st.subheader("Statistics")
    first, second, third = st.columns(3)
    first.metric("Words", numbers.words)
    second.metric("Sentences", numbers.sentences)
    third.metric("Reading time", f"{numbers.reading_time_seconds:.0f} s")

st.caption("Automatic analysis can be wrong. Do not paste sensitive information.")
