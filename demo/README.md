---
title: TextInsight Demo
colorFrom: blue
colorTo: gray
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# TextInsight demo

Paste English or Hinglish text to see sentiment, aspects, keywords, a summary, redaction and
statistics. Everything runs inside the app, with no external model or API calls.

Library: [textinsight-nlp on PyPI](https://pypi.org/project/textinsight-nlp/)

## Run it locally

```bash
pip install -r demo/requirements.txt
```

```bash
streamlit run demo/app.py
```

## Deploy on Hugging Face Spaces

1. Publish a TextInsight release that includes `aspects()`, which the app needs.
2. Create a new Space with the Streamlit SDK.
3. Copy `app.py`, `requirements.txt` and this `README.md` into it.
