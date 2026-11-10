# pandas and scikit-learn

## pandas

```bash
pip install "textinsight-nlp[pandas]"
```

```python
import pandas as pd
import textinsight.integrations.pandas_accessor  # registers .ti

df = pd.DataFrame({"review": ["camera mast hai", "bilkul bekaar hai"]})

df["review"].ti.sentiment()  # a DataFrame with label and score columns
df["review"].ti.keywords()
df["review"].ti.aspects()
df["review"].ti.stats()
df["review"].ti.redact()
df["review"].ti.analyze(["sentiment", "keywords"]).to_html("report.html")
```

Missing values are treated as empty strings, and results keep the original index.

## scikit-learn

```bash
pip install "textinsight-nlp[sklearn]"
```

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from textinsight.integrations.sklearn import TextInsightFeatures

model = make_pipeline(TextInsightFeatures(language="hi-en"), LogisticRegression())
model.fit(texts, labels)
```

The features are the sentiment score, word count, sentence count, type-token ratio, average word
length and average sentence length.
