import numpy as np

import transformers
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer


class NaiveBayesTFiDF(BaseEstimator):
    
    def __init__(
        self,
        *,
        use_tfidf: bool = False,
        use_count: bool = False,
        use_in: bool = True,
        use_idf: bool = False,
        use_special_tokens: bool = True
    ) -> None:
        super().__init__()

        assert sum([use_tfidf, use_count, use_in]) == 1, "Only one of use_tfidf, use_count, use_in can be true"

        self.use_tfidf = use_tfidf
        self.use_count = use_count
        self.use_in = use_in
        self.use_idf = use_idf
        self.use_special_tokens = use_special_tokens
        
        self.model = MultinomialNB()
        self.tfidf = TfidfTransformer(use_idf=use_idf)
        self.tokenizer = transformers.BertTokenizer.from_pretrained('Maltehb/danish-bert-botxo')
        self.features: list = None
    
    def set_params(
        self,
        use_tfidf: bool = False,
        use_count: bool = False,
        use_in: bool = True,
        use_idf: bool = False,
        use_special_tokens: bool = True
    ):
        assert sum([use_tfidf, use_count, use_in]) == 1, "Only one of use_tfidf, use_count, use_in can be true"

        self.use_tfidf = use_tfidf
        self.use_count = use_count
        self.use_in = use_in
        self.use_idf = use_idf
        self.use_special_tokens = use_special_tokens

        self.tfidf.set_params(use_idf=use_idf)

        return self

    def transform_to_features(self, raw_text_docs: np.ndarray) -> np.ndarray:
        tokenized_text = [
            self.tokenizer.encode_plus(
                text,
                add_special_tokens=self.use_special_tokens
            )['input_ids'][1:-1] for text in raw_text_docs
        ]

        if self.features is None:
            features = set()
            for text in tokenized_text:
                features.update(text)
            self.features = sorted(features)

        x_data = np.zeros((len(raw_text_docs), len(self.features)))

        for j, token in enumerate(self.features):
            for i, text in enumerate(tokenized_text):
                if self.use_in:
                    x_data[i, j] = int(token in text)
                else:
                    x_data[i, j] = text.count(token)
        
        return x_data
    
    def fit(self, raw_text_docs: np.ndarray, y_data: np.ndarray) -> None:

        self.features = None
        x_data = self.transform_to_features(raw_text_docs)

        if self.use_tfidf:
            self.tfidf.fit(x_data, y_data)
            x_data = self.tfidf.transform(x_data)

        self.model.fit(x_data, y_data)

        return self

    def predict(self, raw_text_docs: np.ndarray) -> np.ndarray:
        
        x_data = self.transform_to_features(raw_text_docs)
        if self.use_tfidf:
            x_data = self.tfidf.transform(x_data)

        return self.model.predict(x_data)


if __name__ == '__main__':

    import pandas as pd
    from sklearn.model_selection import cross_validate


    data_path = './data/labelled_validation_data.tsv'
    df = pd.read_csv(data_path, sep='\t')

    x_data = df['text'].to_numpy()
    y_data = df['is_generated'].to_numpy()

    model = NaiveBayesTFiDF()
    model.fit(x_data, y_data)

    for kwargs in [
        {},
        {'use_count': True, 'use_in': False},
        {'use_tfidf': True, 'use_idf': True, 'use_in': False},
        {'use_tfidf': True, 'use_idf': False, 'use_in': False}
    ]:
        print(f"kwargs: {kwargs}")
        out = cross_validate(
            NaiveBayesTFiDF(**kwargs),
            x_data,
            y_data,
            cv=2,
            scoring='accuracy',
            return_train_score=True,
            verbose=1
        )

        for key, value in out.items():
            print(f"{key}: {np.mean(value)}")
2
