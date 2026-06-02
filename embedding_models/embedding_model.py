from sentence_transformers import SentenceTransformer, models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import get_project_root
import os

class Embedding_Model:

    def __init__(self, retriever):
        if retriever not in ["multilingual-e5-large", "multilingual-e5-base", "m3e-large", "m3e-base", "all-mpnet-base-v2", "bm25"]:
            raise ValueError("Invalid retriever")
        if retriever != "bm25":
            model_path = os.path.join(get_project_root(), "embedding_models", retriever)
            word_embedding_model = models.Transformer(model_path)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            self.model = retriever

    def get_scores(self, target, corpus, topk=5):
        if self.model == "bm25":
            from rank_bm25 import BM25Okapi
            import jieba
            corpus_embedding = [jieba.lcut(doc) for doc in corpus]
            bm25 = BM25Okapi(corpus_embedding)

            if len(target) == 1:
                target = target[0]
                target_embedding = jieba.lcut(target)
                doc_scores = bm25.get_scores(target_embedding)
                top_indexes = np.argsort(doc_scores)[::-1][:topk]
                similarities = [doc_scores[i] for i in top_indexes]
                pairs = [(similarity, index) for similarity, index in zip(similarities, top_indexes)]
                return pairs
            else:
                raise ValueError("bm25 一次只能查询一个句子")
        else:
            target_embedding = self.model.encode(target)
            corpus_embedding = self.model.encode(corpus)
            similarities = np.squeeze(cosine_similarity(target_embedding, corpus_embedding)).tolist()
            pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
            pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)[:topk]
            return pairs_sorted

if __name__ == '__main__':
    embedding_model = Embedding_Model("multilingual-e5-large")
    target = "我很开心"
    corpus = ["我不开心", "我有点开心", "你很快乐"]
    scores = embedding_model.get_scores([target], corpus)
    for score, index in scores:
        print(f"Similarity: {score:.4f}, Sentence: {corpus[index]}")

    embedding_model = Embedding_Model("bm25")
    target = "我很开心"
    corpus = ["我不开心", "我有点开心", "你很快乐"]
    scores = embedding_model.get_scores([target], corpus)
    for score, index in scores:
        print(f"Similarity: {score:.4f}, Sentence: {corpus[index]}")