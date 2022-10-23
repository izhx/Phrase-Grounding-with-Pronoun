"""
conda create -n=hfnc python=3.6
conda activate hfnc
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .

python -m spacy download en_core_web_md

"""

import time
import json


def hf_coref():
    import spacy
    import neuralcoref

    # Load your usual SpaCy model (one of SpaCy English models)
    nlp = spacy.load('en_core_web_md')  # en

    # Add neural coref to SpaCy's pipe
    neuralcoref.add_to_pipe(nlp)

    # You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
    # doc = nlp('My sister has a dog. She loves him.')

    # words = ['My', 'sister', 'has', 'a', 'dog', '.', 'She', 'loves', 'him', '.']
    # doc = nlp(' '.join(words))

    def predict_coref(sentences, index):
        """
        Predict coreference in a document.
        """
        tokens = [t for s in sentences for t in s["text"]]
        # tokens = words
        start_char_to_word, end_char_to_word = dict(), dict()
        for i, token in enumerate(tokens):
            start_char = sum(len(t) for t in tokens[:i]) + i
            end_char = start_char + len(token)
            start_char_to_word[start_char] = i
            end_char_to_word[end_char] = i

        doc = nlp(' '.join(tokens))
        clusters = list()
        for c in doc._.coref_clusters:
            spans = list()
            for m in c.mentions:
                flag = False
                if index == 34044 and m.start_char == 256:
                    start = start_char_to_word[255]  # 'm
                    end = end_char_to_word[m.end_char]
                    start, end = start - 1, end - 1  # to "i"
                elif index == 36863 and m.start_char == 116:
                    start = start_char_to_word[115]  # Id
                    end = end_char_to_word[m.end_char]
                elif index == 34806 and m.end_char == 404:
                    start = start_char_to_word[m.start_char]
                    end = end_char_to_word[405]  # shes
                else:
                    flag = True
                    start = start_char_to_word[m.start_char]
                    end = end_char_to_word[m.end_char]
                if flag:
                    assert ' '.join(tokens[start: end + 1]) == m.text
                spans.append((start, end))
            clusters.append(spans)
        return clusters

    with open("data/res3.jsonlines") as file:
        data = list()
        for i, line in enumerate(file):
            ins = json.loads(line)
            # if ins['id'] not in {34806, 34044, 36863}:
            #     continue
            # try:
            ins['predicted_clusters'] = predict_coref(ins['dialog'], ins['id'])
            # except Exception:
            #     print(i, ins['id'])
            data.append(ins)
            if i % 500 == 0:
                print(i, time.asctime(time.localtime(time.time())))
        print(file.name)

    with open("data/res3.jsonlines.hf", "w") as file:
        for ins in data:
            file.write(json.dumps(ins) + "\n")
        print(file.name)


def evaulate(filename: str = "data/res3.jsonlines.hf"):
    from allennlp_models.coref.metrics.conll_coref_scores import Scorer
    from src.coref_reader import span_clusters

    scorers = [Scorer(m) for m in (Scorer.muc, Scorer.b_cubed, Scorer.ceafe)]

    def _get(clusters):
        clusters = [tuple(tuple(m) for m in c) for c in clusters]
        mention_to_cluster = {}
        for cluster in clusters:
            for mention in cluster:
                mention_to_cluster[mention] = cluster
        return clusters, mention_to_cluster

    with open(filename) as file:
        for line in file:
            ins = json.loads(line)
            sentences = [sentence["text"] for sentence in ins['dialog']]
            clusters = span_clusters(ins['clusters'], ins['text_ref'], sentences)
            gold_clusters, mention_to_gold = _get(clusters)
            predicted_clusters, mention_to_predicted = _get(ins['predicted_clusters'])
            for scorer in scorers:
                scorer.update(
                    predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold
                )
        print(file.name)

    metrics = (lambda e: e.get_precision(), lambda e: e.get_recall(), lambda e: e.get_f1())
    precision, recall, f1_score = tuple(
        sum(metric(e) for e in scorers) / len(scorers) for metric in metrics
    )
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1_score)


if __name__ == "__main__":
    # hf_coref()
    evaulate("data/res3.jsonlines.hf")
    evaulate("data/res3.jsonlines.al")

"""
hf
Precision: 0.3873170933408809
Recall: 0.3687151484155409
F1: 0.37605741599144343
"""
