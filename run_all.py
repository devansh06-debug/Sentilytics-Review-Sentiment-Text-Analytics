"""Run the whole project: data -> analytics -> NLP -> deep learning."""
from src import analytics, data_gen, deep_learning, nlp


def banner(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


if __name__ == "__main__":
    banner("STEP 0: GENERATE DATA");              data_gen.main()
    banner("STEP 1: DATA ANALYTICS");             analytics.main()
    banner("STEP 2: NLP (TF-IDF + LOGISTIC REG)"); nlp.main()
    banner("STEP 3: DEEP LEARNING (LSTM)");       deep_learning.main()
