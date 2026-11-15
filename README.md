# GoEmotions_NLP_Phase01
NLP Phase 1 on GoEmotions (54,263 Reddit comments, 28 emotion labels): dataset loading, corpus statistics, tokenizer comparison, regex patterns and a reproducible cleaning pipeline (65.8% vocabulary reduction) built with spaCy and NLTK.

# GoEmotions_NLP_Phase02
Turn the Phase 1 cleaned corpus (text_clean, train split: 43,410 comments,
380,557 tokens) into numerical representations and compare them.

Count-based representations
- Bag of Words: unigrams, 20,026 features
- TF-IDF: 1-2-grams, min_df=3, 22,910 features
- Vocabulary growth and sparsity compared for n = 1, 2, 3
- Hand-worked TF-IDF weight and bigram, checked against scikit-learn

Learned embeddings (gensim, 100 dimensions, window 5, min_count 5, 30 epochs)
- Word2Vec CBOW and Skip-gram with identical hyper-parameters
- fastText with character 3-6-grams and an OOV / misspelling demonstration
- Probing: nearest neighbours, similar/unrelated pairs, one analogy,
  count-based vs learned comparison

Packaging for Phase 3
- src/representations.py: single definition of every representation
- src/build_representations.py: fit once and save to models/ (+ manifest.json)
- src/vectorize.py: vectorize(raw_text, method) interface; loads saved models,
  never retrains
- notebooks/03_word_representations.ipynb: Parts A-D with outputs
- Deterministic training (workers=1, fixed seed, stable hash)

Everything is fitted on the train split only.
