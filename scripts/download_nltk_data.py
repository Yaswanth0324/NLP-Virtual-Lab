import os
import nltk


def main():
    # Allow custom NLTK data directory (useful on Render)
    nltk_data_dir = os.environ.get("NLTK_DATA")
    if nltk_data_dir:
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)

    packages = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker",
        "maxent_ne_chunker_tab",
        "words",
        "wordnet",
        "omw-1.4",
        "vader_lexicon",
    ]

    for pkg in packages:
        try:
            nltk.download(pkg)
        except Exception:
            # Keep going to download what we can
            pass


if __name__ == "__main__":
    main()
