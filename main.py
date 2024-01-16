import os
import praw
import nltk
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime


def save_comments_to_file(comments_list):
    with open("all_comments.txt", 'w') as fp:
        for comment in comments_list:
            fp.write("%s\n" % comment.body)
        print("All comments saved in all_comments.txt")


def get_comments(filename):
    with open(filename, 'r') as file:
        comments_list = [line.strip() for line in file]

    def filter_comments(comments_list):
        filtered_comments = [
            c for c in comments_list if c not in ('', "[deleted]")]
        filtered_comments = [f for f in filtered_comments if "http" not in f]
        return filtered_comments
    return filter_comments(comments_list)


def read_file(filename):
    file = open("all_comments.txt", 'r')
    data = file.read()
    return data


def tokenize(data):
    tokens = nltk.word_tokenize(data)
    tokens = [t for t in tokens if t.isalpha()]
    return tokens


def delete_stopwords(tokens):
    stopwords = nltk.corpus.stopwords.words("english")
    filtered_tokens = [t for t in tokens if t.lower() not in stopwords]
    return filtered_tokens


def lemmatize(filtered_tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in filtered_tokens]
    return lemmatized_tokens


def freq_dist(lemmatized_tokens):
    fd = nltk.FreqDist(lemmatized_tokens).most_common(10)
    return fd


def wordcloud(tokens):
    stopwords = nltk.corpus.stopwords.words("english")
    print(stopwords)
    text = ' '.join(tokens)
    wordcloud = WordCloud(stopwords=stopwords).generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("wordcloud.png")


def get_sentiment(comments_list):
    sia = SentimentIntensityAnalyzer()
    compound_scores = [sia.polarity_scores(
        comment)['compound'] for comment in comments_list]
    average_score = sum(compound_scores) / len(compound_scores)

    if average_score > 0:
        return f"Positive: {average_score}"
    elif average_score < 0:
        return f"Negative: {average_score}"
    else:
        return "Neutral"


def main():
    load_dotenv()
    print(os.getenv("CLIENT_ID"))
    # Uncomment this section if first run

    reddit = praw.Reddit(user_agent=True,
                         client_id=os.getenv("CLIENT_ID"),
                         client_secret=os.getenv("CLIENT_SECRET"),
                         username=os.getenv("USERNAME"),
                         password=os.getenv("PASSWORD"))

    url = "https://www.reddit.com/r/coolguides/comments/172z892/a_cool_guide_on_the_human_cost_of_the/"

    post = reddit.submission(url=url)

    post.comments.replace_more(limit=None)
    all_comments = post.comments.list()

    save_comments_to_file(all_comments)

    nltk.download("all")

    filename = "all_comments.txt"
    comments_list = get_comments(filename)
    print(len(comments_list))

    # data = read_file(filename)
    data = ''.join(comments_list)
    tokens = tokenize(data)
    print("Liczba słów po tokenyzacji: ", len(tokens))

    filtered_tokens = delete_stopwords(tokens)
    print("Liczba słów po usunięciu stopwords: ", len(filtered_tokens))

    lemmatized_tokens = lemmatize(filtered_tokens)
    print("Liczba słów po lematyzacji: ", len(lemmatized_tokens))

    fd = freq_dist(lemmatized_tokens)

    words, freq = zip(*fd)
    plt.figure(figsize=(10, 6))
    plt.bar(words, freq, color="skyblue")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Frequency Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("freq_dist.png")
    # plt.show()

    wordcloud(lemmatized_tokens)

    # text = nltk.Text(data)
    # print(text.concordance("hamas", lines=5))

    finder = nltk.collocations.TrigramCollocationFinder.from_words(
        lemmatized_tokens)
    trigram_fd = finder.ngram_fd.most_common(5)
    tuples, freq = zip(*trigram_fd)
    words = [' '.join(t) for t in tuples]
    plt.figure(figsize=(10, 6))
    plt.bar(words, freq, color="skyblue", align="center")
    plt.xticks(rotation=30, ha='right')
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Frequent Three-Word Combinations")
    plt.tight_layout()
    plt.savefig("trigram_freq_dist.png")

    overal_sentiment = get_sentiment(comments_list)
    print(overal_sentiment)


if __name__ == "__main__":
    main()
