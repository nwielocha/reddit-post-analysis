import praw
import nltk


def save_comments_to_file(comments_list):
    with open("all_comments.txt", 'w') as fp:
        for comment in comments_list:
            fp.write("%s\n" % comment.body)
        print("All comments saved in all_comments.txt")


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


def main():
    """
    # Uncomment this section if first run

    reddit = praw.Reddit(user_agent=True,
                         client_id="Y3R6ijGXxjPfCFtnM8sWMw",
                         client_secret="DkaLTGy4-gPUwjpQaruHbB1h_wTOLQ",
                         username="Imaginary_Pickle_556",
                         password="%\"vHQxX?gKB!j9&")

    url = "https://www.reddit.com/r/coolguides/comments/172z892/a_cool_guide_on_the_human_cost_of_the/"

    post = reddit.submission(url=url)

    post.comments.replace_more(limit=None)
    all_comments = post.comments.list()

    save_comments_to_file(all_comments)

    nltk.download("all")

    """

    filename = "all_comments.txt"
    data = read_file(filename)
    tokens = tokenize(data)
    print("Liczba słów po tokenyzacji: ", len(tokens))

    filtered_tokens = delete_stopwords(tokens)
    print("Liczba słów po usunięciu stopwords: ", len(filtered_tokens))

    lemmatized_tokens = lemmatize(filtered_tokens)
    print("Liczba słów po lematyzacji: ", len(lemmatized_tokens))


if __name__ == "__main__":
    main()
