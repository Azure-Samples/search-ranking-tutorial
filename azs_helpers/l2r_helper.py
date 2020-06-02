from collections.abc import MutableMapping
from itertools import zip_longest

import numpy as np
import pandas as pd
from IPython.display import display, HTML
from matplotlib import pyplot as plt
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import minmax_scale, QuantileTransformer

def scores_to_rankings(
    predictions, query_doc_counts, X_test, y_test, verbose=False
):
    """Given ranking scores from XGBoost and query group info,
    convert the scores for each query group into numeric rankings.

    Example: Assume the following:
    A query with 5 documents has the following ground truth ordering:
    [5, 4, 3, 2, 1]

    The XGBoost scores for this query look like this, and can range from (-inf, inf):
    [0.8, 1.2,  0.5, -1.3, -0.5]
    where the largest number (1.2) denotes the most relevant, and smallest (-1.3) denotes the least relevant.

    These scores need to be converted back into judgment values for this query:
    [4, 5, 3, 1, 2]

    Since queries in this dataset are not guaranteed to be the same length, we may end up with
    a result that looks like this.
    [ query1 (5 docs): [5, 3, 4, 2, 1]
      query2 (4 docs): [4, 2, 1, 3]
      query3 (3 docs): [1, 3, 2]
    ]

    This example above is what we are referring to with the term 'dynamic shape'.

    Args:
        predictions (1-d ndarray): Ranking scores for all documents in the test data.
        query_doc_counts (list): Number of documents for each query in the test data.
        Queries are ordered by when they are first seen in the test data.
        E.g. [5, 8, 10] -> means that query1 has 5 docs, query2 has 8 docs, and query3 has 10 docs.
        X_test (pandas.Dataframe): 2-d Dataframe containing all features in the test set.
        y_test (pandas.Series): 1-d Series containing all Judgment values in the test set.
    Returns:
        model_judgments: Ranker judgments grouped by query. 2-d ndarray with dynamic shape.
        azs_judgments: Azure Cognitive Search TF-IDF scores grouped by query. Same order & shape as above.
        baseline: Ground truth judgments grouped by query. Same order & shape as above.
    """
    model_judgments = []
    azs_judgments = []
    baseline = []

    start = 0
    end = 0
    for query_id, query_doc_count in enumerate(query_doc_counts):
        # Slice the ranking scores for this query's documents.
        start = end
        end += query_doc_count
        relevance_scores = predictions[start:end]
        # Find the indices that will sort the query scores in ascending order
        # https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy-without-sorting-array-twice/
        sorted_idxs = relevance_scores.argsort()
        judgments = np.empty_like(sorted_idxs)
        # Judgment labels for each query's documents must be sorted in ascending order
        # for correct score -> judgment value conversion.
        judgments[sorted_idxs] = y_test[start:end].sort_values()
        model_judgments.append(judgments)

        azs_scores = X_test["@search.score"][start:end]
        sorted_azs_idxs = azs_scores.argsort()
        azs_judgments_for_query = np.empty_like(sorted_azs_idxs)
        azs_judgments_for_query[sorted_azs_idxs] = y_test[start:end].sort_values()
        azs_judgments.append(azs_judgments_for_query)

        baseline.append(y_test[start:end])

        if verbose:
            print(f"--------- DEBUG values for query ID {query_id} ---------")
            print("------------ Current predictions ------------")
            print(relevance_scores)
            print("-------------- Sorted indices ------------")
            print(sorted_idxs)
            print("-------------- Judgment values ---------------")
            print(judgments)

    return np.array(model_judgments), np.array(azs_judgments), np.array(baseline)


def customize_features(features):
    # adding a query_id column will help us group the data per query later in training
    if 'query_id' not in features:
        features['query_id'] = features.groupby('query').ngroup()

    if ('title_en_us' in features):
        features['title_en_us'] = features['title_en_us'].map(lambda title: str(title))
        features['title_length_in_words'] = features['title_en_us'].map(lambda title: len(title.split()))
        features['title_length_in_chars'] = features['title_en_us'].map(lambda title: len(title))

    # Discard the columns that we no longer need
    features = features.drop(['query', 'url', 'title_en_us'], axis=1, errors='ignore')
    return features


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("method must be 0 or 1.")
    return 0.0


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max


# function used to flatten nested dictionaries. Useful for flattening features
def flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_range_ndcg(ndcgs, k_start, k_end):
    results = []

    for k in range(k_start, k_end):
        count = 0
        ndcg_sum = 0
        for ndcg_input in ndcgs:
            # current_ndcg = ndcg_at_k(ndcg_input, k)
            # Ground truth for each query always ranges from highest (can be up to 10) down to lowest (1)
            # judgment value.
            # sklearn's ndcg_score can score either a batch of queries or a single query at a time. Hence,
            # it expects a shape of (n_samples, n_labels)
            ground_truth_judgments = np.asarray([list(range(len(ndcg_input), 0, -1))])
            predicted_scores = np.asarray([ndcg_input])
            current_ndcg = ndcg_score(ground_truth_judgments, predicted_scores, k)
            # print(current_ndcg)
            ndcg_sum += current_ndcg
            # no point in using ranking results where all documents have a grade of zero
            if current_ndcg != 0:
                count += 1

        results.append(ndcg_sum / count)
    return results


def evaluate_ndcg(k_start, k_end, plot=False, show_lift=False, **ranking_results):
    d = {}
    for key, value in ranking_results.items():
        d[key] = get_range_ndcg(value, k_start, k_end + 1)

    df = pd.DataFrame(data=d)
    df.set_axis(range(k_start, k_end + 1), axis=0, inplace=True)

    if plot:
        display(df)
        fig, ax = plt.subplots()
        ax.set_xlabel("k (position)")
        ax.set_ylabel("NDCG")
        ax.set_title("NDCG @ k")
        df.plot(ax=ax)

        if show_lift:
            lift = df.pct_change(axis=1) * 100
            lift.set_axis(range(k_start, k_end + 1), axis=0, inplace=True)

            fig, ax = plt.subplots()
            ax.set_xlabel("k (position)")
            ax.set_ylabel("Percent Gain")
            ax.set_title("Lift Percentage @ k")
            lift.plot(marker="x", ax=ax)

    return df


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks
    Recipe from Python 3 docs. Beware: if the last chunk
    is smaller than previous chunks, the caller is responsible
    for truncating the list, lest they get a bunch of repeating
    'None' objects."""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def escape_query(query):
    characters = ['\\', '+', '-', '&', '|', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '/']
    for character in characters:
        query = query.replace(character, "\\{0}".format(character))

    return query


def get_search_results(service, query, urls_filter):
    search_request_body = {
        "search": escape_query(query),
        "featuresMode": "enabled",
        "select": "title_en_us, url_en_us, sectionCount, tableCount, normalized_pageview",
        "searchFields": "body_en_us, description_en_us, title_en_us, apiNames, urlPath, searchTerms, keyPhrases_en_us",
        "scoringStatistics": "global",
        "sessionId" : "my_session",
        "top" : 20
    }

    if len(urls_filter) > 0:
        search_request_body["filter"] = " or ".join(f"url_en_us eq '{url}'" for url in urls_filter)

    return service.search(search_request_body)


def normalize_features(features):
    query_groups = features.groupby('query_id')

    for feature in features.columns:
        if feature not in['normalized_pageview', 'grade', 'query_id'] and features.dtypes[feature] in [float, int]:
            features[feature] = query_groups[feature].transform(lambda x: minmax_scale(x.astype(float)))

    if 'normalized_pageview' in features.columns:
        features['normalized_pageview'] = query_groups['normalized_pageview'].transform(
            lambda x: np.concatenate(
                QuantileTransformer(n_quantiles=min(len(x), 4)).fit_transform(x.values.reshape(-1,1))
            )
        )

    return features

def show_ndcg_results(ndcg_results, k_start, k_end, plot=True):
    results_df = pd.concat(ndcg_results)
    results_grouped = results_df.groupby(level=0)
    num_results = len(ndcg_results)

    results_mean = results_grouped.mean()
    results_mean.set_axis(range(k_start, k_end + 1), axis=0, inplace=True)
    print(f"----------- NDCG mean across all {num_results} folds --------------")
    display(results_mean)

    results_std = results_grouped.std()
    results_std.set_axis(range(k_start, k_end + 1), axis=0, inplace=True)

    results_lift = results_mean.pct_change(axis=1) * 100
    results_lift.set_axis(range(k_start, k_end + 1), axis=0, inplace=True)

    if plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("k (position)")
        ax.set_ylabel("NDCG")
        ax.set_title("NDCG @ k")
        results_mean.plot(yerr=results_std, marker="x", ax=ax)

        fig, ax = plt.subplots()
        ax.set_xlabel("k (position)")
        ax.set_ylabel("Percent Gain")
        ax.set_title("Lift Percentage @ k")
        results_lift.plot(marker="x", ax=ax)


def pretty_print(df):
    return display( HTML( df.to_html().replace("\\n","<br>") ) )
