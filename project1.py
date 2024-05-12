"""
EECS 445 - Winter 2024

Project 1 main file.
"""


import itertools
import string
import warnings
import nltk
from sklearn.metrics import RocCurveDisplay
from typing import Any
from sklearn.metrics import confusion_matrix
import numpy as np
import numpy.typing as npt
import pandas as pd
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from helper import *


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)




def extract_word(input_string: str) -> list[str]:
    """Preprocess review text into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along
    whitespace. Return the resulting array.

    Example:
        > extract_word("I love EECS 445. It's my favorite course!")
        > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Args:
        input_string: text for a single review

    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    # TODO: Implement this function
    for punctuation_char in string.punctuation:
        input_string = input_string.replace(punctuation_char, " ")
    
    # Convert to lowercase and split along whitespace
    lower_string = input_string.lower()
    word_list = lower_string.split()

    return(word_list)

def extract_word_new(input_string: str) -> list[str]:
    """Preprocess review text into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along
    whitespace. Return the resulting array.

    Example:
        > extract_word("I love EECS 445. It's my favorite course!")
        > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Args:
        input_string: text for a single review

    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    # TODO: Implement this function
    
    #stemmer = PorterStemmer()
    stop_words = stopwords.words('english')
    #punc_new = string.punctuation.replace("'", "")
    for punctuation_char in string.punctuation:
        input_string = input_string.replace(punctuation_char, " ")  
    # Convert to lowercase and split along whitespace
    lower_string = input_string.lower()
    word_list = lower_string.split()
    #word_list = [stemmer.stem(token) for token in word_list]
    word_list = [word for word in word_list if word not in stop_words]
    return(word_list)


def extract_dictionary(df: pd.DataFrame) -> dict[str, int]:
    """
    Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words mapping from each
    distinct word to its index (ordered by when it was found).

    Example:
        Input df:

        | reviewText                    | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

        The output should be a dictionary of indices ordered by first occurence in
        the entire dataset. The index should be autoincrementing, starting at 0:

        {
            it: 0,
            was: 1,
            the: 2,
            best: 3,
            of: 4,
            times: 5,
            blurst: 6,
        }

    Args:
        df: dataframe/output of load_data()

    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    count = 0
    for review_text in df['reviewText']:
        # Split the review text into words
        words = extract_word(review_text)
        
        # Iterate over each word
        for word in words:
            # Check if the word is already in the dictionary
            if word not in word_dict:
                # If the word is not in the dictionary, add it and assign an index
                word_dict[word] = count
                count += 1
    return word_dict


def extract_dictionary_new(df: pd.DataFrame) -> dict[str, int]:
    """
    Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words mapping from each
    distinct word to its index (ordered by when it was found).

    Example:
        Input df:

        | reviewText                    | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

        The output should be a dictionary of indices ordered by first occurence in
        the entire dataset. The index should be autoincrementing, starting at 0:

        {
            it: 0,
            was: 1,
            the: 2,
            best: 3,
            of: 4,
            times: 5,
            blurst: 6,
        }

    Args:
        df: dataframe/output of load_data()

    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    count = 0
    for review_text in df['reviewText']:
        # Split the review text into words
        words = extract_word(review_text)
        
        # Iterate over each word
        for word in words:
            # Check if the word is already in the dictionary
            if word not in word_dict:
                # If the word is not in the dictionary, add it and assign an index
                word_dict[word] = count
                count += 1
    return word_dict



def generate_feature_matrix(
    df: pd.DataFrame, word_dict: dict[str, int]
) -> npt.NDArray[np.float64]:
    """
    Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review. For each review, extract a token
    list and use word_dict to find the index for each token in the token list.
    If the token is in the dictionary, set the corresponding index in the review's
    feature vector to 1. The resulting feature matrix should be of dimension
    (# of reviews, # of words in dictionary).

    Args:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices

    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    reviews = df["reviewText"]
    for i in range(0 , number_of_reviews):
        token_list = extract_word(reviews[i])
        for j in range(0, len(token_list)):
            if token_list[j] in word_dict:
                feature_matrix[i][word_dict[token_list[j]]] = 1
    return feature_matrix


def performance(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.int64],
    metric: str = "accuracy",
) -> np.float64:
    """
    Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Args:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')

    Returns:
        the performance as an np.float64
    """
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.


def cv_performance_new(
    clf: LinearSVC | SVC,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,   
):
    skf = StratifiedKFold(n_splits= k, shuffle= False)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]  # Select training and test data
        y_train, y_test = y[train_index], y[test_index]  # Select training and test labels
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        cm = confusion_matrix(y_test, prediction)
        result = 0
        for i in range (0, len(cm)):
            result += cm[i][i]
        return result / float(len(prediction))

def cv_performance(
    clf: LinearSVC | SVC,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,
    metric: str = "accuracy",
) -> float:
    """
    Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.

    Args:
        clf: an instance of LinearSVC() or SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1, -1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')

    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful

    scores = []
    skf = StratifiedKFold(n_splits= k, shuffle= False)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]  # Select training and test data
        y_train, y_test = y[train_index], y[test_index]  # Select training and test labels
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        TP_count = 0
        FP_count = 0
        TN_count = 0
        FN_count = 0
        result = 0
       
        for i in range(0, len(prediction)):
            if prediction[i] == -1:
                if y_test[i] == -1:
                    TN_count += 1
                else:
                    FN_count += 1
            if prediction[i] == -1:
                if y_test[i] == -1:
                    FP_count += 1
                else:
                    TP_count += 1

        if metric == 'accuracy':
            result = (TP_count + TN_count) / float(len(prediction))
        elif metric == 'f1-score':
            precision = TP_count / float(TP_count + FP_count)
            recall = TP_count / float(TP_count + FN_count)
            if precision + recall == 0:
                result = 0
            else:
                result = 2 * (precision * recall) / (precision + recall)
        elif metric == 'auroc':
            predict = clf.decision_function(X_test)
            result = metrics.roc_auc_score(y_test, predict)
        elif metric == 'precision':
            result = TP_count / float(TP_count + FP_count)
        elif metric == 'sensitivity':
            result = TP_count / float(TP_count + FN_count)
        elif metric == 'specificity':
            result = TN_count / float(TN_count + FP_count)
        scores.append(result)
    return np.array(scores).mean()

def metric_calc(
    clf: LinearSVC | SVC,
    X_test: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.float64],
    prediction: npt.NDArray[np.float64],
    metric: str
) -> float:
    scores = []
    TP_count = 0
    FP_count = 0
    TN_count = 0
    FN_count = 0
    result = 0
       
    for i in range(0, len(prediction)):
        if prediction[i] == -1:
            if y_test[i] == -1:
                TN_count += 1
            else:
                FN_count += 1
        else:
            if y_test[i] == -1:
                FP_count += 1
            else:
                TP_count += 1
    if metric == 'accuracy':
        result = (TP_count + TN_count) / float(len(prediction))
    elif metric == 'f1-score':
        precision = TP_count / float(TP_count + FP_count)
        recall = TP_count / float(TP_count + FN_count)
        if precision + recall == 0:
            result = 0
        else:
            result = 2 * (precision * recall) / (precision + recall)
    elif metric == 'auroc':
        predict = clf.decision_function(X_test)
        result = metrics.roc_auc_score(y_test, predict)
    elif metric == 'precision':
        result = TP_count / float(TP_count + FP_count)
    elif metric == 'sensitivity':
        result = TP_count / float(TP_count + FN_count)
    elif metric == 'specificity':
        result = TN_count / float(TN_count + FP_count)
    
    return (result)
    
    


def select_param_linear(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,
    metric: str = "accuracy",
    C_range: list[float] = [],
    loss: str = "hinge",
    penalty: str = "l2",
    dual: bool = True,
) -> float:
    """
    Search for hyperparameters from the given candidates of linear SVM with
    best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1")

    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    # HINT: You should be using your cv_performance function here
    # to evaluate the performance of each SVM
    scores = []
    for C in C_range:
        clf = LinearSVC(loss='hinge', penalty='l2', dual=True, random_state=445, C=C)
        score = cv_performance(clf, X, y, k, metric)
        scores.append(score)
    max_index = scores.index(max(scores))
    print(max(scores))
    return(C_range[max_index])    


def plot_weight(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    penalty: str,
    C_range: list[float],
    loss: str,
    dual: bool,
) -> None:
    """
    Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: string for penalty type to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: string for loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor

    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []
    # TODO: Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y
    for C in C_range:
        clf = LinearSVC(loss= loss, penalty= penalty, dual=dual, random_state=445, C=C)
        clf.fit(X, y)
        theta_vec = clf.coef_[0]
        l0_norm = np.count_nonzero(theta_vec)
        norm0.append(l0_norm)
    

    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()


def select_param_quadratic(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,
    metric: str = "accuracy",
    param_range: npt.NDArray[np.float64] = [],
) -> tuple[float, float]:
    """
    Search for hyperparameters from the given candidates of quadratic SVM
    with best k-fold CV performance.

    Sweeps different settings for the hyperparameters of a quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.

    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    best_C_val = 0.0
    best_r_val = 0.0
    performance = []
    for i in range(0, len(param_range)):
        c = param_range[i][0]
        r = param_range[i][1]
        svc = SVC(kernel="poly", degree=2, C=c, coef0=r, gamma="auto", random_state=445)
        score = cv_performance(svc, X, y, k, metric)
        performance.append(score)
    
    max_index = np.argmax(performance)

    best_C_val = param_range[max_index][0]
    best_r_val = param_range[max_index][1]
   
    return best_C_val, best_r_val


def train_word2vec(filename: str) -> Word2Vec:
    """
    Train a Word2Vec model using the Gensim library.

    First, iterate through all reviews in the dataframe, run your extract_word() function
    on each review, and append the result to the sentences list. Next, instantiate an
    instance of the Word2Vec class, using your sentences list as a parameter and using workers=1.

    Args:
        filename: name of the dataset csv

    Returns:
        created Word2Vec model
    """
    df = load_data(filename)
    sentences = []
    for review in df['reviewText']:
        # Process each review using the extract_word() function
        processed_review = extract_word(review)        
        # Append the processed review to the sentences list
        sentences.append(processed_review)
    
    model = Word2Vec(sentences, workers = 1)
    
    return model


def compute_association(filename: str, w: str, A: list[str], B: list[str]) -> float:
    """
    Args:
        filename: name of the dataset csv
        w: a word represented as a string
        A: set of English words
        B: set of English words

    Returns:
        association between w, A, and B as defined in the spec
    """
    model = train_word2vec(filename)

    # First, we need to find a numerical representation for the English language words in A and B

    # TODO: Complete words_to_array()
    def words_to_array(s: list[str]) -> npt.NDArray[np.float64]:
        """Convert a list of string words into a 2D numpy array of word embeddings,
        where the ith row is the embedding vector for the ith word in the input set (0-indexed).

            Args:
                s (list[str]): List of words to convert to word embeddings

            Returns:
                npt.NDArray[np.float64]: Numpy array of word embeddings
        """
        result = []
        for i in range (0, len(s)):
            word_embedding = model.wv[s[i]]
            result.append(word_embedding)
        return result

    # TODO: Complete cosine_similarity()
    def cosine_similarity(
        array: npt.NDArray[np.float64], w: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate the cosine similarities between w and the input set.

        Args:
            array: array representation of the input set
            w: word embedding for w

        Returns:
            1D Numpy Array where the ith element is the cosine similarity between the word
            embedding for w and the ith embedding in input set
        """
        dot_product = np.dot(array, w)
        norm_w = np.linalg.norm(w)
        norm_array = np.linalg.norm(array, axis=1)
        similarities = dot_product / (norm_array * norm_w)

        return similarities

    # Although there may be some randomness in the word embeddings, we have provided the
    # following test case to help you debug your cosine_similarity() function:
    # This is not an exhaustive test case, so add more of your own!
    test_arr = np.array([[4, 5, 6], [9, 8, 7]])
    test_w = np.array([1, 2, 3])
    test_sol = np.array([0.97463185, 0.88265899])
    assert np.allclose(
        cosine_similarity(test_arr, test_w), test_sol, atol=0.00000001
    ), "Cosine similarity test 1 failed"

    # TODO: Return the association between w, A, and B.
    #      Compute this by finding the difference between the mean cosine similarity between w and the words in A,
    #      and the mean cosine similarity between w and the words in B
    A_arr = words_to_array(A)
    B_arr = words_to_array(B)
    w = model.wv[w]
    A_sim = np.mean(cosine_similarity(A_arr, w))
    B_sim = np.mean(cosine_similarity(B_arr, w))

    return (A_sim - B_sim)

def generate_feature_matrix_new(
    df: pd.DataFrame, word_dict:dict[str, int]   
) -> npt.NDArray[np.float64]:
    number_of_reviews = df.shape[0]
    
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    reviews = df["reviewText"]
    for i in range(0 , number_of_reviews):
        token_list = extract_word_new(reviews[i])
        for j in range(0, len(token_list)):
            if token_list[j] in word_dict:
                feature_matrix[i][word_dict[token_list[j]]] = 1
    return feature_matrix



def main() -> None:
    # Read binary data
    # NOTE: Use the X_train, Y_train, X_test, and Y_test provided below as the training set and test set
    #       for the reviews in the file you read in.
    #
    #       Your implementations of extract_dictionary() and generate_feature_matrix() will be called
    #       to produce these training and test sets (for more information, see get_split_binary_data() in helper.py).
    #       DO NOT reimplement or edit the code we provided in get_split_binary_data().
    #
    #       Please note that dictionary_binary will not be correct until you have correctly implemented extract_dictionary(),
    #       and X_train, Y_train, X_test, and Y_test will not be correct until you have correctly
    #       implemented extract_dictionary() and generate_feature_matrix().
    filename = "data/dataset.csv"
    
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        filename=filename
    )
    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, filename=filename
    )

    # TODO: Questions 2, 3, 4, 5
    #Q2
    '''
    print(X_test)
    print(len(dictionary_binary))
    count = 0
    for i in range(0, len(X_train)):
        count += sum(X_train[i]) / float(len(X_train))
    print("The average number of non-zero features per review in the training data: ", count )
    mat = generate_feature_matrix(load_data(filename), dictionary_binary)
    num_rows = len(mat)
    num_cols = len(mat[0])  # Assuming all rows have the same number of columns
    
    max_ones_count = 0
    max_ones_column = -1
    
    for j in range(num_cols):
        ones_count = sum(mat[i][j] for i in range(num_rows))
        if ones_count > max_ones_count:
            max_ones_count = ones_count
            max_ones_column = j
    print(max_ones_column)
    dictionary_items = list(dictionary_binary.items())
    key, val = dictionary_items[max_ones_column]
    print("popular word:", key )
    
    #Q3.1
    C_values = np.logspace(-3, 3, num=7)
    
    print("Metric: precision: ")
    best_c = select_param_linear(X_train, Y_train, 5, "precision", C_values)
    print("Best C: ", best_c)
    clf = LinearSVC(loss='hinge', penalty='l2', dual=True, random_state=445, C=best_c)
    
    print("Metric: accuracy: ")
    best_c = select_param_linear(X_train, Y_train, 5, "accuracy", C_values)
    print("Best C: ", best_c)
    clf = LinearSVC(loss='hinge', penalty='l2', dual=True, random_state=445, C=best_c)
    
    print("Metric: f1-score: ")
    best_c = select_param_linear(X_train, Y_train, 5, "f1-score", C_values)
    print("Best C: ", best_c)
    clf = LinearSVC(loss='hinge', penalty='l2', dual=True, random_state=445, C=best_c)
    print("Metric: sensitivity: ")
    best_c = select_param_linear(X_train, Y_train, 5, "sensitivity", C_values)
    print("Best C: ", best_c)
    clf = LinearSVC(loss='hinge', penalty='l2', dual=True, random_state=445, C=best_c)
    cv = cv_performance(clf, X_train, Y_train)
    print("CV:", cv)
 
    print("Metric: specificity: ")
    best_c = select_param_linear(X_train, Y_train, 5, "specificity", C_values)
    print("Best C: ", best_c)
    clf = LinearSVC(loss='hinge', penalty='l2', dual=True, random_state=445, C=best_c)
    
    print("Metric: auroc: ")
    best_c = select_param_linear(X_train, Y_train, 5, "auroc", C_values)
    print("Best C: ", best_c)
    clf = LinearSVC(loss='hinge', penalty='l2', dual=True, random_state=445, C=best_c)
    
    clf = LinearSVC(loss='hinge', penalty='l2', dual=True, random_state=445, C=0.1)
    clf.fit(X_train, Y_train)
    predict = clf.predict(X_test)
    print ("acc", metric_calc(clf, X_test, Y_test, predict, "accuracy"))
    print(metric_calc(clf, X_test, Y_test, predict, "f1-score"))
    print(metric_calc(clf, X_test, Y_test, predict, "sensitivity"))
    print(metric_calc(clf, X_test, Y_test, predict, "specificity"))
    print(metric_calc(clf, X_test, Y_test, predict, "auroc"))
    print(metric_calc(clf, X_test, Y_test, predict, "precision"))

    
    #q(d)
    C_values = [10**i for i in range(-3, 1)]
    plot_weight(X_train, Y_train, "l2", C_values , "hinge", True)
    
    #q(e)
    C = 0.1
    clf = LinearSVC(loss='hinge', penalty='l2', dual=True, random_state=445, C=C)
    clf.fit(X_train, Y_train)
    theta = clf.coef_[0]
    print(np.shape(theta))
    sorted_indices = np.argsort(theta)
    keys = []
    coef = []
    five_most_negative_indices = sorted_indices[:5]
    print(five_most_negative_indices)
    five_most_negative_values = theta[five_most_negative_indices]
    sorted_new = np.argsort(-theta)
    five_most_positive_indices = sorted_new[:5]
    five_most_positive_values = theta[five_most_positive_indices]
    dic_list = list(dictionary_binary.items())
    for i in range(0, 5):
        key_n, val_n = dic_list[five_most_negative_indices[i]]
        keys.append(key_n)
        coef.append(five_most_negative_values[i])
        print("coef: ", five_most_negative_values[i]  , " word: ", key_n , "\n")
    for i in range(0, 5):
        key_p, val_p = dic_list[five_most_positive_indices[4-i]]
        keys.append(key_p)
        coef.append(five_most_positive_values[4-i])
        print ("coef: ", five_most_positive_values[4-i]  , " word: ", key_p , "\n")
    

    plt.bar(keys, coef)
    plt.xlabel('Words')
    plt.ylabel('Coefficients')
    plt.title('Bar Plot Example')

    # Show plot
    plt.show()
    
    
    #3.2(a)
    c_range = [10**i for i in range(-3, 1)]
    results = []
    max_result = 0
    max_index = 0
    for c in c_range:
        clf = LinearSVC(loss = 'squared_hinge', penalty='l1',  dual=False, random_state=445, C = c)
        result = cv_performance(clf, X_train, Y_train, 5, "auroc")
        results.append(result)
    for i in range(0, len(results)):
        if results[i] > max_result:
            max_result =  results[i] 
            max_index = i 
    best_c = c_range[max_index]
    

    clf = LinearSVC(loss = 'squared_hinge', penalty='l1',  dual=False, random_state=445, C = best_c)
    clf.fit(X_train, Y_train)
    
    predict = clf.decision_function(X_test)
    result = metrics.roc_auc_score(Y_test, predict)
    

    print("3.2(a)")
    print("Best_C ", best_c)
    print("Metric_Score ", max_result)
    print("test_result (still in doubt)", result)
    
    #3.2(b)
    plot_weight( X_train, Y_train, "l1", c_range, "squared_hinge", False)
    
    #3.3(a)
    #Grid Search
    
    grid = []
    for i in range(-2, 4):
        for j in range(-2, 4):
            grid.append([10**i, 10**j])
    print(grid)
    best_C, r = select_param_quadratic(X_train, Y_train, 5, "auroc", grid)
    print("grid_search, auroc: c and r: ", best_C, r)

    svc = SVC(kernel="poly", degree=2, C=best_C, coef0=r, gamma="auto", random_state=445)
    svc.fit(X_train, Y_train)
    prediction = svc.predict(X_test)
    score = metric_calc(svc, X_test, Y_test, prediction, "auroc")
    print("test_performance: ", score)

    #Random Search
    random = []
    for i in range(0, 25):
        random.append([10**(np.random.uniform(-2, 3)), 10**(np.random.uniform(-2, 3))])
    best_C, r = select_param_quadratic(X_train, Y_train, 5, "auroc", random)
    print("random_search, auroc: c and r: (not correct)", best_C, r)

    svc = SVC(kernel="poly", degree=2, C=best_C, coef0=r, gamma="auto", random_state=445)
    svc.fit(X_train, Y_train)
    prediction = svc.predict(X_test)
    score = metric_calc(svc, X_test, Y_test, prediction, "auroc")
    print("test_performance: ", score)
    
    
    #4.1(c)
    clf = LinearSVC(loss='hinge', penalty='l2', C=0.01, class_weight={-1:1, 1:10})
    clf.fit(X_train, Y_train)
    predict = clf.predict(X_test)
    accuracy = metric_calc(clf, X_test, Y_test, predict, "accuracy")
    f1 = metric_calc(clf, X_test, Y_test, predict, "f1-score")
    auroc = metric_calc(clf, X_test, Y_test, predict, "auroc")
    precision = metric_calc(clf, X_test, Y_test, predict, "precision")
    sensitivity  = metric_calc(clf, X_test, Y_test, predict, "sensitivity")
    specificity = metric_calc(clf, X_test, Y_test, predict, "specificity")

    print("accuracy", accuracy)
    print("f1-score", f1)
    print("auroc", auroc)
    print("precision", precision)
    print("sensitivity", sensitivity)
    print("specificity", specificity)

    #4.2(a)
    class_weight = {-1: 1, 1: 1}
    clf = LinearSVC(loss='hinge', penalty='l2', C=0.01, class_weight=class_weight)
    clf.fit(IMB_features, IMB_labels)
    predict = clf.predict(IMB_test_features)
    accuracy = metric_calc(clf, IMB_test_features, IMB_test_labels, predict, "accuracy")
    f1 = metric_calc(clf, IMB_test_features, IMB_test_labels, predict, "f1-score")
    auroc = metric_calc(clf, IMB_test_features, IMB_test_labels, predict, "auroc")
    precision = metric_calc(clf, IMB_test_features, IMB_test_labels, predict, "precision")
    sensitivity  = metric_calc(clf, IMB_test_features, IMB_test_labels, predict, "sensitivity")
    specificity = metric_calc(clf, IMB_test_features, IMB_test_labels, predict, "specificity")
    print("4.2:")
    print("accuracy", accuracy)
    print("f1-score", f1)
    print("auroc", auroc)
    print("precision", precision)
    print("sensitivity", sensitivity)
    print("specificity", specificity)
    
    #4.3(b)
    class_weight = {-1: 10, 1: 5}
    class_weight_2 = {-1: 7, 1: 5}
    clf = LinearSVC(loss='hinge', penalty='l2', C=0.01, class_weight=class_weight)
    clf2 = LinearSVC(loss='hinge', penalty='l2', C=0.01, class_weight=class_weight_2)
    clf.fit(IMB_features, IMB_labels)
    clf2.fit(IMB_features, IMB_labels)
    predict = clf.predict(IMB_test_features)
    predict_2 = clf2.predict(IMB_test_features)
    accuracy = metric_calc(clf, IMB_test_features, IMB_test_labels, predict, "accuracy")
    f1 = metric_calc(clf, IMB_test_features, IMB_test_labels, predict, "f1-score")
    auroc = metric_calc(clf, IMB_test_features, IMB_test_labels, predict, "auroc")
    precision = metric_calc(clf, IMB_test_features, IMB_test_labels, predict, "precision")
    sensitivity  = metric_calc(clf, IMB_test_features, IMB_test_labels, predict, "sensitivity")
    specificity = metric_calc(clf, IMB_test_features, IMB_test_labels, predict, "specificity")

    accuracy_2 = metric_calc(clf2, IMB_test_features, IMB_test_labels, predict_2, "accuracy")
    f1_2 = metric_calc(clf, IMB_test_features, IMB_test_labels, predict_2, "f1-score")
    auroc_2 = metric_calc(clf, IMB_test_features, IMB_test_labels, predict_2, "auroc")
    precision_2 = metric_calc(clf, IMB_test_features, IMB_test_labels, predict_2, "precision")
    sensitivity_2 = metric_calc(clf, IMB_test_features, IMB_test_labels, predict_2, "sensitivity")
    specificity_2 = metric_calc(clf, IMB_test_features, IMB_test_labels, predict_2, "specificity")
    
    better_clf = "10:5" if f1 > f1_2 else "7:5"
    print("4.3(b):")
    print(better_clf)
    print("accuracy", max(accuracy, accuracy_2))
    print("f1-score", max(f1, f1_2))
    print("auroc", max(auroc, auroc_2))
    print("precision", max(precision, precision_2))
    print("sensitivity", max(sensitivity, sensitivity_2))
    print("specificity", max(specificity, specificity_2))
    
    #4.4
    
    clf1 = LinearSVC(loss='hinge', penalty='l2', C=0.01)
    clf1.fit(IMB_features, IMB_labels)
    fpr, tpr, thresholds = metrics.roc_curve(IMB_test_labels, clf.decision_function(IMB_test_features))
    fpr1, tpr1, thresholds1 = metrics.roc_curve(IMB_test_labels, clf1.decision_function(IMB_test_features))
    fig, ax = plt.subplots()
    roc_auc = metrics.auc(fpr,tpr)
    display = metrics.RocCurveDisplay(fpr = fpr, tpr = tpr, roc_auc = roc_auc, estimator_name = "different_weight")
    display.plot(ax = ax)
    roc_auc1 = metrics.auc(fpr1, tpr1)
    display1= metrics.RocCurveDisplay(fpr = fpr1, tpr = tpr1, roc_auc = roc_auc1, estimator_name = "equal_weight")
    display1.plot(ax = ax)
    plt.savefig("roc.png")
    plt.close()

    #5.1(abc)
    actor_count, actress_count = count_actors_and_actresses(filename)
    print(actor_count, actress_count)
    plot_actors_and_actresses(filename, "rating")
    
    #5.1(d)
    svm = LinearSVC(loss = 'hinge', penalty = 'l2', C = 0.1, random_state = 445)
    svm.fit(X_train, Y_train)
    thetas = svm.coef_[0]
    actor_ind = dictionary_binary['actor']
    actress_ind = dictionary_binary['actress']
    print("actor_coef", thetas[actor_ind])
    print("actress_coef", thetas[actress_ind])

    #5.2(a)
    model = train_word2vec(filename)
    actor_embedding = model.wv['actor']
    print(actor_embedding)
    print(len(actor_embedding))

    #5.2(b)
    similar_words = model.wv.most_similar('plot', topn=5)
    print(similar_words)

    #5.3(a)
    w = 'smart'
    A = ['her', 'woman', 'women']
    B = ['him', 'man', 'men']
    print(compute_association(filename, w, A, B))
    
    # Read multiclass data
    # TODO: Question 6: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    '''
    (
        multiclass_features,
        multiclass_labels,
        multiclass_dictionary,
    ) = get_multiclass_training_data()

    heldout_features = get_heldout_reviews(multiclass_dictionary)
    '''
    scores = []
    C_range = [0.0001, 0.001, 0.005, 0.007, 0.008, 0.009, 0.01, 0.012, 0.05, 0.1, 1, 10]
    penalty = 'l2'
    for C in C_range:
        clf = LinearSVC(C=0.1, penalty = penalty) 
        ovr_classifier = OneVsRestClassifier(clf)
        score = cv_performance_new(clf, multiclass_features, multiclass_labels, 5)
        scores.append(score)
    max_index = scores.index(max(scores))
    print(max(scores))
    print(C_range[max_index])
    '''

    
    clf = LinearSVC(C = 0.1, multi_class = "ovr", loss = "squared_hinge")
    score = cv_performance_new(clf, multiclass_features, multiclass_labels, 5)
    print(score)
    clf.fit(multiclass_features, multiclass_labels)
    predicts = clf.predict(heldout_features)
    generate_challenge_labels(predicts, 'jlzheng')
    
if __name__ == "__main__":
    main()
