import random
import nltk
import math
import numpy as np

class Node:  # a node in the tree
    def __init__(self, label, word=None,pos=None):
        self.label = label
        self.word = word
        self.parent = None  # reference to parent
        self.left = None  # reference to left child
        self.right = None  # reference to right child
        self.pos = pos
        # true if I am a leaf (could have probably derived this from if I have
        # a word)
        self.isLeaf = False
        # true if we have finished performing fowardprop on this node (note,
        # there are many ways to implement the recursion.. some might not
        # require this flag)
        self.annotated = False
        self.plabel = None
    
    def isRoot(self):
        return self.parent==None

    def __str__(self):
        if self.isLeaf:
            return '[{0}:{1}]'.format(self.word, self.label)
        return '({0} <- [{1}:{2}] -> {3})'.format(self.left, self.word, self.label, self.right)
    def getLeafWord(self):
        return self.word

    def set_label(self,label):
        self.label = label
    
class Tree:    
    def __init__(self, root_node):
        self.root = root_node
        # get list of labels as obtained through a post-order traversal
        self.labels = get_labels(self.root)
        self.num_words = len(getLeaves(self.root))

    def get_words(self):
        leaves = getLeaves(self.root)
        words = [node.word for node in leaves]
        return words

def get_labels(node):
    if node is None:
        return []
    return get_labels(node.left) + get_labels(node.right) + [node.label]




def getLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return getLeaves(node.left) + getLeaves(node.right)


def getNonLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node.parent]
    else:
        return getLeaves(node.left) + getLeaves(node.right)



def getLabel(l):
    return (l.label)

def isLabel(l):
    if hasattr(l,'label'):
        if l.label is None:
            return False
        else:
            return True
    else:
        return False


def composeList(l):
    if type(l[0]) == list and type(l[1]) == list:
        l = composeList(l[0]) + composeList(l[1])
    elif type(l[0]) == list and type(l[1]) == str:
        l = composeList(l[0]) + l[1]
    elif type(l[0]) == str and type(l[1]) == list:
        l = l[0] + composeList(l[1])
    elif type(l[0]) == str and type(l[1]) == str:
        l = l[0] + l[1]
    else:
        print("Check, somethings wrong")

    return l


def calculate_probabilities(corpus):
    """
    Calculates the probabilities according to: P(w_i) = 1 - sqrt(t / f(w_i)) where t is some threshold (reported as 10e-5)
    :param corpus: list of all words in the corpus
    :return: dictionary mapping words to probabilities
    """

    fdist = nltk.FreqDist(corpus)
    t = 10e-5

    probs = {}
    for k in fdist.keys():
        probs[k] = 1 - (math.sqrt(t / fdist[k]))

    return probs

