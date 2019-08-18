from pycorenlp import StanfordCoreNLP
import nltk.tree
import pptree
from mytree import *
from utils import *
from treeUtil import *
import pickle

def printLabelTree(tree):
    def inorder(node,nnode):
        if node.isLeaf:
            newnode = pptree.Node('H',nnode)
            wnode = pptree.Node(node.word,newnode)
        elif nnode is not None:
            newnode = pptree.Node('H',nnode)
            inorder(node.left,newnode)
            inorder(node.right,newnode)
        elif node.isRoot():
            newnode = pptree.Node('H')
            inorder(node.left,newnode)
            inorder(node.right,newnode)
            return newnode
        return None
    pptree.print_tree(inorder(tree.root,None))

def make_tree(text):
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,depparse,parse',
        'outputFormat': 'json'
    })
    tree = str(output['sentences'][0]['parse'])
    # print (tree)
    parse_string = ' '.join(str(tree).split())
    # print(parse_string)
    # print ("\n\n")
    tree = nltk.tree.Tree.fromstring(parse_string)
    tree.chomsky_normal_form()
    tree.collapse_unary(collapseRoot=True,collapsePOS=True)
    nt = convertNLTK_tree(tree)
    return nt


if __name__ == '__main__':
    pro,anti = [],[]
    nlp = StanfordCoreNLP('http://act4dgem.cse.iitd.ac.in:9000')
    fin = open('pro_statements.txt','r')
    for line in fin:
        pro.append(make_tree(line))
    fin.close()
    fin = open('anti_statements.txt','r')
    for line in fin:
        anti.append(make_tree(line))
    trees = [pro,anti]
    fin.close()
    fout = open('tech_trees.pkl','wb')
    pickle.dump(trees,fout)
    fout.close()

    # printLabelTree(nt)









    # nlp = StanfordCoreNLP('http://localhost:9000')
    # text = (
    # 'Forcing middle-class workers to bear a greater share of the cost of government weakens their support for needed investments and stirs resentment toward those who depend on public services the most .')
