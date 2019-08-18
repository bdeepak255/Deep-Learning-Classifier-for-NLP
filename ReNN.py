import sys
import random
import progressbar
import torch
import pickle
from mytree import *
from utils import *
from treeUtil import *
import tqdm
import argparse
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import functools
# from lbfgs_utils import compute_stats, get_grad
# from LBFGS import LBFGS
from sklearn.linear_model import LogisticRegression
import time,gensim

w2vec = False
embedDim = 300

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trees', dest='trees', required=True)
    parser.add_argument('-e', '--epochs', dest='epochs', default=1, required=True)
    parser.add_argument('-d', '--dim', dest='dim', default=300, required=True)
    
    return parser.parse_args()


class RecursiveNN(nn.Module):
	def __init__(self, vocab, embedSize=300, numClasses=2, beta = 0.3, use_weight = True, non_trainable = False):
		super(RecursiveNN, self).__init__()
		if (w2vec):
			# self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(w2vec_weights), freeze = False)
			# self.embedding = nn.Embedding.from_pretrained(w2vec_weights, freeze = False)
			self.embedding = nn.Embedding(len(vocab), embedSize)
			self.embedding.load_state_dict({'weight': w2vec_weights})
			self.embedding.weight.requires_grad = True
			if non_trainable:
				self.embedding.weight.requires_grad = False
			else:
				self.embedding = nn.Embedding(len(vocab), embedSize)
		self.embedding = nn.Embedding(len(vocab), embedSize)
		self.W = nn.Linear(2*embedSize, embedSize, bias=True)
		self.nonLinear = torch.tanh
		self.projection = nn.Linear(embedSize, numClasses, bias=True)
		self.nodeProbList = []
		self.labelList = []
		self.loss = Var(torch.FloatTensor([0]))
		self.V = vocab
		self.beta = beta
		self.use_weight = use_weight
		self.total_rep = None #
		self.count_rep = 0 #

	def traverse(self, node):
		if node.isLeaf:
			if node.getLeafWord() in self.V:  # check if right word is in vocabulary
				word = node.getLeafWord()
			else:  # otherwise use the unknown token
				word = 'UNK'
			# print(self.V[word],len(self.V),word,(torch.LongTensor([int(self.V[word])])))
			currentNode = (self.embedding(Var(torch.LongTensor([int(self.V[word])]))))
		else: currentNode = self.nonLinear(self.W(torch.cat((self.traverse(node.left),self.traverse(node.right)),1)))
		currentNode = currentNode/(torch.norm(currentNode))

		assert node.label!=None
		self.nodeProbList.append(self.projection(currentNode))
		# print (node.label)
		self.labelList.append(torch.LongTensor([node.label]))
		loss_weight = 1-self.beta if node.annotated else self.beta
		self.loss += (loss_weight*F.cross_entropy(input=torch.cat([self.projection(currentNode)]),target=Var(torch.cat([torch.LongTensor([node.label])]))))
		
		#
		if not node.isRoot():
			if self.total_rep is None:
				self.total_rep = currentNode.data.clone()
			else:
				self.total_rep += currentNode.data.clone()
			self.count_rep += 1
		#
		
		return currentNode        

	def forward(self, x):
		self.nodeProbList = []
		self.labelList = []
		self.loss = Var(torch.FloatTensor([0]))
		self.traverse(x)
		self.labelList = Var(torch.cat(self.labelList))
		return torch.cat(self.nodeProbList)

	def getLoss(self, tree):
		nodes = self.forward(tree)
		predictions = nodes.max(dim=1)[1]
		loss = self.loss
		return predictions,loss

	def getRep(self, tree):
		self.count_rep = 0
		self.total_rep = None
		self.nodeProbList = []
		self.labelList = []
		self.loss = Var(torch.FloatTensor([0]))
		
		root_rep = self.traverse(tree)

		return (torch.cat((root_rep,self.total_rep/self.count_rep),1)).data.numpy().T.flatten()


	def evaluate(self, trees):
			pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(trees)).start()
			n = nAll = correctRoot = correctAll = 0.0
			for j, tree in enumerate(trees):
					predictions,_ = self.getLoss(tree.root)
					correct = ((predictions.cpu().data).numpy()==(self.labelList.cpu().data).numpy())
					correctAll += correct.sum()
					nAll += np.shape(correct.squeeze())[0] if np.size(correct)!=1 else 1 
					correctRoot += correct.squeeze()[-1] if np.size(correct)!=1 else correct[-1]
					n += 1
					pbar.update(j)
			pbar.finish()
			return correctRoot / n, correctAll/nAll
	
	def eval_sent_lvl(self,trees,clf):
		pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(trees)).start()
		n = nAll = correctRoot = correctAll = 0.0
		X_predict = []
		Y_gold = []
		for j, tree in enumerate(trees):
			tree_rep = model.getRep(tree.root)
			X_predict.append(tree_rep)
			Y_gold.append(tree.root.label)
		acc = clf.score(np.array(X_predict),np.array(Y_gold))
		return acc



# def getLabelList(node):
# 	labelList = []
# 	def traverse(tnode):
# 		if not tnode.isLeaf:
# 			traverse(tnode.left)
# 			traverse(tnode.right)
# 		labelList.append(([tnode.label]))
# 	traverse(node)
# 	return labelList
# def getLabelLists(batch):
# 	labels = []
# 	for sample in batch:
# 		labels += getLabelList(sample.root)
# 	return np.array(labels)




CUDA=False
def Var(v):
    if CUDA: return Variable(v.cuda())
    else: return Variable(v)

# if len(sys.argv)>1:
#     if sys.argv[1].lower()=="cuda": CUDA=True

# args = arg_parse()

trees = []
raw_words = []
vocab = []

print("Loading trees...")
# for file in os.listdir(args.trees):
#     trees.append([Tree(line) for line in open(str(args.trees) + '/' + str(file), 'r').read().splitlines()])
# [lib, con, neutral] = pickle.load(open(str(args.trees), 'rb'))

[pro, anti] = pickle.load(open('tech_trees.pkl','rb'))


for tree in pro:
    tree.root.set_label('pro')
for tree in anti:
    tree.root.set_label('anti')
# for tree in neutral:
#     tree.set_label('Neutral')
#     print(tree.label)



temp_trees = []
temp_trees.extend(pro[:84])
temp_trees.extend(anti)
# temp_trees.extend(neutral)

# trees = []
# trees.extend(pro)
# trees.extend(anti)
 

trees.append([Tree(convert(tree.root)) for tree in temp_trees])

trees = [s for l in trees for s in l]  # chain all trees into one array

# trees = [l for l in trees]  # chain all trees into one array

# print (str(trees[0].root))

print("{} trees loaded!".format(len(trees)))

# print (trees[0].get_words())

print("Building vocab...")
for tree in tqdm.tqdm(trees):
	words = tree.get_words()
	raw_words.append(words)
	for word in words:
		if word not in vocab: # default_dict can be used
			vocab.append(word)

raw_words = [s for l in raw_words for s in l]  # chain the sublists in raw_words
vocab = {w: i for (i, w) in enumerate(vocab)}
if 'UNK' not in vocab:
    vocab['UNK'] = max(vocab.values()) + 1  # Set UNK token

print("{} words found with a vocabulary size of {}".format(len(raw_words), len(vocab)))



#Word2vec embeddings Loading
if (w2vec):
  # Needs to be done just once, stores in pickle file then
 	'''
  print ("Loading Word2vec Weights")
  w2vec_start = time.time()
  w2vec_model = gensim.models.KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin", binary = True)
  w2vec_end = time.time()
  print ("Loaded Word2vec Weights! " + str (round((w2vec_end-w2vec_start)/60,2)))

  w2vec_weights_ = np.zeros((len(vocab),embedDim))
  for word in vocab.keys():
    word_index = vocab[word]
    # print (word,word_index)
    try: 
      word_embedding = np.array(w2vec_model.wv[word], dtype = 'float32')
    except: #if word not in google word2vec
      # word_embedding = np.zeros(embedDim, dtype = 'float32')
      word_embedding = np.random.normal(scale=0.6, size=(embedDim, ))
    # print (word_embedding)
    w2vec_weights_[word_index,:] = torch.FloatTensor(word_embedding)

  w2vec_weights_ = torch.from_numpy(w2vec_weights_)
  print ("vocab specific embedding object created")

  print ("Saving to pickle object")
  fout = open("vocab1314_w2vec_weights.pkl",'wb')  # for testing just 40 trees
  pickle.dump(w2vec_weights_,fout)
  fout.close()
  print ("Saved to pickle object")
  '''

  # fin = open ("vocab1314_w2vec_weights.pkl",'rb')
  # w2vec_weights = pickle.load(fin)
  # fin.close()
  # print ("\nWord2vec weights loaded from pickle object")




if CUDA: model = RecursiveNN(vocab).cuda()
else: model = RecursiveNN(vocab)

# num_parameter = 0
# print (model)
# for parameter in model.parameters():
# 	num_parameter += 1
# print (num_parameter)	


max_epochs = 100
widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]

l2_reg = {  'embedding.weight' : 1e-6,'W.weight' : 1e-4,'W.bias' : 1e-4,'projection.weight' : 1e-3,'projection.bias' : 1e-3}


random.shuffle(trees)

# BATCH_SIZE = 1024
# SUB_BATCH_SIZE = 128
# def opfun(X):
#     output = []
#     for x in X:
#         output.append(model.forward(x.root))
#     return torch.cat(output)

trn,dev = trees[:int((len(trees)+1)*.90)],trees[int(len(trees)*.90+1):]
# optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search='Wolfe', debug=True)


optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0.0)
# bestAll=bestRoot=0.0
BATCH_SIZE = len(trn)
# optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=10, history_size = 10)
bestAll=bestRoot=0.0
best_trn_All = best_trn_Root = 0.0 
for epoch in range(max_epochs):
	print("\n\nEpoch %d" % epoch)
	pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(trn)/BATCH_SIZE).start()
	params = []
	for i in range(0,len(trn),BATCH_SIZE):
		batch = trn[i:min(i+BATCH_SIZE,len(trn))]
		def closure():
			optimizer.zero_grad()
			_,total_loss = model.getLoss(trn[0].root)
			for tree in batch:
				_, loss = model.getLoss(tree.root)
				total_loss += loss
				
			total_loss = total_loss/len(batch)
			#L2 reg
			param_dict = dict()
			for name, param in model.named_parameters():
				param_dict[name] = param.data.clone()
				if param.requires_grad:
						total_loss += 0.5*l2_reg[name]*(torch.norm(param)**2)
			params.append(param_dict)
			print('Loss = ',total_loss.data)
			total_loss.backward()
			clip_grad_norm_(model.parameters(),5,2)
			return total_loss
		pbar.update(i/BATCH_SIZE)
		optimizer.step(closure)
		
	pbar.finish()

	avg_param = dict()
	for name, param1 in model.named_parameters():
			avg_param[name] = param1.data.clone()
			
	for i in range(1,len(params)):
		for name, param in params[i].items():
			avg_param[name] += param.clone()
	for name, param in model.named_parameters():
		if name == 'embedding.weight':
			continue
		param.data = avg_param[name]/len(params)



	# X_train, Y_train = [],[]
	# for tree in trn:
	# 	X_train.append(model.getRep(tree.root))
	# 	Y_train.append(tree.root.label)
	# X = np.array(X_train)
	# Y = np.array(Y_train)
	# LR_clf = LogisticRegression().fit(X,Y)


	correctRoot, correctAll = model.evaluate(dev)
	# correctRoot = model.eval_sent_lvl(dev,LR_clf)
	if bestAll<correctAll: bestAll=correctAll
	if bestRoot<correctRoot: bestRoot=correctRoot
	print("\nValidation All-nodes accuracy:"+str(round(correctAll,2))+"(best:"+str(round(bestAll,2))+")")
	print("Validation Root accuracy:" + str(round(correctRoot,2))+"(best:"+str(round(bestRoot,2))+")")

	correct_trn_Root, correct_trn_All = model.evaluate(trn)
	# correctRoot = model.eval_sent_lvl(dev,LR_clf)
	if best_trn_All<correct_trn_All: best_trn_All=correct_trn_All
	if best_trn_Root<correct_trn_Root: best_trn_Root=correct_trn_Root
	print("\nTraining All-nodes accuracy:"+str(round(correct_trn_All,2))+"(best:"+str(round(best_trn_All,2))+")")
	print("Training Root accuracy:" + str(round(correct_trn_Root,2))+"(best:"+str(round(best_trn_Root,2))+")")




	random.shuffle(trn)



# for epoch in range(max_epochs):
	
	
# 	# training mode
# 	model.train()
# 	params = []
  
# 	print("Epoch %d" % epoch)
	
# 	#Create batch
# 	random_index = np.random.permutation(len(trn))
# 	batch = [trn[i] for i in random_index[0:BATCH_SIZE]]
	
	
# 	#compute init grad
# 	grad, obj = get_grad(optimizer, batch, opfun, getLabelLists,SUB_BATCH_SIZE)

# 	# two-loop recursion to compute search direction
# 	p = optimizer.two_loop_recursion(-grad)

# 	#define closure fn
# 	def closure():
# 		#show progress
# 		pbar = progressbar.ProgressBar(widgets=widgets, maxval=(BATCH_SIZE)).start()
	
# 		optimizer.zero_grad()
# 		total_loss = torch.tensor(0, dtype=torch.float)
		
# 		#model parameters at iter
# 		param_dict = dict()
# 		for name, param in model.named_parameters():
# 			param_dict[name] = param.data.clone()
# 			if param.requires_grad:
# 					total_loss += 0.5*l2_reg[name]*(torch.norm(param)**2)
# 		params.append(param_dict)
		

# 		step = 0

# 		#compute loss for each batch
# 		for idx in np.array_split(random_index[0:BATCH_SIZE], max(int(BATCH_SIZE/SUB_BATCH_SIZE), 1)):
# 			#sub samples
# 			subsmpl = [trn[i] for i in idx]
# 			#outputs
# 			ops = opfun(subsmpl)
			
# 			if CUDA:
# 				tgts = torch.from_numpy(getLabelLists(subsmpl)).cuda().long().squeeze()
# 			else:
# 				tgts = torch.from_numpy(getLabelLists(subsmpl)).long().squeeze()
			
# 			#L2 reg
# 			total_loss += F.cross_entropy(ops, tgts)*(len(subsmpl)/BATCH_SIZE)
# 			step += len(idx)
# 			pbar.update(step)
# 		pbar.finish()

# 		#L2-reg
# 		for name, param in model.named_parameters():
# 			if param.requires_grad:
# 				total_loss += 0.5*l2_reg[name]*(param.norm(2)**2)
		
# 		return total_loss
		
# 	# perform line search step
# 	options = {'closure': closure, 'current_loss': obj}
# 	obj, grad, lr, _, _, _, _, _ = optimizer.step(p, grad, options=options)
			
# 	# curvature update
# 	optimizer.curvature_update(grad)
# 	model.eval()


# 	#compute acc before paramter averaging
# 	correctRoot, correctAll = model.evaluate(dev)
# 	if bestAll<correctAll: bestAll=correctAll
# 	if bestRoot<correctRoot: bestRoot=correctRoot
# 	print("\nValidation All-nodes accuracy:"+str(round(correctAll,2))+"(best:"+str(round(bestAll,2))+")")
# 	print("Validation Root accuracy:" + str(round(correctRoot,2))+"(best:"+str(round(bestRoot,2))+")")
	

# 	#update params by averaging
# 	avg_param = dict()
# 	for name, param1 in model.named_parameters():
# 		avg_param[name] = param1.data.clone()
			
# 	for i in range(1,len(params)):
# 		for name, param in params[i].items():
# 			avg_param[name] += param.clone()
# 	for name, param in model.named_parameters():
# 		if name == 'embedding.weight':
# 			continue
# 		param.data = avg_param[name]/len(params)


# 	#compute acc after update
# 	correctRoot, correctAll = model.evaluate(dev)
# 	if bestAll<correctAll: bestAll=correctAll
# 	if bestRoot<correctRoot: bestRoot=correctRoot
# 	print("\nValidation All-nodes accuracy:"+str(round(correctAll,2))+"(best:"+str(round(bestAll,2))+")")
# 	print("Validation Root accuracy:" + str(round(correctRoot,2))+"(best:"+str(round(bestRoot,2))+")")
# 	random.shuffle(trn)
