import pickle

with open('expert_moves.pickle', 'rb') as re:
	b = pickle.load(re)
print(len(b))
