# RL-practice
Making NN play games before I go for the big goal of stock trading

Goals:
	
	-Tic Tac Toe
	-Tic Tac Toe (3D)
	-Connect 4
	-Connect 4x4
	-(side project: what to wear today, takes weather data in the area and tells you what clothes to bring)
	-(side project: learn faces of people in the electrician shop)
	
Optional:

	-frozen lake
	-car racing from gym
	-bipedal waker from gym
	
	
So how did I get the Tic Tac Toe going?
	
	1. Made a passable version of Tic Tac Toe with ways to win (3 in a row in any direction, tougher than you might think).
	2. Put in a random number generator that randomly plays Tic Tac Toe.
	3. Put in a value system that values the board based on "your" pieces, opponents pieces, and whose turn it is.
	4. Put the random number generator against itself, recorded board history and board values. Then used that data to train a Critic which doesn't work right now :(
	5. Kept the random player playing itself, recorded history and moves from winning games. Used that data to train Actor, which plays like a child but can win if you let it.
