# RL-practice
Making Neural Networks play games before I go for my main goal of a stock trading bot

Goals:
	
- [x] Tic Tac Toe
- [ ] Tic Tac Toe (3D)
- [ ] Connect 4
- [ ] Connect 4x4
- [ ] (side project not RL: what to wear today, takes weather data in the area and tells you what clothes to bring)
- [ ] (side project not RL: learn faces of people at my work)
	
Optional:

	-frozen lake
	-car racing from gym
	-bipedal waker from gym
	
<br/>
So how did I get the Tic Tac Toe going?
	
	1. Made a passable version of Tic Tac Toe with ways to win (3 in a row in any direction, tougher than you might think).
	2. Put in a random number generator that randomly plays Tic Tac Toe.
	3. Put in a value system that values the board based on "your" pieces, opponents pieces, and whose turn it is.
	4. Put the random number generator against itself, recorded board history and board values. Then used that data to train a Critic.
	5. Had the model player playing a random player, then had it play itself (but introduced random play), recorded history and moves from winning games. Used that data to train Actor, which plays like a child but can win if you let it. I eventually only let games be recorded if the winning move had a score of 0.5 or more, that way it would try to win sooner.
	6. Train the Actor to play both as X's or O's without changing the board. 
	7. Convert models to TFLite
	8. Set up to work on Raspberry pi
	9. Set up neopixels. Had to create another python script and use sockets to communicate between the two.
	10. Working game playing against the computer with buttons and lights, set up so every human win and some draws are saved as expert moves for future training
<br/>

Comparing Random play to basic play:

<img width="340" alt="random_tic_tac_toe" src="https://user-images.githubusercontent.com/66873149/128274336-b690d634-d4e6-42d0-aa97-1c0d050b343d.PNG">
<img width="292" alt="random_tic_tac_toe2" src="https://user-images.githubusercontent.com/66873149/128274347-cbb38b6e-a9a9-4463-ae7c-d38aa917c3d7.PNG">


See with random play, it plays randomly. Sometimes it may look like it's getting lucky but that's just it. LUCK! There is no strategy, it just places it's O unless it can't, then it will try elsewhere. Like in the first picture, I thought I might have been playing the algorithm until it moved in the top left corner. Then I realized I was playing the random player. The second picture is to further demenstrate how much playing randomly sucks as a strategy. Trying to let a random generator win Tic Tac Toe can be a difficult task, there actually takes some strategy to losing against it.

<img width="291" alt="algo" src="https://user-images.githubusercontent.com/66873149/128274240-48c7586d-39c9-4963-965b-2b1ab8bed093.PNG">
<img width="294" alt="algo2" src="https://user-images.githubusercontent.com/66873149/128274392-0684e7ed-bb47-42f1-abaa-ac53d7a9e3e0.PNG">
<img width="222" alt="algo3" src="https://user-images.githubusercontent.com/66873149/128274404-9855897a-e1be-46c6-89cd-2a987bd70f79.PNG">

In the previous three photos my opponent was the "childish" algorithm, it already has a strategy. Whenever it goes first, it takes the middle (makes sense, best chance of winning from the middle). With the limited testing I've had so far, it likes to go for a horizontal win and in both cases documented my first move was on the right or left of the middle square. I thought it was odd that the algorithm's next move was to complete the middle row, pretty much a wasted move. Maybe in the training data it randomly won a lot in the middle row but as soon as it plays against an opponent with more than one braincell, it had been thwarted!

Now in the first of the photos documenting the algorithm's play, after I realized it was retarded, I went for the win in the bottom right corner to show it who's boss. I didn't think much of it. I thought more of it when I ended up in a mirrored situation later on and I thought it was odd it kept playing the middle row in the early game instead of going literally anywhere else! So I wanted to see what happend when I block it by moving to the top right corner. I was pleasantly surprised when the algorithm went for the diagonal win. Now if this was the random player, it would have a 33% chance of winning the next move and that's not too bad! Remember this algorithm was trained on data from winning random plays not world champion tic tac toe players so it's understandable why the algorithm sucks. 

Now it plays a little better :) Hopefully playing against humans will get better data.
