package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to.
 * @author ae187
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.1;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=100;
	
	/**
	 * The discount factor (gamma)
	 */
	double discount=0.9;
	
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon=0.1;
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair, you can do
	 * qTable.get(game).get(move) which return the Q(game,move) value stored. Be careful with 
	 * cases where there is currently no value. You can use the containsKey method to check if the mapping is there.
	 * 
	 */
	
	QTable qTable=new QTable();
	
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 * @throws IllegalMoveException 
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount) throws IllegalMoveException
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
		this.discount=discount;
		initQTable();
		train();
	}
	
	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 *  
	 */
	
	protected void initQTable()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
		{
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				this.qTable.addQValue(g, m, 0.0);
				//System.out.println("initing q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
			
		}
		
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 * @throws IllegalMoveException 
	 */
	public QLearningAgent() throws IllegalMoveException
	{
		this(new RandomAgent(), 0.1, 100, 0.9);
		
	}
	
	
	/**
	 *  Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating q-values according 
	 *  to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where with the probability {@code epsilon} the
	 *  agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 *  At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned q-values. This is currently
	 *  done for you on the last line of the method.
	 * @throws IllegalMoveException 
	 */
	
	public void train() throws IllegalMoveException
	{	
		int count = 0;
		Random random = new Random();
		double r = 0;
		int winCount = 0;
		for (int i = 0; i<numEpisodes; i++) { 
			while (!env.isTerminal()) {
				Game board = env.getCurrentGameState();
				    r = random.nextDouble();
					if (r>epsilon) {		// exploitation
						double curValue;
						double bestValue = -100;
						double oldQ = 0;
						Move bestMove = null;
						for (Move move : board.getPossibleMoves()) { // finds best move using Q value exploitation, if all Q = 0  it just takes the first move
							curValue = qTable.getQValue(board, move);
							if (curValue > bestValue) {				
								bestValue = curValue;
								bestMove = move;
								oldQ = qTable.getQValue(board, bestMove);
							}
						}
						Outcome outcome = env.executeMove(bestMove); //executes exploitation move
						board = env.getCurrentGameState(); //updates games state
						
						if (env.isTerminal()) {
							if (board.state==2) {
								winCount++;
							}
							
							break;
						}
						
						
						curValue = 0;
						bestValue = -100;
						Move bestMoveSP = null;
						for (Move move : board.getPossibleMoves()) { // finds maxQ of sPrime
							curValue = qTable.getQValue(board, move);
							if (curValue > bestValue) {
								bestValue = curValue;
								bestMoveSP = move;
							}
						}
						double maxQ = qTable.getQValue(board, bestMoveSP);
						double newQ = (1-this.alpha)*oldQ+ this.alpha*(outcome.localReward+discount*maxQ);
						qTable.addQValue(board,bestMove, newQ );
					}
					if (r<epsilon) {		// exploration
						List<Move> moves = board.getPossibleMoves();
						Move randMove = moves.get(random.nextInt(moves.size()));	
						double oldQ = qTable.getQValue(board, randMove);  // explores by find a randMove
						
						Outcome outcome = env.executeMove(randMove); //executes exploration move
						board = env.getCurrentGameState();		//updates games state
						
						if (env.isTerminal()) {
							if (env.getCurrentGameState().state==2) {
								winCount++;
							}
							break;
						}
						double curValue;
						double bestValue = -100;
						Move bestMove = null;
						for (Move move : board.getPossibleMoves()) { //updates games state
							curValue = qTable.getQValue(board, move);
							if (curValue > bestValue) {
								bestValue = curValue;
								bestMove = move;
							}
						}
						double maxQ = qTable.getQValue(board, bestMove); //max Q value of sPrime
						double newQ = (1-this.alpha)*oldQ+ this.alpha*(outcome.localReward+discount*maxQ);
						qTable.addQValue(board, randMove, newQ);
					} 
					count++;
				}
				env = new TTTEnvironment();// starts a new game
		}
		System.out.println(count+"moves played");
		System.out.println("wins"+winCount);
		System.out.println(env.game.getState());
		
		//--------------------------------------------------------
		//you shouldn't need to delete the following lines of code.
		this.policy=extractPolicy();
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			//System.exit(1);
		}
	}
	
	/** Implement this method. It should use the q-values in the {@code qTable} to extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy()
	{
		HashMap<Game, Move> movePolicy = new HashMap<Game, Move>();
		Move bestMove = null;
		double curValue;
		double maxValue;
		for (Game board : qTable.keySet())	{
			maxValue = -100;
			for(Move move: qTable.get(board).keySet()) {
				curValue = qTable.getQValue(board, move);
				if (curValue > maxValue) {
					maxValue = curValue;
					bestMove = move;
				}
				movePolicy.put(board, bestMove);
			}
		}
		return new Policy(movePolicy);
		
	}
	
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play your agent against a human agent (yourself).
		QLearningAgent agent=new QLearningAgent();
		
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
	
	
	


	
}
