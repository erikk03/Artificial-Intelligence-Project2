# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPacPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        FoodAsList = newFood.asList()
        closest_food = float('inf')

        score = successorGameState.getScore()

        for i in FoodAsList:
            closest_food = min(closest_food, manhattanDistance(newPacPos, i))       # find closest food from pacman distance
        
        return_value = score + 1.0/closest_food

        return return_value

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        
        return self.min_or_max(gameState, 0, 0)[0]

    # Define two functions for minimax algorithm
    # minimax returns the value that evaluationFunction returns, or min_or_max value, based on each case
    # works as the recursive function when called inside min_or_max
    def minimax(self, gameState, agentIndex, depth):
        NumAgents = gameState.getNumAgents()

        if (depth is self.depth * NumAgents):
            return self.evaluationFunction(gameState)
        
        if (gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)

        return self.min_or_max(gameState, agentIndex, depth)[1]
        
    # min_or_max returns the minimum or maximum value as the best action based on agentIndex value
    def min_or_max(self, gameState, agentIndex, depth):
        NumAgents = gameState.getNumAgents()
        LegalActions = gameState.getLegalActions(agentIndex)

        if (agentIndex is 0):                                                           # Initialize BestAction value based on agentIndex
            BestAction = ("maximum", -float("inf"))
        elif (agentIndex is not 0):
            BestAction = ("minimum", float("inf"))

        for i in LegalActions:
            successor = gameState.generateSuccessor(agentIndex, i)
            SuccessorAction = (i, self.minimax(successor, (depth + 1)%NumAgents, depth + 1))
            
            if (agentIndex is 0):                                                       # Find BestAction 
                BestAction = max(BestAction, SuccessorAction, key=lambda x:x[1])        # Is the max when agentIndex==0. Player is MAX
            elif (agentIndex is not 0):
                BestAction = min(BestAction, SuccessorAction, key=lambda x:x[1])        # min for when player is MIN

        return BestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.min_or_max(gameState, 0, 0, -float("inf"), float("inf"))[0]
    
    # Define two functions for alphabeta algorithm
    # alphabeta returns the value that evaluationFunction returns, or min_or_max value, based on each case
    # works as the recursive function when called inside min_or_max
    # like minimax    
    def alphabeta(self, gameState, agentIndex, depth, a, b):
        NumAgents = gameState.getNumAgents()

        if (depth is self.depth * NumAgents):
            return self.evaluationFunction(gameState)
        
        if (gameState.isWin() or gameState.isLose()):
            return self. evaluationFunction(gameState)

        return self.min_or_max(gameState, agentIndex, depth, a, b)[1]
        
    # min_or_max returns the minimum or maximum value as the best action based on agentIndex value
    # this time we want to stop searching all the LegalActions in some cases
    def min_or_max(self, gameState, agentIndex, depth, a, b):
        NumAgents = gameState.getNumAgents()
        LegalActions = gameState.getLegalActions(agentIndex)

        if (agentIndex is 0):                                                       # Initialize BestAction value based on agentIndex
            BestAction = ("maximum", -float("inf"))
        elif (agentIndex is not 0):
            BestAction = ("minimum", float("inf"))

        for i in LegalActions:
            
            if (b < a):                                                             # If b<a no need to continue, go to next legal action
                break

            successor = gameState.generateSuccessor(agentIndex, i)
            SuccessorAction = (i, self.alphabeta(successor, (depth + 1)%NumAgents, depth + 1, a, b))
            
            if (agentIndex is 0):                                                   # Find BestAction based on agentIndex
                BestAction = max(BestAction, SuccessorAction, key=lambda x:x[1])
                
                if (BestAction[1] > b):                                             # If BestAction[1] > b no need to continue
                    return BestAction                                               # Just return BestAction
                else:
                    a = max(BestAction[1], a)                                       # Else, parameter a == BestAction[1] if BestAction[1] > previous_a
            elif (agentIndex is not 0):
                BestAction = min(BestAction, SuccessorAction, key=lambda x:x[1])

                if (BestAction[1] < a):                                             # If BestAction[1] < a, return BestAction
                    return BestAction
                else:
                    b = min(BestAction[1], b)                                       # b is lower value beetween BestAction[1] and previous b

        return BestAction
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        NumAgents = gameState.getNumAgents()
        Depth = self.depth
        MaximumDepth = Depth * NumAgents
        
        return self.expectimax(gameState, 0, MaximumDepth, "expect")[0]

    # Define two functions for expectimax algorithm
    # expectimac returns the value that evaluationFunction returns, or exp_or_max value, based on each case
    # works as the recursive function when called inside exp_or_max
    def expectimax(self, gameState, agentIndex, depth, action):

        if (depth is 0):
            return (action, self.evaluationFunction(gameState))
        
        if (gameState.isWin() or gameState.isLose()):
            return (action, self.evaluationFunction(gameState))

        return self.exp_or_max(gameState, agentIndex, depth, action)

    # exp_or_max returns the expected or maximum value based on agentIndex
    # if agentIndex is 0(Playes is MAX), we want maximum value
    # if agentIndex is not 0(Player is MIN), we want the expected value    
    def exp_or_max(self, gameState, agentIndex, depth, action):
        NumAgents = gameState.getNumAgents()
        LegalActions = gameState.getLegalActions(agentIndex)

        if (agentIndex is 0):
            BestAction = ("maximum", -float("inf"))

            for i in LegalActions:
                nextAgent = (agentIndex + 1) % NumAgents
                SuccessorAction = None

                if depth != self.depth * NumAgents:
                    SuccessorAction = action
                else:
                    SuccessorAction = i

                successor = gameState.generateSuccessor(agentIndex, i)
                SuccessorValue = self.expectimax(successor, nextAgent, depth - 1, SuccessorAction)
                BestAction = max(BestAction, SuccessorValue, key = lambda x:x[1])

            return BestAction 
        elif (agentIndex is not 0):                             # for agentIndex != 0, we want to return the action with a score
            score = 0                                           # score initialization 0, is going to change based on propability
            propability = 1.0/len(LegalActions)

            for i in LegalActions:
                nextAgent = (agentIndex + 1) % NumAgents
                successor = gameState.generateSuccessor(agentIndex, i)
                BestAction = self.expectimax(successor, nextAgent, depth - 1, action)
                score = score + BestAction[1] * propability
            
            return (action, score)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    In the previous evaluation function we evaluated based on closest_food to pacman.
    In this evaluation function we are going to add some more parameters.
    Firstly, score is very important, so we just save it in return_value.
    Next, we are going to evaluate based on the currentGameState, if we won or lost the game.
    As we underdstand, win is the most important aspect.
    After that, we want to know how many capsules are in the game.
    We add to the return_value parameters that include the closest_food,
    as in the previous evaluation function.
    Last but not least, we want to avoid ghosts that are too close to pacman.
    """
    # Useful information you can extract from a GameState (pacman.py)
    PacPos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    GhostPos = currentGameState.getGhostPositions()
    newScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    capsules = currentGameState.getCapsules()
    
    FoodAsList = Food.asList()
    closest_food = float('inf')

    return_value = currentGameState.getScore()                  # starting evaluation based on score

    if currentGameState.isWin():                                # new evaluation based on GameState, if we won or lost
        return_value = return_value + 1000.0
    elif currentGameState.isLose():
        return_value = return_value - 1000.0

    return_value = return_value + 1000.0/(len(capsules) + 1)    # new evaluation based on capsules

    for i in FoodAsList:                                        # new evaluation based on closest food from pacman
        closest_food = min(closest_food, manhattanDistance(PacPos, i))
    
    return_value = return_value + 1.0/closest_food

    for i in GhostPos:                                          # evaluation when too close to a ghost
        distance_from_ghost = manhattanDistance(PacPos, i)
        if (distance_from_ghost < 2):
            return -float('inf')
    
    return return_value                                         # return final evaluation

# Abbreviation
better = betterEvaluationFunction
