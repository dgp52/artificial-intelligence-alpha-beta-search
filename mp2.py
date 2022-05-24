#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Deepkumar Patel
Created on Feburary 11, 2021, ‏‎8:54:40 PM
Semester: Spring 1, 2021
Course name: Artificial Intelligence
Assignment: Machine Problem 2
Agent vs Monster Minimax Search with alpha/beta pruning
Description:
    In this game, the agent has to grab the gold and escape to the exit position.
    The agent has to do this while trying to escape from the monster,
    which will eat the agent if they are co-located.
    The monster can move in four directions, except the walls.
        It can also stay in the same position.
    The agent can move in the same directions, but can also:
        grab the gold (if co-located with it), and
        build a wall in one of four directions, assuming the square is empty
    The agent must move or build a wall in each turn (cannot do nothing)

Uses a custom evaluation function for estimating utilities past a maximum depth
"""

import numpy as np
import random
import math

class GenGameBoard: 
    """
    Class responsible for representing the game board and game playing methods
    """
    num_pruned = 0 # counts number of pruned branches due to alpha/beta
    MAX_DEPTH = 20  # max depth before applying evaluation function
    depth = 0      # current depth within minimax search
    has_reached_terminal = False
    
    UP = 'w'
    DOWN = 's'
    LEFT = 'a'
    RIGHT = 'd'
    UP_BUILD = 'wb'
    DOWN_BUILD = 'sb'
    LEFT_BUILD = 'ab'
    RIGHT_BUILD = 'db'
    
    def __init__(self, board_size=4):
        """
        Constructor method - initializes each position variable and the board
        """
        self.board_size = board_size  # Holds the size of the board
        self.marks = np.empty((board_size, board_size),dtype='str')  # Holds the mark for each position
        self.marks[:,:] = ' '
        self.has_gold = False
        self.monster_pos = (0,0)
        self.player_pos = (3,0)
        self.gold_pos = (1,2)
        self.exit_pos = (3,0)
        self.max_moves = self.board_size * 2 + 1
        self.num_moves = 0
        self.depth_reached = 0
    
    def print_board(self, player_move): 
        """
        Prints the game board using current marks
        """ 
        if not player_move:
            #Print number of depth and pruned branches
            print("Depth reached: " + str(self.depth_reached))
            print("Number pruned due to a/b: " + str(self.num_pruned))

        # Prthe column numbers
        print(' ',end='')
        for j in range(self.board_size):
            print(" "+str(j+1), end='')        
        
        # Prthe rows with marks
        print("")
        for i in range(self.board_size):
            # Prthe line separating the row
            print(" ",end='')
            for j in range(self.board_size):
                print("--",end='')
            
            print("-")

            # Prthe row number
            print(i+1,end='')
            
            # Prthe marks on self row
            for j in range(self.board_size):
                if (i,j)==self.monster_pos:
                    print("|W",end='') 
                elif (i,j)==self.gold_pos and not self.has_gold:                    
                    print("|G",end='')
                elif (i,j)==self.player_pos:
                    print("|P",end='')                 
                else:
                    print("|"+self.marks[i][j],end='')
            
            print("|")
                
        
        # Prthe line separating the last row
        print(" ",end='')
        for j in range(self.board_size):
            print("--",end='')
        
        print("-")
    
    def make_move(self, action, player_move):
        """
        Makes the move for either player or monster
        """        
        #Gracefully handle No/None action
        if action == None:
            return

        assert action in self.get_actions(player_move)
        
        # Make the move
        if player_move:
            if action==self.UP:
                self.player_pos = (self.player_pos[0] - 1, self.player_pos[1])
            elif action==self.DOWN:
                self.player_pos = (self.player_pos[0] + 1, self.player_pos[1])
            elif action==self.LEFT:
                self.player_pos = (self.player_pos[0], self.player_pos[1] - 1)
            elif action==self.RIGHT:
                self.player_pos = (self.player_pos[0], self.player_pos[1] + 1)
            elif action==self.UP_BUILD:
                self.marks[self.player_pos[0]-1, self.player_pos[1]] = '#'
            elif action==self.DOWN_BUILD:
                self.marks[self.player_pos[0]+1, self.player_pos[1]] = '#'
            elif action==self.LEFT_BUILD:
                self.marks[self.player_pos[0], self.player_pos[1]-1] = '#'
            elif action==self.RIGHT_BUILD:
                self.marks[self.player_pos[0], self.player_pos[1]+1] = '#'
            self.num_moves = self.num_moves + 1
        else:
            if action==self.UP:
                self.monster_pos = (self.monster_pos[0] - 1, self.monster_pos[1])
            elif action==self.DOWN:
                self.monster_pos = (self.monster_pos[0] + 1, self.monster_pos[1])
            elif action==self.LEFT:
                self.monster_pos = (self.monster_pos[0], self.monster_pos[1] - 1)
            elif action==self.RIGHT:
                self.monster_pos = (self.monster_pos[0], self.monster_pos[1] + 1)
    
    def undo_move(self, action, player_move):
        #Undo the move for either player or monster

        # Undo the move
        if player_move:
            if action==self.UP:
                self.player_pos = (self.player_pos[0] + 1, self.player_pos[1])
            elif action==self.DOWN:
                self.player_pos = (self.player_pos[0] - 1, self.player_pos[1])
            elif action==self.LEFT:
                self.player_pos = (self.player_pos[0], self.player_pos[1] + 1)
            elif action==self.RIGHT:
                self.player_pos = (self.player_pos[0], self.player_pos[1] - 1)
            elif action==self.UP_BUILD:
                self.marks[self.player_pos[0]-1, self.player_pos[1]] = ' '
            elif action==self.DOWN_BUILD:
                self.marks[self.player_pos[0]+1, self.player_pos[1]] = ' '
            elif action==self.LEFT_BUILD:
                self.marks[self.player_pos[0], self.player_pos[1]-1] = ' '
            elif action==self.RIGHT_BUILD:
                self.marks[self.player_pos[0], self.player_pos[1]+1] = ' '
            self.num_moves = self.num_moves - 1
        else:
            if action==self.UP:
                self.monster_pos = (self.monster_pos[0] + 1, self.monster_pos[1])
            elif action==self.DOWN:
                self.monster_pos = (self.monster_pos[0] - 1, self.monster_pos[1])
            elif action==self.LEFT:
                self.monster_pos = (self.monster_pos[0], self.monster_pos[1] + 1)
            elif action==self.RIGHT:
                self.monster_pos = (self.monster_pos[0], self.monster_pos[1] - 1)

    def game_won(self, player_move):
        """
        Determines whether a game winning condition exists for the player or monster
        """
        if player_move:
            if self.has_gold and self.player_pos == self.exit_pos:
                return True
            else:
                return False
        else:
            if self.num_moves == self.max_moves or self.monster_pos == self.player_pos:
                return True
            else:
                return False

    def get_actions(self, player_move):
        '''Generates a list of possible moves'''
        moves = []
        
        if player_move:
            if self.player_pos[0]>0 and self.marks[self.player_pos[0]-1, self.player_pos[1]]==' ':
                moves.append(self.UP)
                moves.append(self.UP_BUILD)
            if self.player_pos[0]<self.marks.shape[0]-1 and self.marks[self.player_pos[0]+1, self.player_pos[1]]==' ':
                moves.append(self.DOWN)
                moves.append(self.DOWN_BUILD)
            if self.player_pos[1]>0 and self.marks[self.player_pos[0], self.player_pos[1]-1]==' ':
                moves.append(self.LEFT)
                moves.append(self.LEFT_BUILD)
            if self.player_pos[1]<self.marks.shape[1]-1 and self.marks[self.player_pos[0], self.player_pos[1]+1]==' ':
                moves.append(self.RIGHT)
                moves.append(self.RIGHT_BUILD)
        else:
            if self.monster_pos[0]>0 and self.marks[self.monster_pos[0]-1, self.monster_pos[1]]==' ':
                moves.append(self.UP)
            if self.monster_pos[0]<self.marks.shape[0]-1 and self.marks[self.monster_pos[0]+1, self.monster_pos[1]]==' ':
                moves.append(self.DOWN)
            if self.monster_pos[1]>0 and self.marks[self.monster_pos[0], self.monster_pos[1]-1]==' ':
                moves.append(self.LEFT)
            if self.monster_pos[1]<self.marks.shape[1]-1 and self.marks[self.monster_pos[0], self.monster_pos[1]+1]==' ':
                moves.append(self.RIGHT)
            moves.append('') # stay move
                   
        return moves
    
    def no_more_moves(self, player_move):
        """
        Determines whether there are any moves left for player or monster
        """
        return len(self.get_actions(player_move))==0

    # TODO - self method should run minimax to determine the value of each move
    # Then make best move for the computer by placing the mark in the best spot
    def make_comp_move(self):        
        # Make AI move
        best_action = self.alpha_beta_search()
        self.make_move(best_action, False)
    
    def is_terminal(self):
        """
        Determines if the current board state is a terminal state
        """
        #Check to see if we have a terminal state irrespective of either player 
        if self.no_more_moves(True) or self.game_won(True) or self.no_more_moves(False) or self.game_won(False):
            self.has_reached_terminal = True
            return True
        else:
            return False
    
    """ TODO - Functions for needed for ALPHA/BETA SEARCH """
    def alpha_beta_search(self):
        #Flag to determine if a terminal state has reached
        self.has_reached_terminal = False
        #Number of depth reached
        self.depth_reached = 0
        #Number of branches pruned
        self.num_pruned = 0
        #Call max value
        v, best_action = self.max_value(-math.inf, math.inf)
        #Return the best action
        return best_action

    def max_value(self, alpha, beta):
        #Check if it has reached a terminal state
        if not self.has_reached_terminal:
            #Increase depth reached count
            self.depth_reached = self.depth_reached + 1;
        #Check if current state is a terminal state
        if self.is_terminal():
            #Get the utility of the terminal state
            return self.get_utility(), None
        v = -math.inf
        #Get all actions
        actions = self.get_actions(False)
        #Loop through all actions
        for action in self.get_actions(False):
            #Make the move
            self.make_move(action, False)
            #Call min value
            min_val, b = self.min_value(alpha, beta)
            #Undo the move
            self.undo_move(action, False)
            if min_val > v:
                #Set v and best action
                v = min_val
                best_action = action
            if v >= beta or v == 1:
                #Since terminal state can only hold 2 values (-1/1), prune as soon as we get a 1 and add rest of pruned branches because technically those action will get ignored due to v == 1
                self.num_pruned = self.num_pruned + (len(actions)-actions.index(action))-1
                return v, best_action
            alpha = max(alpha, v)
        #Finally, return the value for v and best action
        return v, best_action

    def min_value(self, alpha, beta):
        #Check if it has reached a terminal state
        if not self.has_reached_terminal:
            #Increase depth reached count
            self.depth_reached = self.depth_reached + 1;
        #Check if current state is a terminal state
        if self.is_terminal():
            #Get the utility of the terminal state
            return self.get_utility(), None
        v = math.inf
        #Get all actions
        actions = self.get_actions(True)
        #Loop through all actions
        for action in self.get_actions(True):
            #Make the move
            self.make_move(action, True)
            #Call max value
            max_val, b = self.max_value(alpha, beta)
            #Undo the move
            self.undo_move(action, True)
            if max_val < v:
                #Set v and best action
                v = max_val
                best_action = action
            if v <= alpha or v == -1:
                #Since terminal state can only hold 2 values (-1/1), prune as soon as we get a -1 and add rest of pruned branches because technically those action will get ignored due to v == -1
                self.num_pruned = self.num_pruned + (len(actions)-actions.index(action))-1
                return v, best_action
            beta = min(beta, v)
        #Finally, return the value for v and best action
        return v, best_action

    def get_utility(self):
        #If player wins, then return -1. Otherwise, 1
        if self.game_won(True):
            return -1
        else:
            return 1

###########################            
### Program starts here ###
###########################        

# Print out the header info
print("CLASS: Artificial Intelligence, Lewis University")
print("NAME: Deepkumar Patel")

# Define constants
LOST = 0
WON = 1 
       
# Create the game board of the given size and print it
board = GenGameBoard(4)
board.print_board(True)  
        
# Start the game loop
while True:
    # *** Player's move ***        
    
    # Try to make the move and check if it was possible  
    print("Player's Move #", (board.num_moves+1))
    possible_moves = board.get_actions(True)
    move = input("Choose your move "+str(possible_moves)+": ")
    while move not in possible_moves:
        print("Not a valid move")
        move = input("Choose your move "+str(possible_moves)+": ")
    board.make_move(move, True)

    # Check for gold co-location    
    if not board.has_gold and board.player_pos==board.gold_pos:
        board.has_gold = True
    
    # Display the board
    board.print_board(True)
            
    # Check for ending condition
    # If game is over, check if player won and end the game
    if board.game_won(True):
        # Player won
        result = WON
        break
    elif board.no_more_moves(True):
        # No moves left -> lost
        result = LOST
        break
            
    # *** Computer's move ***
    board.make_comp_move()
    
    # Print out the board again
    board.print_board(False)    
    
    # Check for ending condition
    # If game is over, check if computer won and end the game
    if board.game_won(False):
        # Computer won
        result = LOST
        break
        
# Check the game result and print out the appropriate message
print("GAME OVER")
if result==WON:
    print("You Won!")            
else:
    print("You Lost!")

