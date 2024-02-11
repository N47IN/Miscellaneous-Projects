# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:45:50 2023

@author: Bijo Sebastian
"""


"""
Implement your search algorithms here
"""

import operator

def depthFirstSearch(problem):
  """
  Your search algorithm needs to return a list of actions that reaches the goal
  Strategy: Search the deepest nodes in the search tree first
  """
  "*** YOUR CODE HERE ***"

def breadthFirstSearch(problem):
  """
  Your search algorithm needs to return a list of actions that reaches the goal
  Strategy: Search the shallowest nodes in the search tree first.
  """
  start_state = problem.getStartState()
  goal_state = problem.getGoalState()
  fringe={}
  depth = 0
  fringe.update({repr(start_state):0})
  expanded = 0
  curr_state = start_state
  closed = []
  fringee =[curr_state]
  while curr_state !=goal_state:
    next_state = problem.getSuccessors(curr_state)
    fringe.popitem()
    fringee.pop()
    closed.append(curr_state)
    
  
    
    depth+=1
    path = []
    for i in next_state:
     if i[0] not in closed:
      if repr(i[0]) not in fringe.keys() or fringe[repr(i[0])]<depth:
        try :
          if fringe[repr(i[0])]<depth:
           fringe.pop(repr(i[0]))
           fringee.pop(i[0])

        except :
          pass

      fringe.update({repr(i[0]):depth})
      fringee.append(i[0])
    curr_state = fringee[-1]
    print(curr_state)
    expanded +=1
    print(expanded)




 


def uniformCostSearch(problem):
  """
  Your search algorithm needs to return a list of actions that reaches the goal
  Strategy: Search the node of least total cost first.
  """
  "*** YOUR CODE HERE ***"
  
