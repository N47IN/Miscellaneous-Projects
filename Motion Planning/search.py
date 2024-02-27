# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:45:50 2023

@author: Bijo Sebastian
"""

"""
Implement your search algorithms here
"""

import operator
import math

class Node():

  def __init__(self,val,action, parent,depth=0,cost =0):
    self.val = val
    self.parent = parent
    self.depth = depth
    self.action = action
    self.cost = cost

def heuristic_1(problem, state):
  goal = problem.getGoalState()
  dist = math.sqrt((state[0]-goal[0])**2+(state[1]-goal[1])**2)
  return dist


def heuristic_2(problem, state):
  goal = problem.getGoalState()
  dist = abs(state[0]-goal[0])+ abs(state[1]-goal[1])
  return dist

def weighted_AStarSearch(problem, heuristic_ip):

  start_state = problem.getStartState()
  goal_state = problem.getGoalState()
  closed = []
  depth = 0
  fringe_nodes = []
  fringe = []
  actions = []
  start_state = Node(start_state, action=None, parent= None)
  fringe_nodes.append(start_state)
  fringe.append(start_state.val)
  curr_node = fringe_nodes.pop(0)
  expanded = 0
  temp_id = 1
  temp = 100000
  
  while not problem.isGoalState(curr_node.val):     
   next_state = problem.getSuccessors(curr_node.val)
   
   h = heuristic_2(problem,state=curr_node.val)
   for i in next_state:
     depth = curr_node.depth + i[2]
     child = Node(i[0],action = i[1],parent = curr_node,depth=depth,cost=depth+h)
     if i[0] not in closed and i[0] not in fringe:       
      fringe.append(child.val)
      fringe_nodes.append(child)
   
   for i in range(len(fringe_nodes)):
     if fringe_nodes[i].cost <= temp :
       temp = fringe_nodes[i].cost
       temp_id =i
   curr_node = fringe_nodes.pop(temp_id)
   fringe.pop(temp_id)
   temp = 10000
   #print(fringe)
   closed.append(curr_node.val)
   expanded +=1



  while curr_node.action:
      actions.append(curr_node.action)
      curr_node = curr_node.parent

  actions = actions[::-1]
  return actions
