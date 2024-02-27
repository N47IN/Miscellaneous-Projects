# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:16:30 2023

@author: Bijo Sebastian
"""

#Definitions based on color map
start_id = 1
goal_id = 8
obstacle_id = 16
beacon_id = 12
free_space_id1 = 3
free_space_id2 = 18
free_space_id1_cost = 1
free_space_id2_cost = 3
fringe_id = 4
expanded_id = 6
path_id = 10


class Maps:
    """
    This class outlines the structure of the maps
    """    
    map_data = []
    start = []
    goal = []
    
#Maze maps
map_4 = Maps()
map_4.map_data = [
     [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
     [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
     [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  8, 16],
     [16,  3,  3,  3,  3,  3,  3,  3,  3, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],         
     [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
     [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
     [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
     [16,  3,  3, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,  3,  3, 16],
     [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16,  3,  3,  3,  3,  3,  3, 16],
     [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16,  3,  3,  3,  3,  3,  3, 16],
     [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16,  3,  3,  3,  3,  3,  3, 16],
     [16, 16, 16, 16, 16, 16, 16, 16, 16,  3,  3,  3, 16,  3,  3,  3,  3,  3,  3, 16],
     [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
     [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
     [16,  1,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
     [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]]
map_4.start = [14, 1]
map_4.goal = [2, 18]

maps_dictionary = { 4:map_4}

 
