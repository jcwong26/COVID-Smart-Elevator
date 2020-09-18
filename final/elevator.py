# Elevator class

import random

class Elevator:

    # Initialize new Elevator object
    def __init__(self, num_floors):
        self.num_floors = num_floors

        # set of requested floors for the elevator to travel to
        self.up_requested = {}
        self.down_requested = {}

        # ordered list of floors the elevator will go to
        self.path = []

        # start at first floor, default going up
        self.cur_floor = 1
        self.direction = "up"
        
        # set of floors with people waiting so floors can be added to the path when visited
        self.up_waiting = {}
        self.down_waiting = {}

    # Elevator is called from a floor
    def call(self, floor, direction):
        if direction == "up":
            # Add floor to path if it is the same direction and above
            if floor > self.cur_floor and self.direction == "up":
                self.add_floor(floor)
            else:
                self.up_requested.add(floor)

        else:
            # Add floor to path if it is the same direction and below
            if floor < self.cur_floor and self.direction == "down":
                self.add_floor(floor)
            else:
                self.down_requested.add(floor)


    # Add floor to path (floor requested inside elevator)
    def add_floor(self, floor):
        if floor not in path:
            self.path.append(floor)

            if self.direction == "up":
                self.path.sort()
            else:
                self.path.sort(reverse=True) 


    # Visit floor from path
    def visit_floor(self):
        # remove from path
        self.path = self.path[1:]

        #pick up customer and add floor
        if self.direction == "up" and self.cur_floor in self.up_waiting:
            add_floor(int(input("Enter floor to go up: "))) # Replace w Hand Gesture Recognition
            self.up_waiting.remove(self.cur_floor)
        elif self.direction == "down" and self.cur_floor in self.down_waiting:
            add_floor(int(input("Enter floor to go down: "))) #Replace w Hand Gesture Recognition
            self.down_waiting.remove(self.cur_floor)

    # Move to next floor
    def move(self):
        if self.direction == "up":
            self.cur_floor = self.cur_floor + 1
        elif self.direction == "down":
            self.cur_floor = self.cur_floor - 1

        self.visit_floor()

        # Check if path is empty and change direction accordingly
        if not self.path:
            if direction == "up":
                self.path.extend(self.down_requested)
                self.down_requested = {}
                self.direction = "down"
            elif direction == "down":
                self.path.extend(self.up_requested)
                self.up_requested = {}
                self.direction = "up"

    ## Todo: add condition for when the elevator is not moving to moving and vice versa

                

