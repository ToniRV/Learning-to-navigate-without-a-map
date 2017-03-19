
## References:
This code was mainly imported from
https://github.com/ArekSredzki/dstar-lite

Here we just tweaked the interface such that it is suitable to our needs. Implementation details and references can be read [here](https://github.com/ArekSredzki/dstar-lite).

Below is provided part of the README of the repository used as reference.

# D*-Lite Class

This software is an implementation of the D*-Lite algorithm as explained in [Koenig, 2002].

This is the non-optimized version as explained in Figure 5 of the paper. There are a few minor improvements that were made to this algorithm explained in section 3 below.

This source is released under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007 available at: http://www.gnu.org/licenses/gpl.html

Please note this is an early release and the software still has small bugs.

## Running the dstar test program:
you will need to have the OpenGL/GLUT libraries installed for this to work. But you do not need them to use the Dstar class in your own program.
```
$ git clone git@github.com:ToniRV/Learning-to-navigate-without-a-map.git
$ cd Learning-to-navigate-without-a-map/dstar-lite
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./dstar_draw # Use the Commands below.
```
### Given a grid map
First, follow the instructions [here](https://github.com/ToniRV/Learning-to-navigate-without-a-map#instructions-to-use-the-dataset) to download sample datasets.
Second, TODO

### Commands
* [q/Q] - Quit
* [r/R] - Replan
* [a/A] - Toggle Auto Replan
* [c/C] - Clear (restart)
* left mouse click - make cell untraversable (cost -1)
* middle mouse click - move goal to cell
* right mouse click - move start to cell

### The cell colors are as follows:
* Red - untraversable
* Green - traversable but with changed cost
* Red/Green with small purple square - The cell is on the openList
* Yellow - start cell
* Purple - goal cell

## Implementational Details:
Here is a list of the more interesting tweaks that we applied to improve the D* Lite algorithm explained in [Koenig, 2002].

1. The Open Hash and Lazy Remove:
 In order to speed up the time it takes to add/remove/search on the open list we used both a stl::priority_queue and a "stl"::hash_map to store states. The queue keeps the states in sorted order so it is easy to find the next best state while the hash is used to quickly determine what states are in the queue. When a cell is inserted into the openlist it is pushed onto the queue and put into the hash table. In order to check if a cell is on the open list one can just check if it is in the hash table. The hash table also stores a hash of the cells key so cells that are outdated in the queue can still be removed. Any time a cell is popped off the queue we check if it is in the hash, if not it is discarded an a new one is chosen.

2. Euclidean Path Optimization
 Obtaining a path from the D* generated cost map is generally done by starting at the start node and doing a greedy search by expanding successor nodes with the lowest cost to  goal. This approach can generate a path that starts heading out 45 degrees toward the goal instead of straight at it. This happens because the 8-way connected distance is an approximation and there is no difference between taking all of the angular moves first and taking all of the straight moves. In order to generate a path that is closer to the true shortest cost we added a simple modification to the greedy search. When we compare the costs to all of the successor cells we choose the one that minimizes:
```cpp
(cellA.g != cellB.g) return (cellA.g <  cellB.g)
return ((euclidean_dist(cellA,start) + euclidean_dist(cellA,goal))
       < (euclidean_dist(cellB,start) + euclidean_dist(cellB,goal)))
```
This means we break ties by choosing the cell that lies closest to the straight line between the start and goal states.

3. Goal Changes
 The D* Lite algorithm explained in [Koenig, 2002] doesn't include code to handle the goal changing locations. To do this we simply clear the map, change the location of the goal, re-add all the obstacles, and replan. There is probably a more efficient way of dealing with this but this modification worked great for our purposes.

4. Max Steps
 If there is no path to the goal D* can have a hard time detecting it and will likely loop forever. In order to partially deal with this issue the search will automatically return after a set number of node expansions (set to 'maxsteps'). After the search returns it can start again where it left off with another call to replan().

