#include <iostream>

#include "dstar_lite/dstar.h"

int main(int argc, char **argv) {
  // Check that there are 3 arguments, aka argc == 4.
  if (argc != 4) {
    std::cerr << "Terminating. Incorrect number of arguments."
              << "Expected 3." << std::endl;
    return EXIT_FAILURE;
  }

  // Parse grid.
  std::string flat_grid = argv[3];

  // Construct a grid.
  const uint grid_size = (uint)std::sqrt(flat_grid.size());
  std::vector<std::vector<int>>
      occupancy_grid (grid_size, vector<int>(grid_size));

  // Parse input
  // Retrieve start position
  const uint start = std::atoi(argv[1]);
  std::array<uint, 2> start_indices;
  start_indices[0] = std::floor(start / grid_size); //x index
  start_indices[1] = start - start_indices[0] * grid_size; //y index

  // Retrieve goal position
  const uint goal = std::atoi(argv[2]);
  std::array<uint, 2> goal_indices;
  goal_indices[0] = std::floor(goal / grid_size); //x index
  goal_indices[1] = goal - goal_indices[0] * grid_size; //y index

  // Reshape input to a grid with x, y coordinates
  for (uint i = 0, j = 0, k = 0; k < flat_grid.size(); k++) {
    j = k % grid_size;

    // Check that we are not out of bounds.
    if ( i < grid_size && j < grid_size) {
      const int a = 1 - ((int)flat_grid.at(k) - 48);
      occupancy_grid[i][j] = a;
    } else {
      std::cerr << "Index out of bounds, check that"
                   " input grid is squared." << std::endl;
      return EXIT_FAILURE;
    }

    if (j == (grid_size - 1)) {
      i++;
    }
  }

  Dstar* dstar = new Dstar();
  dstar->init(start_indices[0], start_indices[1], goal_indices[0], goal_indices[1]);
  for (uint i = 0; i < occupancy_grid.size(); i++) {
    for (uint j = 0; j < occupancy_grid.at(i).size(); j++) {
      if (occupancy_grid.at(i).at(j) == 1) {
        dstar->updateCell(i, j, -1);
      }
    }
  }

  if (!dstar->replan()) {
    std::cerr << "No found path to goal" << std::endl;
    return EXIT_FAILURE;
  }

  list<state> path = dstar->getPath();
  for(const state& waypoint: path) {
    // Send path to cout.
    std::cout << waypoint.x * grid_size + waypoint.y << std::endl;
  }

  if (debug_gui) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(1000, 800);
    glutInitWindowPosition(50, 20);

    window = glutCreateWindow("Dstar Visualizer");

    glutDisplayFunc(&DrawGLScene);
    glutIdleFunc(&DrawGLScene);
    glutReshapeFunc(&ReSizeGLScene);
    glutKeyboardFunc(&keyPressed);
    glutMouseFunc(&mouseFunc);
    glutMotionFunc(&mouseMotionFunc);

    InitGL(800, 600);

    dstar->draw();

    glutMainLoop();
  }

  return EXIT_SUCCESS;
}
