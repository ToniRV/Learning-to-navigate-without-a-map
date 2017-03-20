#include <iostream>

#include "dstar_lite/dstar.h"

#ifdef MACOS
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include <stdlib.h>
#include <unistd.h>
#include <zmq.hpp>
#include <string>
#include <iostream>
#include <sstream>
#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>

#define sleep(n)    Sleep(n)
#endif

#include "dstar_lite/dstar.h"

bool constexpr debug_gui = false;

int hh, ww;

int window;
Dstar *dstar;

int scale = 30;
int mbutton = 0;
int mstate = 0;

bool b_autoreplan = true;

void InitGL(int Width, int Height)
{
  hh = Height;
  ww = Width;

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClearDepth(1.0);

  glViewport(0,0,Width,Height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0,Width,0,Height,-100,100);
  glMatrixMode(GL_MODELVIEW);

}

void ReSizeGLScene(int Width, int Height)
{
  hh = Height;
  ww = Width;

  glViewport(0,0,Width,Height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0,Width,0,Height,-100,100);
  glMatrixMode(GL_MODELVIEW);

}

void DrawGLScene()
{

  usleep(100);

  glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
  glLoadIdentity();
  glPushMatrix();

  glScaled(scale,scale,1);

  if (b_autoreplan) dstar->replan();

  dstar->draw();

  glPopMatrix();
  glutSwapBuffers();

}

void keyPressed(unsigned char key, int x, int y)
{
  usleep(100);

  switch(key) {
  case 'q':
  case 'Q':
    glutDestroyWindow(window);
    exit(0);
    break;
  case 'r':
  case 'R':
    dstar->replan();
    break;
  case 'a':
  case 'A':
    b_autoreplan = !b_autoreplan;
    break;
  case 'c':
  case 'C':
    dstar->init(40,50,140, 90);
    break;
  }
}

void mouseFunc(int button, int state, int x, int y) {

  y = hh -y+scale/2;
  x += scale/2;

  mbutton = button;

  if ((mstate = state) == GLUT_DOWN) {
    if (button == GLUT_LEFT_BUTTON) {
      dstar->updateCell(x/scale, y/scale, -1);
    } else if (button == GLUT_RIGHT_BUTTON) {
      dstar->updateStart(x/scale, y/scale);
    } else if (button == GLUT_MIDDLE_BUTTON) {
      dstar->updateGoal(x/scale, y/scale);
    }
  }
}

void mouseMotionFunc(int x, int y)  {

  y = hh -y+scale/2;
  x += scale/2;

  y /= scale;
  x /= scale;

  if (mstate == GLUT_DOWN) {
    if (mbutton == GLUT_LEFT_BUTTON) {
      dstar->updateCell(x, y, -1);
    } else if (mbutton == GLUT_RIGHT_BUTTON) {
      dstar->updateStart(x, y);
    } else if (mbutton == GLUT_MIDDLE_BUTTON) {
      dstar->updateGoal(x, y);
    }
  }

}

int main(int argc, char **argv) {
  // Prepare our context and socket
  // for request/reply communication with client.
  zmq::context_t context (1);
  zmq::socket_t socket (context, ZMQ_REP);
  socket.bind ("tcp://*:5555");

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

  dstar = new Dstar();
  dstar->init(start_indices[0], start_indices[1]
      , goal_indices[0], goal_indices[1]);
  for (uint i = 0; i < occupancy_grid.size(); i++) {
    for (uint j = 0; j < occupancy_grid.at(i).size(); j++) {
      if (occupancy_grid.at(i).at(j) == 1) {
        dstar->updateCell(i, j, -1);
      }
    }
  }

  uint replan_counter = 0;
  uint update_counter = 0;
  while (true) {
    zmq::message_t request;

    // Wait for next request from client
    socket.recv(&request);
    std::string rpl = std::string(
        static_cast<char*>(request.data()), request.size());

    std::cout << rpl << std::endl;

    if (rpl == "replan") {
      replan_counter++;
      std::cout << "[Dstar cpp] Requested to replan ["
                << replan_counter << "]" << std::endl;

      if (!dstar->replan()) {
        std::cerr << "No found path to goal." << std::endl;
        return EXIT_FAILURE;
      }

      // Parse calculated path.
      list<state> path = dstar->getPath();
      std::string path_string;
      std::ostringstream convert;   // stream used for the conversion
      for(const state& waypoint: path) {
        // Make message reply.
        convert << waypoint.x * grid_size + waypoint.y << '.';
      }
      path_string = convert.str();

      // Send reply back to client
      zmq::message_t reply (path_string.size());
      memcpy (reply.data (), path_string.c_str(), path_string.size());
      socket.send (reply);
    } else if (rpl == "update") {
      update_counter++;
      std::cout << "[Dstar cpp] Requested to update cell ["
                << update_counter << "]" << std::endl;
      // Send reply back to client to acknowledge request.
      zmq::message_t reply (2);
      memcpy (reply.data (), "go", 2);
      socket.send (reply);

      // Wait for cell coordinates to be updated.
      socket.recv(&request);
      std::string rpl = std::string(
      static_cast<char*>(request.data()), request.size());
      istringstream iss(rpl);
      vector<string> x_y_coords{istream_iterator<string>{iss},
                      istream_iterator<string>{}};
      if (x_y_coords.size() != 2) {
        // Send reply back to client with error.
        zmq::message_t reply (2);
        memcpy (reply.data (), "Expected 2 coordinates", 2);
        socket.send (reply);
      } else {
        std::cerr << "X coord of obstacle: " << std::atoi(x_y_coords.at(0).c_str())
                  << "Y coord of obstacle: " << std::atoi(x_y_coords.at(1).c_str())
                  << std::endl;
        dstar->updateCell(std::atoi(x_y_coords.at(0).c_str()),
                          std::atoi(x_y_coords.at(1).c_str()), -1);
        // Send reply back to client of success.
        zmq::message_t reply (2);
        memcpy (reply.data (), "ok", 2);
        socket.send (reply);
      }
    } else {
      std::cerr << "[Dstar cpp] Error, could not understand command: "
                << rpl << std::endl;
      // Send reply back to client
      zmq::message_t reply (5);
      memcpy (reply.data (), "fail", 5);
      socket.send (reply);
    }
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
