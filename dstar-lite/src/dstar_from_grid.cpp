/* dstar_from_grid.cpp
 */

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
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <unistd.h>

#include "dstar_lite/dstar.h"

int hh, ww;

int window;
Dstar *dstar;

int scale = 50;
int mbutton = 0;
int mstate = 0;

bool b_autoreplan = false;

void InitGL(int Width, int Height) {
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

void ReSizeGLScene(int Width, int Height) {
  hh = Height;
  ww = Width;

  glViewport(0,0,Width,Height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0,Width,0,Height,-100,100);
  glMatrixMode(GL_MODELVIEW);
}

void DrawGLScene() {
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

void keyPressed(unsigned char key, int x, int y) {
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
    dstar->init(1, 1, 3, 3);
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

void mouseMotionFunc(int x, int y) {
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

  // Init GLUT
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);

  // Parse csv file with world grids.
  // Currently only loads the first grid in the csv file...
  ifstream file("../resources/gridworld_8.csv");
  std::vector<string> value;
  string tmp;
  if (file.is_open()) {
    while (getline(file, tmp)) {
      value.push_back(tmp);
    }
    file.close();
  } else {
    cout << "Could not open the csv file." << endl;
  }

  // Construct a grid.
  const uint grid_size = (uint)std::sqrt(value.size());
  std::vector<std::vector<int>> occupancy_grid (grid_size, vector<int>(grid_size));


  // Reshape input to a grid with x, y coordinates
  bool first = true;
  for (uint i, j, k = 0; k < value.size(); k++) {
    j = k % ((uint)std::sqrt(value.size()));

    // Check that we are not out of bounds.
    if ( i < grid_size && j < grid_size) {
      occupancy_grid[i][j] = std::atoi(&value.at(k).at(0));
    } else {
      cerr << "Index out of bounds, check that input grid is squared." << endl;
    }

    if (j == 0) {
      if (first) {
        first = false;
      } else {
        i++;
      }
    }
  }

  // Initialize window for visualization.
  glutInitWindowSize(500, 500);
  glutInitWindowPosition(20, 20);

  window = glutCreateWindow("Dstar Visualizer");

  glutDisplayFunc(&DrawGLScene);
  glutIdleFunc(&DrawGLScene);
  glutReshapeFunc(&ReSizeGLScene);
  glutKeyboardFunc(&keyPressed);
  glutMouseFunc(&mouseFunc);
  glutMotionFunc(&mouseMotionFunc);

  InitGL(30, 20);

  dstar = new Dstar();
  dstar->init(3, 2, 6, 6);
  for (uint i = 0; i < occupancy_grid.size(); i++) {
    for (uint j = 0; j < occupancy_grid.at(i).size(); j++) {
      std::cout << "Occ grid vals: " << occupancy_grid[i][j] << '\n';
      if (occupancy_grid.at(i).at(j) == 1) {
        dstar->updateCell(i+1, j+1, -1);
      }
    }
  }
  dstar->draw();


  printf("----------------------------------\n");
  printf("Dstar Visualizer\n");
  printf("Commands:\n");
  printf("[q/Q] - Quit\n");
  printf("[r/R] - Replan\n");
  printf("[a/A] - Toggle Auto Replan\n");
  printf("[c/C] - Clear (restart)\n");
  printf("left mouse click - make cell untraversable (cost -1)\n");
  printf("middle mouse click - move goal to cell\n");
  printf("right mouse click - move start to cell\n");
  printf("----------------------------------\n");

  glutMainLoop();

  return 1;
}
