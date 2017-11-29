from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from gl_utils import *
import numpy as np
from PIL import Image

n_vertices, positions, normals, uvs, vao, vbo = 0, None, None, None, None, None
angle = 0
updown = 0
walk = 0

def init_gl():
    global tex_id

    glClearColor(0, 0, 0, 0)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    wallpaper = "beige-wallpaper.jpg"
    try:
        im = Image.open(wallpaper)
    except:
        print("Error:", sys.exc_info()[0])
        raise  
    w = im.size[0]
    h = im.size[1]
    image = im.tobytes("raw", "RGB", 0)
    tex_id = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, 3, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, image)


def display():
    global angle, updown, walk

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluPerspective(80, 1, 0.1, 100)
    # eye at up
    gluLookAt(0, 0, walk, 0, updown, -10, 0, 1, 0)
    glRotatef(angle, 0, 1, 0)

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glEnable(GL_TEXTURE_2D)

    glBegin(GL_QUADS)
    # Floor
    glVertex3f(-40, -40, -40)
    glVertex3f(-40, -40, 40)
    glVertex3f(40, -40, 40)
    glVertex3f(40, -40, -40)

    # Ceiling
    glVertex3f(-40, 40, -40)
    glVertex3f(40, 40, -40)
    glVertex3f(40, 40, 40)
    glVertex3f(-40, 40, 40)

    # Walls
    glTexCoord2f(0, 0)
    glVertex3f(-40, -40, 40)
    glTexCoord2f(0, 1)
    glVertex3f(40, -40, 40)
    glTexCoord2f(1, 1)
    glVertex3f(40, 40, 40)
    glTexCoord2f(1, 0)
    glVertex3f(-40, 40, 40)

    glTexCoord2f(0, 0)
    glVertex3f(-40, -40, -40)
    glTexCoord2f(0, 1)
    glVertex3f(40, -40, -40)
    glTexCoord2f(1, 1)
    glVertex3f(40, 40, -40)
    glTexCoord2f(1, 0)
    glVertex3f(-40, 40, -40)

    glTexCoord2f(0, 0)
    glVertex3f(40, 40, 40)
    glTexCoord2f(0, 1)
    glVertex3f(40, -40, 40)
    glTexCoord2f(1, 1)
    glVertex3f(40, -40, -40)
    glTexCoord2f(1, 0)
    glVertex3f(40, 40, -40)

    glTexCoord2f(0, 0)
    glVertex3f(-40, 40, 40)
    glTexCoord2f(0, 1)
    glVertex3f(-40, -40, 40)
    glTexCoord2f(1, 1)
    glVertex3f(-40, -40, -40)
    glTexCoord2f(1, 0)
    glVertex3f(-40, 40, -40)
    glEnd(); 

    glDisable(GL_TEXTURE_2D)
    glutSwapBuffers()

def keyboard(key, x, y):
    global angle, updown, walk
    key = key.decode("utf-8")
    if key == 'd':
        angle += 1
    elif key == 'a':
        angle -= 1
    elif key == 'w':
        if 0 < walk <= 40:
            walk -= 1
        else:
            walk = 0

    elif key == 's':
        if 0 <= walk < 40:
            walk += 1
        else:
            walk = 40

    elif key == 'z':
        updown -= 1
    elif key == 'x':
        updown += 1    
    elif ord(key) == 27:
        exit(0)

    glutPostRedisplay()

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutInitWindowPosition(50, 50)    
    glutCreateWindow(b"Room")
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    # glEnable(GL_DEPTH_TEST)
    init_gl()
    glutMainLoop()

if __name__ == "__main__":
    main()    