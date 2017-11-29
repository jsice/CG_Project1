from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from gl_utils import *
import numpy as np
from PIL import Image

n_vertices, positions, normals, uvs, vao, vbo = 0, None, None, None, None, None
texture_id = None
angle = 0
updown = 0
walk = 0

def load_obj(filename):
    global n_vertices, positions, colors, normals, uvs, vao, vbo
    try:
        f = open(filename, 'r')
    except:
        print("%s not found!" % filename)
        exit(1)
    lines = f.read().split('\n')
    f.close()

    pos = []
    nor = []
    fac = []

    n_vertices = 0
    for line in lines:
        if len(line) == 0 or line[0] == '#': continue
        vals = line.split()
        if vals[0] == 'v':
            pos.append(vals[1:])
            n_vertices += 1
        elif vals[0] == 'vn':
            nor.append(vals[1:])
        elif vals[0] == 'f':
            fac += vals[1:]
    n_vertices = len(fac)
    positions = zeros((n_vertices, 4), dtype=float32)
    colors = zeros((n_vertices, 3), dtype=float32)
    normals = zeros((n_vertices, 3), dtype=float32)
    uvs = zeros((n_vertices, 3), dtype=float32)

    cnt = 0
    for f in fac:
        vals = f.split('/')
        positions[cnt] = pos[int(vals[0])-1] + [1]
        normals[cnt] = nor[int(vals[2])-1]
        colors[cnt] = [1,0,0]
        uvs[cnt] = pos[int(vals[0])-1]
        cnt += 1
    print("Loaded %d vertices" % n_vertices)


def init_tex():
    global tex_id

    glClearColor(0, 0, 0, 0)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    wallpaper = "texture/heic0813c_publication.jpg"
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
    glScalef(50,50,50)

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glEnable(GL_TEXTURE_2D)

    glBegin(GL_QUADS)
    for vertex in positions:
        glTexCoord3fv(vertex[0:3])
        glVertex3fv(vertex[0:3])
    glEnd()

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
        if -9 < walk <= 20:
            print(walk)
            walk -= 1
        else:
            walk = -9

    elif key == 's':
        if -9 <= walk < 20:
            print(walk)
            walk += 1
        else:
            walk = 20

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
    load_obj("room.obj")
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    init_tex()
    glutMainLoop()

if __name__ == "__main__":
    main()    