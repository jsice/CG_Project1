from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from gl_utils import *
import numpy as np
from PIL import Image

prog_id, angle = 0, 0
tex_id = None
n_vertices_obj, positions_obj, normals_obj, colors_obj, uvs_obj, centroid_obj, vao_obj, vbo_obj = 0, None, None, None, None, None, None, None
n_vertices, positions, normals, colors, uvs, centroid, vao, vbo = 0, None, None, None, None, None, None, None
screenWidth, screenHeight = 800, 600
uniform_locs = {}
angle = 0
updown = 0
walk = 0

def print_shader_info_log(shader, prompt=""):
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not result:
        print("%s: %s" % (prompt, glGetShaderInfoLog(shader).decode("utf-8")))
        return -1
    else:
        return 0

def print_program_info_log(program, prompt=""):
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not result:
        print("%s: %s" % (prompt, glGetProgramInfoLog(program).decode("utf-8")))
        return -1
    else:
        return 0



def load_obj(filename):
    global n_vertices_obj, positions_obj, colors_obj, normals_obj, uvs_obj, vao, vbo
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

    n_vertices_obj = 0
    for line in lines:
        if len(line) == 0 or line[0] == '#': continue
        vals = line.split()
        if vals[0] == 'v':
            pos.append(vals[1:])
            n_vertices_obj += 1
        elif vals[0] == 'vn':
            nor.append(vals[1:])
        elif vals[0] == 'f':
            fac += vals[1:]
    n_vertices_obj = len(fac)
    positions_obj = zeros((n_vertices_obj, 4), dtype=float32)
    colors_obj = zeros((n_vertices_obj, 3), dtype=float32)
    normals_obj = zeros((n_vertices_obj, 3), dtype=float32)
    uvs_obj = zeros((n_vertices_obj, 3), dtype=float32)

    cnt = 0
    for f in fac:
        vals = f.split('/')
        positions_obj[cnt] = pos[int(vals[0])-1] + [1]
        normals_obj[cnt] = nor[int(vals[2])-1]
        colors_obj[cnt] = [1,0,0]
        uvs_obj[cnt] = pos[int(vals[0])-1]
        cnt += 1
    print("Loaded %d vertices" % n_vertices_obj)

def load_tri(filename):
    global n_vertices, positions, normals, uvs, colors
    try:
        f = open(filename, 'r')
    except:
        print("%s not found!" % filename)
        exit(1)
    lines = f.read().split('\n')
    f.close()
    n_vertices = len(lines)
    positions = zeros((n_vertices, 3), dtype=float32)
    colors = zeros((n_vertices, 3), dtype=float32)
    normals = zeros((n_vertices, 3), dtype=float32)
    uvs = zeros((n_vertices, 2), dtype=float32)
    cnt = 0
    for line in lines:
        if len(line) == 0 or line[0] == '#': continue
        vals = line.split()
        positions[cnt] = vals[0:3]
        normals[cnt] = vals[3:6]
        uvs[cnt] = vals[6:8]
        cnt +=1 
    n_vertices = cnt
    # normals[:] = (normals+1)*0.5
    print("Loaded %d vertices" % n_vertices)

def init():
    global  vao_obj, vbo_obj, vao, vbo,prog_id
    cubemap()

    vert_id = glCreateShader(GL_VERTEX_SHADER)
    frag_id = glCreateShader(GL_FRAGMENT_SHADER)

    vert_code = b'''
#version 110
varying vec3 normal;
void main()
{
   gl_Position = ftransform();
   normal = (gl_NormalMatrix * gl_Normal).xyz;
}'''
    frag_code = b'''
#version 110
varying vec3 normal;
void main()
{
   vec3 n = normalize(normal);
   vec3 color = (n + vec3(1.0, 1.0, 1.0)) * 0.5;
   gl_FragColor = vec4(color, 1);
}'''
    glShaderSource(vert_id, vert_code)
    glShaderSource(frag_id, frag_code)

    glCompileShader(vert_id)
    glCompileShader(frag_id)
    print_shader_info_log(vert_id, "Vertex Shader")
    print_shader_info_log(frag_id, "Fragment Shader")

    prog_id = glCreateProgram()
    glAttachShader(prog_id, vert_id)
    glAttachShader(prog_id, frag_id)

    glLinkProgram(prog_id)
    print_program_info_log(prog_id, "Link error")

def cubemap():
    global tex_id
    im = Image.open("texture/cubemap_512.png")
    w, h = im.size[0], im.size[1]
    sub_w = int(w/4)
    sub_h = int(h/3)
    box = [[2,1], [0,1], [1,0], [1,2], [1,1], [3,1]]
    cube_side_id = [GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z]
    glActiveTexture(GL_TEXTURE0)
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex_id)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    for i in range(6):
        image = im.crop((box[i][0]* sub_w, box[i][1]*sub_h, (box[i][0]+1)* sub_w, (box[i][1]+1) * sub_h)).tobytes("raw", "RGB", 0)
        gluBuild2DMipmaps(cube_side_id[i], GL_RGB, sub_w, sub_h, GL_RGB, GL_UNSIGNED_BYTE, image)
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
    glEnable(GL_TEXTURE_CUBE_MAP)

def display():
    global angle, updown, walk

    vertices = [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1], [1,-1,-1],[1,-1,1],[1,1,1],[1,1,-1], 
    [-1,-1,-1],[-1,-1,1],[1,-1,1],[1,-1,-1], [-1,1,-1],[-1,1,1],[1,1,1],[1,1,-1],
    [-1,1,1],[-1,-1,1],[1,-1,1],[1,1,1], [-1,1,-1],[-1,-1,-1],[1,-1,-1],[1,1,-1]] 

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluPerspective(100, 1, 0.1, 100)
    # eye at up
    gluLookAt(0, 0, walk, 0, updown, -10, 0, 1, 0)
    glRotatef(angle, 0, 1, 0)
    glScalef(50,50,50)

    glBegin(GL_QUADS)
    for vertex in vertices:
        glTexCoord3fv(vertex[0:3])
        glVertex3fv(vertex[0:3])
    glEnd()

    glUseProgram(prog_id) 
    
    glDrawArrays(GL_TRIANGLES, 0, n_vertices)
    glutSwapBuffers()

def keyboard(key, x, y):
    global angle, updown, walk
    key = key.decode("utf-8")
    if key == 'd':
        angle += 1
    elif key == 'a':
        angle -= 1
    elif key == 'w':
        if -5 < walk <= 15:
            walk -= 1
        else:
            walk = -5
    elif key == 's':
        if -5 <= walk < 15:
            walk += 1
        else:
            walk = 15
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
    glutInitWindowSize(screenWidth, screenHeight)
    glutInitWindowPosition(50, 50)    
    glutCreateWindow(b"Room")
    # load_obj("room.obj")
    glutDisplayFunc(display)
    load_tri("models/ball.tri")
    glutKeyboardFunc(keyboard)
    glEnable(GL_DEPTH_TEST)
    init()
    glutMainLoop()

if __name__ == "__main__":
    main()    