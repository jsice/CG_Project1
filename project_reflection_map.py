import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image
from numpy import *
from gl_utils import *
from math import *
from ctypes import c_void_p

prog_id, angle = 0, 0
tex_id = None
n_vertices, positions, normals, colors, uvs, centroid, vao, vbo = \
0, None, None, None, None, None, None, None
screenWidth, screenHeight = 800, 600

render_program, shadow_program = 0, 0
uniform_locs = {}


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

def compile_program(vertex_code, fragment_code):
    vert_id = glCreateShader(GL_VERTEX_SHADER)
    frag_id = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(vert_id, vertex_code)
    glShaderSource(frag_id, fragment_code)
    glCompileShader(vert_id)
    glCompileShader(frag_id)
    print_shader_info_log(vert_id, "Vertex Shader")
    print_shader_info_log(frag_id, "Fragment Shader")

    prog_id = glCreateProgram()
    glAttachShader(prog_id, vert_id)
    glAttachShader(prog_id, frag_id)

    glLinkProgram(prog_id)
    print_program_info_log(prog_id, "Link error")
    return prog_id  

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
    global  vao, vbo,prog_id
    cubemap()
    vert_code = b'''
#version 330
uniform mat4 model_mat, view_mat, proj_mat;
in vec3 vPos, vNor;
out vec3 reflectDir;
void main() {

    gl_Position = proj_mat * view_mat * model_mat * vec4(vPos, 1);
    mat4 adjunct_mat = transpose(inverse(model_mat));
    vec3 P = (model_mat * vec4(vPos, 1)).xyz;
    vec3 N = (adjunct_mat * vec4(vNor, 0)).xyz;
    vec3 ePos = (inverse(view_mat) * vec4(0,0,0,1)).xyz;
    reflectDir = reflect(normalize(P - ePos), N);
    //reflectDir = refract(normalize(P - ePos), N, 1/1.5);
}
                '''
    frag_code = b''' 
#version 130
uniform samplerCube cube_map;
in vec3 reflectDir;
out vec4 gl_FragColor;
void main() {
    gl_FragColor = mix(texture(cube_map, reflectDir), vec4(0,1,0,1), 0);
}
                '''                
    prog_id = compile_program(vert_code, frag_code)
    glUseProgram(prog_id)
    for name in ["model_mat", "view_mat", "proj_mat","cube_map"]:
        uniform_locs[name] = glGetUniformLocation(prog_id, name)
    
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(2)
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
    glBufferData(GL_ARRAY_BUFFER, positions, GL_DYNAMIC_DRAW)
    location = glGetAttribLocation(prog_id, "vPos")
    glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(location)


    glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
    glBufferData(GL_ARRAY_BUFFER, normals, GL_STATIC_DRAW)
    location = glGetAttribLocation(prog_id, "vNor")
    if location != -1:
        glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(location)    

    glBindVertexArray(0)
 

def cubemap():
    global tex_id
    im = Image.open("cube_map/street_HCmap.jpg")
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
    vertices = [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1], [1,-1,-1],[1,-1,1],[1,1,1],[1,1,-1], 
    [-1,-1,-1],[-1,-1,1],[1,-1,1],[1,-1,-1], [-1,1,-1],[-1,1,1],[1,1,1],[1,1,-1],
    [-1,1,1],[-1,-1,1],[1,-1,1],[1,1,1], [-1,1,-1],[-1,-1,-1],[1,-1,-1],[1,1,-1]] 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    e_pos = (20*cos(angle*pi/180), 3, 20*sin(angle*pi/180))
    e_at = (-1,3,0)
   
    glUseProgram(0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(40., screenWidth/screenHeight, 0.01, 500.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(*e_pos, *e_at, 0, 1, 0)
    glScalef(50,50,50)

    # glRotatef(angle,0,1,0)
    glBegin(GL_QUADS)
    for vertex in vertices:
        texCoord = [-vertex[0]] + [vertex[1]] + [-vertex[2]]
        glTexCoord3fv(texCoord)
        glVertex3fv(vertex)

    glEnd()

    glUseProgram(prog_id) 
    proj_mat = Perspective(40., screenWidth/screenHeight, 0.01, 500.0)
    glUniformMatrix4fv(uniform_locs["proj_mat"], 1, True, proj_mat.A)
    view_mat = LookAt(*e_pos, *e_at, 0, 1, 0)
    glUniformMatrix4fv(uniform_locs["view_mat"], 1, True, view_mat.A)
    model_mat =  Scale(3,3,3)
    glUniformMatrix4fv(uniform_locs["model_mat"], 1, True, model_mat.A)
    
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, n_vertices)
    glutSwapBuffers()

def animate():
    global angle
    angle = angle + 0.25
    glutPostRedisplay()

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(screenWidth, screenHeight)
    glutInitWindowPosition(50, 50)    
    glutCreateWindow(b"Reflection map")
    glutDisplayFunc(display)
    load_tri("models/horse.tri")
    glutIdleFunc(animate)
    glEnable(GL_DEPTH_TEST)
    init()
    glutMainLoop()

if __name__ == "__main__":
    main()    