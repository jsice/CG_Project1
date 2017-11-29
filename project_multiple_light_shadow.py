import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image
from numpy import *
from gl_utils import *
from math import *

n_vertices, positions, normals, colors, uvs, centroid, vao, vbo = \
0, None, None, None, None, None, None, None
screenWidth, screenHeight = 800, 600
shadow_tex_id1, shadow_tex_id2, shadow_map_size, shadow_FBO1, shadow_FBO2 = 0, 0, 0, 0, 0
render_program, shadow_program, test_program = 0, 0, 0
uniform_locs = {}
uniform_mat_value = {}
uniform3_value = {}
uniform_value = {}
uniform_mat = ["model_mat", "view_mat", "proj_mat"]
uniform3 = ["l1_pos", "l2_pos", "l_dcol", "l_scol", "m_kd", "m_ks"]
uniform = ["m_shininess"]
eye = {}
isLight1Open = 0
isLight2Open = 0

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

def init():
    global shadow_program, render_program, vao, vbo, test_program
    
    vert_code = b'''
#version 330
uniform mat4 model_mat, view_mat, proj_mat, shadow_mat1, shadow_mat2;
in vec3 vPos, vNor, vCol;
in vec2 vTex;
out vec3 fPos, fNor, fCol, e_pos;
out vec4 fSha1, fSha2;
void main() {
    fPos = (model_mat * vec4(vPos, 1)).xyz;
    gl_Position = proj_mat * view_mat * model_mat * vec4(vPos, 1);
    mat4 adjunct_mat = transpose(inverse(model_mat));
    fNor = (adjunct_mat * vec4(vNor, 0)).xyz;
    fCol = vCol;
    e_pos = (inverse(view_mat) * vec4(0,0,0,1)).xyz;
    fSha1 = shadow_mat1 * vec4(vPos, 1);
    fSha2 = shadow_mat2 * vec4(vPos, 1);
}
'''
    frag_code = b'''
#version 330
vec3 ambient_color = vec3(0.15, 0.15, 0.15);
uniform vec3 l2_pos, l1_pos, l_dcol, l_scol, m_kd, m_ks;
uniform float m_shininess;
uniform int isLight1Open, isLight2Open;
uniform sampler2D shadow_map1, shadow_map2;
in vec3 e_pos, fPos, fCol, fNor;
in vec4 fSha1, fSha2;
out vec4 gl_FragColor;

vec3 calculateLightFrom(vec3 pos, vec4 fSha, sampler2D shadow_map) {
    vec3 N = normalize(fNor);
    vec3 L = normalize(pos - fPos);
    float LdotN = max(0, dot(L, N));
    vec3 diffuse = fCol * l_dcol * LdotN;
    vec3 V = normalize(e_pos - fPos);
    vec3 H = normalize(L + V);
    float HdotN = max(0, dot(H, N));
    vec3 specular = m_ks * l_scol * pow(HdotN, m_shininess);
    float visibility = 1.0;
    float bias = 0.001;
    int i, j;
    for(i=-1; i<=1; i++)
        for(j=-1; j<=1; j++)
        if (textureOffset(shadow_map, fSha.st/fSha.w, ivec2(i, j)).r < fSha.z/fSha.w - bias)
            visibility -= 0.1;
    return (diffuse + specular) * visibility;
}
void main() {
    vec3 color_from_light1 = calculateLightFrom(l1_pos, fSha1, shadow_map1);
    vec3 color_from_light2 = calculateLightFrom(l2_pos, fSha2, shadow_map2);

    vec3 color = (color_from_light1 * isLight1Open + color_from_light2 * isLight2Open);

    gl_FragColor = vec4(color + ambient_color, 1);
    gl_FragColor = clamp(gl_FragColor, 0, 1);
}
'''
    render_program = compile_program(vert_code, frag_code)
    glUseProgram(render_program)
    for name in uniform_mat:
        uniform_locs[name] = glGetUniformLocation(render_program, name)
    for name in uniform3:
        uniform_locs[name] = glGetUniformLocation(render_program, name)
    for name in uniform:
        uniform_locs[name] = glGetUniformLocation(render_program, name)
    for name in ["shadow_mat1", "shadow_mat2", "shadow_map1", "shadow_map2", "isLight1Open", "isLight2Open"]:
        uniform_locs[name] = glGetUniformLocation(render_program, name)


    vao = glGenVertexArrays(2)

    glBindVertexArray(vao[0])
    vbo = glGenBuffers(4)

    glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
    glBufferData(GL_ARRAY_BUFFER, positions, GL_DYNAMIC_DRAW)
    location = glGetAttribLocation(render_program, "vPos")
    glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(location)

    glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
    glBufferData(GL_ARRAY_BUFFER, normals, GL_STATIC_DRAW)
    location = glGetAttribLocation(render_program, "vNor")
    if location != -1:
        glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(location)  

    glBindBuffer(GL_ARRAY_BUFFER, vbo[2])
    glBufferData(GL_ARRAY_BUFFER, colors, GL_STATIC_DRAW)
    location = glGetAttribLocation(render_program, "vCol")
    if location != -1:
        glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(location)  

    glBindBuffer(GL_ARRAY_BUFFER, vbo[3])
    glBufferData(GL_ARRAY_BUFFER, uvs, GL_STATIC_DRAW)
    location = glGetAttribLocation(render_program, "vTex")
    if location != -1:
        glVertexAttribPointer(location, 2, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(location)  

    glBindVertexArray(0)


    vert_code = b'''
#version 120
uniform mat4 MVP;
in vec3 vPos;
void main() {
    gl_Position = MVP * vec4(vPos, 1);
}
                '''
    frag_code = b''' 
#version 120
void main() {
}
                '''                
    shadow_program = compile_program(vert_code, frag_code)

    glUseProgram(shadow_program)
    uniform_locs["MVP"] = glGetUniformLocation(shadow_program, "MVP")

    vert_code = b'''
#version 150
uniform mat4 MVP1;
in vec3 vPos, vCol;
out vec3 fCol;
void main() {
    gl_Position = MVP1 * vec4(vPos, 1);
    fCol = vCol;
}
                '''
    frag_code = b''' 
#version 150
in vec3 fCol;
out vec4 gl_FragColor;
void main() {
    gl_FragColor = vec4(fCol, 1);
}
'''
    test_program = compile_program(vert_code, frag_code)
    
    glUseProgram(test_program)
    uniform_locs["MVP1"] = glGetUniformLocation(test_program, "MVP1")

    init_gl()
    init_uniform_value()
    
def init_gl():
    global shadow_tex_id1, shadow_tex_id2, shadow_map_size, shadow_FBO1, shadow_FBO2
    glClearColor(0, 0, 0, 0)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    shadow_map_size = 1024*2
    shadow_FBO1 = glGenFramebuffers(1) # FBO setup
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_FBO1)
    shadow_tex_id1 = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, shadow_tex_id1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_map_size, shadow_map_size, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1.0, 1.0, 1.0, 1.0])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow_tex_id1, 0) # mipmap level 0
    glDrawBuffer(GL_NONE)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    shadow_FBO2 = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_FBO2)
    shadow_tex_id2 = glGenTextures(1)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, shadow_tex_id2)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_map_size, shadow_map_size, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1.0, 1.0, 1.0, 1.0])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow_tex_id2, 0) # mipmap level 0
    glDrawBuffer(GL_NONE)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)

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
        colors[cnt] = vals[3:6]
        normals[cnt] = vals[6:9]
        uvs[cnt] = vals[9:11]
        cnt += 1
    n_vertices = cnt
    print("Loaded %d vertices" % n_vertices)

def init_uniform_value():
    eye["pos"] = [0, 10, 10]
    eye["at"] = [0, 0, 0]
    uniform_mat_value["model_mat"] = Identity()
    uniform_mat_value["view_mat"] = LookAt(*eye["pos"], *eye["at"], 0, 1, 0)
    uniform_mat_value["proj_mat"] = Perspective(60, screenWidth/screenHeight, 0.1, 100)
    uniform3_value["l1_pos"] = [0, 2, 10]
    uniform3_value["l2_pos"] = [-3, 2, 10]
    uniform3_value["l_dcol"] = [0.5, 0.5, 0.5]
    uniform3_value["l_scol"] = [1, 1, 1]
    uniform3_value["m_kd"] = [1, 1, 1]
    uniform3_value["m_ks"] = [1, 1, 1]
    uniform_value["m_shininess"] = 50
    
def display():
    B = matrix(((0.5, 0, 0, 0.5), (0, 0.5, 0, 0.5), (0, 0, 0.5, 0.5), (0, 0, 0, 1)), dtype=float32)
    li_pos = uniform3_value["l1_pos"]
    li_at = eye["at"]
    light_proj_mat = Perspective(90, screenWidth/screenHeight, 0.1, 100)
    light_view_mat = LookAt(*li_pos, *li_at, 0, 1, 0)
    model_mat = uniform_mat_value["model_mat"]
    shadow_mat1 = B * light_proj_mat * light_view_mat * model_mat

    glUseProgram(shadow_program)
    glBindTexture(GL_TEXTURE_2D, 0)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_FBO1)
    glViewport(0, 0, shadow_map_size, shadow_map_size)
    glDrawBuffer(GL_NONE)
    glClear(GL_DEPTH_BUFFER_BIT)
    MVP = light_proj_mat * light_view_mat * model_mat
    glUniformMatrix4fv(uniform_locs["MVP"], 1, True, MVP.A)

    glBindVertexArray(vao[0])
    glDrawArrays(GL_TRIANGLES, 0, n_vertices)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    li_pos = uniform3_value["l2_pos"]
    light_view_mat = LookAt(*li_pos, *li_at, 0, 1, 0)
    shadow_mat2 = B * light_proj_mat * light_view_mat * model_mat

    glUseProgram(shadow_program)
    glBindTexture(GL_TEXTURE_2D, 0)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_FBO2)
    glViewport(0, 0, shadow_map_size, shadow_map_size)
    glDrawBuffer(GL_NONE)
    glClear(GL_DEPTH_BUFFER_BIT)
    MVP = light_proj_mat * light_view_mat * model_mat
    glUniformMatrix4fv(uniform_locs["MVP"], 1, True, MVP.A)


    glBindVertexArray(vao[0])
    glDrawArrays(GL_TRIANGLES, 0, n_vertices)


    glUseProgram(render_program)
    for name in uniform_mat:
        glUniformMatrix4fv(uniform_locs[name], 1, True, uniform_mat_value[name].A)
        
    for name in uniform3:
        glUniform3f(uniform_locs[name], *uniform3_value[name])
        
    for name in uniform:
        glUniform1f(uniform_locs[name], uniform_value[name])

    glUniform1i(uniform_locs["isLight1Open"], isLight1Open)
    glUniform1i(uniform_locs["isLight2Open"], isLight2Open)
    
    glUniformMatrix4fv(uniform_locs["shadow_mat1"], 1, True, shadow_mat1.A)
    glUniformMatrix4fv(uniform_locs["shadow_mat2"], 1, True, shadow_mat2.A)
    glUniform1i(uniform_locs["shadow_map1"], 0)

    glBindTexture(GL_TEXTURE_2D, shadow_tex_id2)
    glUniform1i(uniform_locs["shadow_map2"], 1)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDrawBuffer(GL_BACK)
    glViewport(0, 0, screenWidth, screenHeight)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBindTexture(GL_TEXTURE_2D, shadow_tex_id1)
    glBindVertexArray(vao[0])
    glDrawArrays(GL_TRIANGLES, 0, n_vertices)


    #test light view
    # glBindFramebuffer(GL_FRAMEBUFFER, 0)
    # glDrawBuffer(GL_BACK)
    # glViewport(0, 0, screenWidth, screenHeight)
    # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # glUseProgram(test_program)
    # MVP = light_proj_mat * light_view_mat * model_mat
    # glUniformMatrix4fv(uniform_locs["MVP1"], 1, True, MVP.A)
    # glBindVertexArray(vao[0])
    # glDrawArrays(GL_TRIANGLES, 0, n_vertices)

    glutSwapBuffers()

def keyboard(key, x, y):
    global isLight1Open, isLight2Open

    key = key.decode("utf-8")
    if key == '1':
        isLight1Open = 1 - isLight1Open
        glutPostRedisplay()
    elif key == '2':
        isLight2Open = 1 - isLight2Open
        glutPostRedisplay()

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(screenWidth, screenHeight)
    glutInitWindowPosition(50, 50)
    glutCreateWindow(b"GLSL Example")
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    # load_tri(sys.argv[1])
    load_tri("models\objects_and_walls.tri")
    init()
    glutMainLoop()

if __name__ == "__main__":
    main()    