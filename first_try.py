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
{}, {}, {}, {}, {}, {}, {}, {}
screenWidth, screenHeight = 800, 600
shadow_tex_id1, shadow_tex_id2, shadow_map_size, shadow_FBO1, shadow_FBO2 = 0, 0, 0, 0, 0
render_program, shadow_program = 0, 0
uniform_locs = {}
uniform_mat_value = {}
uniform3_value = {}
uniform_value = {}
uniform_mat = ["model_mat_render", "view_mat_render", "proj_mat_render"]
uniform3 = ["l1_pos_render", "l2_pos_render", "l_dcol_render", "l_scol_render", "m_kd_render", "m_ks_render"]
uniform = ["m_shininess_render"]
eye = {}
model_name = []
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

def load_tri(filename, name):
    global n_vertices, positions, normals, uvs, colors, vao, vbo
    try:
        f = open(filename, 'r')
    except:
        print("%s not found!" % filename)
        exit(1)
    lines = f.read().split('\n')
    f.close()
    n_vertices[name] = len(lines)
    positions[name] = zeros((n_vertices[name], 3), dtype=float32)
    colors[name] = zeros((n_vertices[name], 3), dtype=float32)
    normals[name] = zeros((n_vertices[name], 3), dtype=float32)
    uvs[name] = zeros((n_vertices[name], 2), dtype=float32)
    cnt = 0
    for line in lines:
        if len(line) == 0 or line[0] == '#': continue
        vals = line.split()
        positions[name][cnt] = vals[0:3]
        normals[name][cnt] = vals[3:6]
        if len(vals) > 6:
            uvs[name][cnt] = vals[6:8]
        cnt +=1 
    n_vertices[name] = cnt
    model_name.append(name)
    # normals[:] = (normals+1)*0.5
    print("Loaded %d vertices" % n_vertices[name])
    vao[name] = glGenVertexArrays(2)
    vbo[name] = glGenBuffers(6)

   


def init():
    global  shadow_program, render_program, vao, vbo, prog_id
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
    
    for name in model_name:
        glBindVertexArray(vao[name][0])

        glBindBuffer(GL_ARRAY_BUFFER, vbo[name][0])
        glBufferData(GL_ARRAY_BUFFER, positions[name], GL_DYNAMIC_DRAW)
        location = glGetAttribLocation(prog_id, "vPos")
        glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(location)


        glBindBuffer(GL_ARRAY_BUFFER, vbo[name][1])
        glBufferData(GL_ARRAY_BUFFER, normals[name], GL_STATIC_DRAW)
        location = glGetAttribLocation(prog_id, "vNor")
        if location != -1:
            glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
            glEnableVertexAttribArray(location)   

    glBindVertexArray(0)


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
    if (vCol == vec3(0, 0, 0)) {
        fCol = vec3(1, 0, 0);
    }
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


    # vao = glGenVertexArrays(2)
    for name in model_name:
        glBindVertexArray(vao[name][1])
        # vbo = glGenBuffers(4)

        glBindBuffer(GL_ARRAY_BUFFER, vbo[name][2])
        glBufferData(GL_ARRAY_BUFFER, positions[name], GL_DYNAMIC_DRAW)
        location = glGetAttribLocation(render_program, "vPos")
        glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(location)

        glBindBuffer(GL_ARRAY_BUFFER, vbo[name][3])
        glBufferData(GL_ARRAY_BUFFER, normals[name], GL_STATIC_DRAW)
        location = glGetAttribLocation(render_program, "vNor")
        if location != -1:
            glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
            glEnableVertexAttribArray(location)  

        glBindBuffer(GL_ARRAY_BUFFER, vbo[name][4])
        glBufferData(GL_ARRAY_BUFFER, colors[name], GL_STATIC_DRAW)
        location = glGetAttribLocation(render_program, "vCol")
        if location != -1:
            glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
            glEnableVertexAttribArray(location)  

        glBindBuffer(GL_ARRAY_BUFFER, vbo[name][5])
        glBufferData(GL_ARRAY_BUFFER, uvs[name], GL_STATIC_DRAW)
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

    init_gl()
    init_value()

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
    glActiveTexture(GL_TEXTURE1)
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
    glActiveTexture(GL_TEXTURE2)
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

def init_value():
    eye["pos"] = [-30,-30,-50]
    eye["at"] = [0,-40,0]
    uniform_mat_value["model_mat_render"] = Identity()
    uniform_mat_value["view_mat_render"] = LookAt(*eye["pos"], *eye["at"], 0, 1, 0)
    uniform_mat_value["proj_mat_render"] = Perspective(60, screenWidth/screenHeight, 0.1, 100)
    uniform3_value["l1_pos_render"] = [0, 2, 10]
    uniform3_value["l2_pos_render"] = [-30, 2, 10]
    uniform3_value["l_dcol_render"] = [0.5, 0.5, 0.5]
    uniform3_value["l_scol_render"] = [1, 1, 1]
    uniform3_value["m_kd_render"] = [1, 1, 1]
    uniform3_value["m_ks_render"] = [1, 1, 1]
    uniform_value["m_shininess_render"] = 50

def display():
    vertices = [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1], [1,-1,-1],[1,-1,1],[1,1,1],[1,1,-1], 
    [-1,-1,-1],[-1,-1,1],[1,-1,1],[1,-1,-1], [-1,1,-1],[-1,1,1],[1,1,1],[1,1,-1],
    [-1,1,1],[-1,-1,1],[1,-1,1],[1,1,1], [-1,1,-1],[-1,-1,-1],[1,-1,-1],[1,1,-1]] 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # e_pos = (20*cos(angle*pi/180), 0, 20*sin(angle*pi/180))
    # e_at = (-1,4,0)
    e_pos = eye["pos"]
    e_at = eye["at"]
    # glUseProgram(0)
    # glMatrixMode(GL_PROJECTION)
    # glLoadIdentity()
    # gluPerspective(40., screenWidth/screenHeight, 0.01, 500.0)
    # glMatrixMode(GL_MODELVIEW)
    # glLoadIdentity()
    # gluLookAt(*e_pos, *e_at, 0, 1, 0)
    # glScalef(50,50,50)

    # glRotatef(angle,0,1,0)
    # glBegin(GL_QUADS)
    # for vertex in vertices:
    #     texCoord = [-vertex[0]] + [vertex[1]] + [-vertex[2]]
    #     glTexCoord3fv(texCoord)
    #     glVertex3fv(vertex)

    # glEnd()

    ######


    B = matrix(((0.5, 0, 0, 0.5), (0, 0.5, 0, 0.5), (0, 0, 0.5, 0.5), (0, 0, 0, 1)), dtype=float32)
    li_pos = uniform3_value["l1_pos_render"]
    li_at = eye["at"]
    light_proj_mat = Perspective(90, screenWidth/screenHeight, 0.1, 100)
    light_view_mat = LookAt(*li_pos, *li_at, 0, 1, 0)
    model_mat = uniform_mat_value["model_mat_render"]
    shadow_mat1 = B * light_proj_mat * light_view_mat * model_mat

    glUseProgram(shadow_program)
    glBindTexture(GL_TEXTURE_2D, 0)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_FBO1)
    glViewport(0, 0, shadow_map_size, shadow_map_size)
    glDrawBuffer(GL_NONE)
    glClear(GL_DEPTH_BUFFER_BIT)
    
    model_mat = model_mat * Translate(0, -30, 0) * Scale(10, 10, 10)
    MVP = light_proj_mat * light_view_mat * model_mat
    glUniformMatrix4fv(uniform_locs["MVP"], 1, True, MVP.A)
    glBindVertexArray(vao["ball"][1])
    glDrawArrays(GL_TRIANGLES, 0, n_vertices["ball"])
    model_mat = model_mat * Scale(0.1, 0.1, 0.1) * Translate(0, 30, 0) * Scale(50, 50, 50)
    MVP = light_proj_mat * light_view_mat * model_mat
    glUniformMatrix4fv(uniform_locs["MVP"], 1, True, MVP.A)
    glBindVertexArray(vao["room"][1])
    glDrawArrays(GL_TRIANGLES, 0, n_vertices["room"])
        

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    li_pos = uniform3_value["l2_pos_render"]
    light_view_mat = LookAt(*li_pos, *li_at, 0, 1, 0)
    model_mat = uniform_mat_value["model_mat_render"]
    shadow_mat2 = B * light_proj_mat * light_view_mat * model_mat

    glUseProgram(shadow_program)
    glBindTexture(GL_TEXTURE_2D, 0)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_FBO2)
    glViewport(0, 0, shadow_map_size, shadow_map_size)
    glDrawBuffer(GL_NONE)
    glClear(GL_DEPTH_BUFFER_BIT)
    MVP = light_proj_mat * light_view_mat * model_mat
    glUniformMatrix4fv(uniform_locs["MVP"], 1, True, MVP.A)


    model_mat = model_mat * Translate(0, -30, 0) * Scale(10, 10, 10)
    MVP = light_proj_mat * light_view_mat * model_mat
    glUniformMatrix4fv(uniform_locs["MVP"], 1, True, MVP.A)
    glBindVertexArray(vao["ball"][1])
    glDrawArrays(GL_TRIANGLES, 0, n_vertices["ball"])
    model_mat = model_mat * Scale(0.1, 0.1, 0.1) * Translate(0, 30, 0) * Scale(50, 50, 50)
    MVP = light_proj_mat * light_view_mat * model_mat
    glUniformMatrix4fv(uniform_locs["MVP"], 1, True, MVP.A)
    glBindVertexArray(vao["room"][1])
    glDrawArrays(GL_TRIANGLES, 0, n_vertices["room"])


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
    glBindTexture(GL_TEXTURE_2D, shadow_tex_id1)
    glUniform1i(uniform_locs["shadow_map1"], 1)

    glBindTexture(GL_TEXTURE_2D, shadow_tex_id2)
    glUniform1i(uniform_locs["shadow_map2"], 2)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDrawBuffer(GL_BACK)
    glViewport(0, 0, screenWidth, screenHeight)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    model_mat = uniform_mat_value["model_mat_render"]
    model_mat = model_mat * Translate(0, -30, 0) * Scale(10, 10, 10)
    glUniformMatrix4fv(uniform_locs["model_mat_render"], 1, True, model_mat.A)
    glBindVertexArray(vao["ball"][1])
    glDrawArrays(GL_TRIANGLES, 0, n_vertices["ball"])
    model_mat = model_mat * Scale(0.1, 0.1, 0.1) * Translate(0, 30, 0) * Scale(50, 50, 50)
    glUniformMatrix4fv(uniform_locs["model_mat_render"], 1, True, model_mat.A)
    glBindVertexArray(vao["room"][1])
    glDrawArrays(GL_TRIANGLES, 0, n_vertices["room"])


    #####
    glUseProgram(prog_id) 
    
    
    # proj_mat = Perspective(40., screenWidth/screenHeight, 0.01, 500.0)
    # glUniformMatrix4fv(uniform_locs["proj_mat"], 1, True, proj_mat.A)
    # view_mat = LookAt(*e_pos, *e_at, 0, 1, 0)
    # glUniformMatrix4fv(uniform_locs["view_mat"], 1, True, view_mat.A)
    # model_mat =  Translate(0, -40, 0) * Scale(10, 10, 10)
    # glUniformMatrix4fv(uniform_locs["model_mat"], 1, True, model_mat.A)
    
    model_mat = uniform_mat_value["model_mat_render"]
    model_mat = model_mat * Translate(0, -30, 0) * Scale(10, 10, 10)
    glUniformMatrix4fv(uniform_locs["model_mat"], 1, True, model_mat.A)
    glUniformMatrix4fv(uniform_locs["view_mat"], 1, True, uniform_mat_value["view_mat_render"].A)
    glUniformMatrix4fv(uniform_locs["proj_mat"], 1, True, uniform_mat_value["proj_mat_render"].A)
    glBindVertexArray(vao["ball"][0])
    glDrawArrays(GL_TRIANGLES, 0, n_vertices["ball"])
    ######

    # e_pos = eye["pos"]
    # e_at = eye["at"]
    # glUseProgram(0)
    # glMatrixMode(GL_PROJECTION)
    # glLoadIdentity()
    # gluPerspective(40., screenWidth/screenHeight, 0.01, 500.0)
    # glMatrixMode(GL_MODELVIEW)
    # glLoadIdentity()
    # gluLookAt(*e_pos, *e_at, 0, 1, 0)
    # glScalef(50,50,50)

    # glRotatef(angle,0,1,0)
    # glBegin(GL_QUADS)
    # for vertex in vertices:
    #     texCoord = [-vertex[0]] + [vertex[1]] + [-vertex[2]]
    #     glTexCoord3fv(texCoord)
    #     glVertex3fv(vertex)

    # glEnd()







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
    load_tri("models/ball.tri", "ball")
    load_tri("models/room.tri", "room")
    glutIdleFunc(animate)
    glEnable(GL_DEPTH_TEST)
    init()
    glutMainLoop()

if __name__ == "__main__":
    main()    