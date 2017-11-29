import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image
from numpy import *
from gl_utils import *
from math import *
from ctypes import c_void_p

n_vertices, positions, normals, uvs, colors, vao, vbo = {}, {}, {}, {}, {}, {}, {}
screenWidth, screenHeight = 800, 600
shadow_map_size = 0
model_name = []
light_attrib = ["l1_pos", "l2_pos", "l_dcol", "l_scol", "m_kd", "m_ks"]
glsl_program = {}
uniform_locs = {}
value = {}
tex_id = {}
shadow_FBO = {}
isLight1Open = 0
isLight2Open = 0
isToon = True

def keyboard(key, x, y):
    global isLight1Open, isLight2Open, isToon

    key = key.decode("utf-8")
    if key == '1':
        isLight1Open = 1 - isLight1Open
        glutPostRedisplay()
    elif key == '2':
        isLight2Open = 1 - isLight2Open
        glutPostRedisplay()
    if key == 't':
        isToon = not isToon
        glutPostRedisplay()
    elif ord(key) == 27:
        exit(0)

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
    print("Loaded %d vertices" % n_vertices[name])
    vao[name] = glGenVertexArrays(1)
    vbo[name] = glGenBuffers(4)

def load_cube_map(filename):
    im = Image.open(filename)
    w, h = im.size[0], im.size[1]
    sub_w = int(w/4)
    sub_h = int(h/3)
    box = [[2,1], [0,1], [1,0], [1,2], [1,1], [3,1]]
    cube_side_id = [GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z]
    glActiveTexture(GL_TEXTURE0)
    tex_id["cube"] = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex_id["cube"])
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    for i in range(6):
        image = im.crop((box[i][0]* sub_w, box[i][1]*sub_h, (box[i][0]+1)* sub_w, (box[i][1]+1) * sub_h)).tobytes("raw", "RGB", 0)
        gluBuild2DMipmaps(cube_side_id[i], GL_RGB, sub_w, sub_h, GL_RGB, GL_UNSIGNED_BYTE, image)
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
    
def init_glsl():
    vert_code = '''
    #version 330
    uniform mat4 proj_mat, view_mat, model_mat;
    in vec3 vPos;
    out vec3 fPos;
    void main() {
        gl_Position = proj_mat * view_mat * model_mat * vec4(vPos, 1);
        fPos = vPos;
    }
    '''
    frag_code = '''
    #version 330
    uniform samplerCube cube_map;
    in vec3 fPos;
    out vec4 gl_FragColor;
    void main() {
        vec4 tex_color = texture(cube_map, fPos);
        gl_FragColor = tex_color;
    }
    '''

    glsl_program["cube"] = compile_program(vert_code, frag_code)
    glUseProgram(glsl_program["cube"])
    for name in ["model_mat", "view_mat", "proj_mat", "cube_map"]:
        uniform_locs["cube_" + name] = glGetUniformLocation(glsl_program["cube"], name)

    glBindVertexArray(vao["room"])

    glBindBuffer(GL_ARRAY_BUFFER, vbo["room"][0])
    glBufferData(GL_ARRAY_BUFFER, positions["room"], GL_DYNAMIC_DRAW)
    location = glGetAttribLocation(glsl_program["cube"], "vPos")
    glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(location)

    vert_code = b'''
#version 330
uniform mat4 model_mat, view_mat, proj_mat;
uniform vec3 ePos;
in vec3 vPos, vNor;
out vec3 reflectDir, refractDir;
void main() {

    gl_Position = proj_mat * view_mat * model_mat * vec4(vPos, 1);
    mat4 adjunct_mat = transpose(inverse(model_mat));
    vec3 P = (model_mat * vec4(vPos, 1)).xyz;
    vec3 N = normalize((adjunct_mat * vec4(vNor, 0)).xyz);
    vec3 I = normalize(P - ePos);
    reflectDir = reflect(I, N);
    //refractDir = refract(I, N, 0.33);
}
                '''
    frag_code = b''' 
#version 130
uniform samplerCube cube_map;
in vec3 reflectDir, refractDir;
out vec4 gl_FragColor;
void main() {
    vec4 reflect_color = texture(cube_map, reflectDir);
    //vec4 refract_color = texture(cube_map, refractDir);
    //gl_FragColor = mix(reflect_color, refract_color, 0.8);
    gl_FragColor = reflect_color;
}
                '''
    glsl_program["reflection"] = compile_program(vert_code, frag_code)
    glUseProgram(glsl_program["reflection"])
    for name in ["model_mat", "view_mat", "proj_mat","cube_map", "ePos"]:
        uniform_locs["reflection_" + name] = glGetUniformLocation(glsl_program["reflection"], name)

    glBindVertexArray(vao["ball"])
    glBindBuffer(GL_ARRAY_BUFFER, vbo["ball"][0])
    glBufferData(GL_ARRAY_BUFFER, positions["ball"], GL_DYNAMIC_DRAW)
    location = glGetAttribLocation(glsl_program["reflection"], "vPos")
    glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(location)
    glBindBuffer(GL_ARRAY_BUFFER, vbo["ball"][1])
    glBufferData(GL_ARRAY_BUFFER, normals["ball"], GL_STATIC_DRAW)
    location = glGetAttribLocation(glsl_program["reflection"], "vNor")
    if location != -1:
        glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(location)

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
    glsl_program["shadow"] = compile_program(vert_code, frag_code)

    glUseProgram(glsl_program["shadow"])
    uniform_locs["shadow_MVP"] = glGetUniformLocation(glsl_program["shadow"], "MVP")

    vert_code = b'''
#version 330
uniform mat4 model_mat, view_mat, proj_mat, shadow_mat1, shadow_mat2;
in vec3 vPos, vNor;
out vec3 fPos, fNor, fCol, e_pos;
out vec4 fSha1, fSha2;
void main() {
    fPos = (model_mat * vec4(vPos, 1)).xyz;
    gl_Position = proj_mat * view_mat * model_mat * vec4(vPos, 1);
    mat4 adjunct_mat = transpose(inverse(model_mat));
    fNor = (adjunct_mat * vec4(vNor, 0)).xyz;
    fCol = (vNor + 1) * 0.5;
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
uniform int isLight1Open, isLight2Open, has_cube_map;
uniform sampler2D shadow_map1, shadow_map2;
uniform samplerCube cube_map;
in vec3 e_pos, fPos, fCol, fNor;
in vec4 fSha1, fSha2;
out vec4 gl_FragColor;
vec3 model_color;
vec3 calculateLightFrom(vec3 pos, vec4 fSha, sampler2D shadow_map) {
    vec3 N = normalize(fNor);
    vec3 L = normalize(pos - fPos);
    float LdotN = max(0, dot(L, N));
    vec3 diffuse = model_color * l_dcol * LdotN;
    vec3 V = normalize(e_pos - fPos);
    vec3 H = normalize(L + V);
    float HdotN = max(0, dot(H, N));
    vec3 specular = m_ks * l_scol * pow(HdotN, m_shininess);
    float visibility = 1.0;
    float bias = 0.0005;
    int i, j;
    for(i=-1; i<=1; i++)
        for(j=-1; j<=1; j++)
        if (textureOffset(shadow_map, fSha.st/fSha.w, ivec2(i, j)).r < fSha.z/fSha.w - bias)
            visibility -= 0.1;
    return (diffuse + specular) * visibility;
}
void main() {
    if (has_cube_map == 1) {
        model_color = texture(cube_map, fPos).rgb;
    } else {
        model_color = fCol;
    }
    vec3 color_from_light1 = calculateLightFrom(l1_pos, fSha1, shadow_map1);
    vec3 color_from_light2 = calculateLightFrom(l2_pos, fSha2, shadow_map2);

    vec3 color = (color_from_light1 * isLight1Open + color_from_light2 * isLight2Open);

    gl_FragColor = vec4(color + ambient_color, 1);
    gl_FragColor = clamp(gl_FragColor, 0, 1);
}
'''
    glsl_program["render_shadow"] = compile_program(vert_code, frag_code)
    glUseProgram(glsl_program["render_shadow"])
    for name in ["model_mat", "view_mat", "proj_mat", "l1_pos", "l2_pos", "l_dcol", "l_scol", "m_kd", "m_ks", "m_shininess", "shadow_mat1", "shadow_mat2", "shadow_map1", "shadow_map2", "isLight1Open", "isLight2Open", "cube_map", "has_cube_map"]:
        uniform_locs["render_shadow_" + name] = glGetUniformLocation(glsl_program["render_shadow"], name)

    for model in ["ball", "room"]:
        glBindVertexArray(vao[model])

        glBindBuffer(GL_ARRAY_BUFFER, vbo[model][0])
        glBufferData(GL_ARRAY_BUFFER, positions[model], GL_DYNAMIC_DRAW)
        location = glGetAttribLocation(glsl_program["render_shadow"], "vPos")
        glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(location)

        glBindBuffer(GL_ARRAY_BUFFER, vbo[model][1])
        glBufferData(GL_ARRAY_BUFFER, normals[model], GL_STATIC_DRAW)
        location = glGetAttribLocation(glsl_program["render_shadow"], "vNor")
        if location != -1:
            glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
            glEnableVertexAttribArray(location)

        vert_code = b'''
#version 330
uniform mat4 model_mat, view_mat, proj_mat;
in vec3 vPos, vNor;
out vec3 e_pos, fPos, fNor, fCol;
void main() {
    float distance = abs((view_mat * model_mat * vec4(vPos, 1)).z);
    fPos = (model_mat * vec4(vPos, 1)).xyz;
    gl_Position = proj_mat * view_mat * model_mat * vec4(vPos, 1);
    mat4 adjunct_mat = transpose(inverse(model_mat));
    fNor = (adjunct_mat * vec4(vNor, 0)).xyz;
    fCol = (vNor + 1) * 0.5;
    e_pos = (inverse(view_mat) * vec4(0,0,0,1)).xyz;
}
                '''
    frag_code = b''' 
#version 130
vec3 ambient_color = vec3(0.15, 0.15, 0.15);
uniform vec3 l1_pos, l2_pos, l_dcol, l_scol, m_kd, m_ks;
uniform float m_shininess;
uniform int isLight1Open, isLight2Open;
in vec3 e_pos, fPos, fCol, fNor;
out vec4 gl_FragColor;
vec3 diff_color1, diff_color2, spec_color1, spec_color2;
void main() {
    vec3 N = normalize(fNor);
    vec3 L1 = normalize(l1_pos - fPos);
    vec3 L2 = normalize(l2_pos - fPos);
    float diffuse1 = max(0, dot(L1, N));
    float diffuse2 = max(0, dot(L2, N));
    vec3 V = normalize(e_pos - fPos);
    vec3 H1 = normalize(L1 + V);
    vec3 H2 = normalize(L2 + V);
    float HdotN1 = max(0, dot(H1, N));
    float HdotN2 = max(0, dot(H2, N));
    float specular1 = pow(HdotN1, m_shininess);
    float specular2 = pow(HdotN2, m_shininess);
    if (diffuse1 == 0)
        specular1 = 0.0;
    if (diffuse2 == 0)
        specular2 = 0.0;
    float edge = max(dot(N, V), 0);

    vec3 diff_color, spec_color;
    if (diffuse1 > 0.5)
        diff_color1 = fCol;
    else
        diff_color1 = 0.5*fCol;
    if (diffuse2 > 0.5)
        diff_color2 = fCol;
    else
        diff_color2 = 0.5*fCol;
    
    if (specular1 > 0.3)
        spec_color1 = m_ks;
    else
        spec_color1 = vec3(0,0,0);
    if (specular2 > 0.3)
        spec_color2 = m_ks;
    else
        spec_color2 = vec3(0,0,0);

    if (edge < 0.3)
        edge = 0;
    else
        edge = 1;

    vec3 color1 = edge * (diff_color1 + spec_color1) * fCol;
    vec3 color2 = edge * (diff_color2 + spec_color2) * fCol;
    vec3 color = (color1 * isLight1Open + color2 * isLight2Open) / (isLight1Open + isLight2Open) + ambient_color;
    gl_FragColor = vec4(clamp(color, 0, 1),1);    
}
'''
    glsl_program["toon"] = compile_program(vert_code, frag_code)
    glUseProgram(glsl_program["toon"])
    for name in ["model_mat", "view_mat", "proj_mat", "l1_pos", "l2_pos", "l_dcol", "l_scol", "m_kd", "m_ks", "m_shininess", "isLight1Open", "isLight2Open"]:
        uniform_locs["toon_" + name] = glGetUniformLocation(glsl_program["toon"], name)
    
    glBindVertexArray(vao["ball"])

    glBindBuffer(GL_ARRAY_BUFFER, vbo["ball"][0])
    glBufferData(GL_ARRAY_BUFFER, positions["ball"], GL_DYNAMIC_DRAW)
    location = glGetAttribLocation(glsl_program["toon"], "vPos")
    glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(location)

    glBindBuffer(GL_ARRAY_BUFFER, vbo["ball"][1])
    glBufferData(GL_ARRAY_BUFFER, normals["ball"], GL_STATIC_DRAW)
    location = glGetAttribLocation(glsl_program["toon"], "vNor")
    if location != -1:
        glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(location)


    vert_code = b'''
#version 150
uniform mat4 MVP1;
in vec3 vPos, vNor;
out vec3 fCol;
void main() {
    gl_Position = MVP1 * vec4(vPos, 1);
    fCol = (vNor + 1) * 0.5;
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
    glsl_program["test"] = compile_program(vert_code, frag_code)
    
    glUseProgram(glsl_program["test"])
    uniform_locs["MVP1"] = glGetUniformLocation(glsl_program["test"], "MVP1")

def init_texture():
    global shadow_map_size
    glShadeModel(GL_SMOOTH)
    shadow_map_size = 1024
    
    shadow_FBO["light1"] = glGenFramebuffers(1) # FBO setup
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_FBO["light1"])
    tex_id["light1"] = glGenTextures(1)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, tex_id["light1"])
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_map_size, shadow_map_size, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1.0, 1.0, 1.0, 1.0])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex_id["light1"], 0) # mipmap level 0
    glDrawBuffer(GL_NONE)
    shadow_FBO["light2"] = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_FBO["light2"])
    tex_id["light2"] = glGenTextures(1)
    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_2D, tex_id["light2"])
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_map_size, shadow_map_size, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1.0, 1.0, 1.0, 1.0])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex_id["light2"], 0) # mipmap level 0
    glDrawBuffer(GL_NONE)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)

def init_value():
    value["e_pos"] = [-30, 0, 30]
    value["e_at"] = [10, -20, 10]
    value["model_mat"] = Identity()
    value["view_mat"] = LookAt(*value["e_pos"], *value["e_at"], 0, 1, 0)
    value["proj_mat"] = Perspective(60, screenWidth/screenHeight, 0.1, 300)
    value["l1_pos"] = [-30, 30, -40]
    value["l2_pos"] = [30, 30, 40]
    value["l_dcol"] = [0.5, 0.5, 0.5]
    value["l_scol"] = [0.5, 0.5, 0.5]
    value["m_kd"] = [1, 1, 1]
    value["m_ks"] = [1, 1, 1]
    value["m_shininess"] = 50

def display():
    B = matrix(((0.5, 0, 0, 0.5), (0, 0.5, 0, 0.5), (0, 0, 0.5, 0.5), (0, 0, 0, 1)), dtype=float32)

    #create texture for shadow_map1
    li_pos = value["l1_pos"]
    li_at = value["e_at"]
    light_proj_mat = Perspective(90, screenWidth/screenHeight, 0.1, 300)
    light_view_mat = LookAt(*li_pos, *li_at, 0, 1, 0)
    model_mat = value["model_mat"]
    shadow_mat1 = B * light_proj_mat * light_view_mat * model_mat
    model_mat = model_mat * Scale(50, 50, 50)

    glUseProgram(glsl_program["shadow"])
    glBindTexture(GL_TEXTURE_2D, 0)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_FBO["light1"])
    glViewport(0, 0, shadow_map_size, shadow_map_size)
    glDrawBuffer(GL_NONE)
    glClear(GL_DEPTH_BUFFER_BIT)
    MVP = light_proj_mat * light_view_mat * model_mat
    glUniformMatrix4fv(uniform_locs["shadow_MVP"], 1, True, MVP.A)
    glBindVertexArray(vao["room"])
    glDrawArrays(GL_QUADS, 0, n_vertices["room"])
    model_mat = value["model_mat"] * Translate(10, -20, 10) * Scale(10, 10, 10)
    MVP = light_proj_mat * light_view_mat * model_mat
    glUniformMatrix4fv(uniform_locs["shadow_MVP"], 1, True, MVP.A)
    glBindVertexArray(vao["ball"])
    glDrawArrays(GL_TRIANGLES, 0, n_vertices["ball"])

    #create texture for shadow_map2
    li_pos = value["l2_pos"]
    li_at = value["e_at"]
    light_proj_mat = Perspective(90, screenWidth/screenHeight, 0.1, 300)
    light_view_mat = LookAt(*li_pos, *li_at, 0, 1, 0)
    model_mat = value["model_mat"]
    shadow_mat2 = B * light_proj_mat * light_view_mat * model_mat
    model_mat = model_mat * Scale(50, 50, 50)

    glUseProgram(glsl_program["shadow"])
    glBindTexture(GL_TEXTURE_2D, 0)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_FBO["light2"])
    glViewport(0, 0, shadow_map_size, shadow_map_size)
    glDrawBuffer(GL_NONE)
    glClear(GL_DEPTH_BUFFER_BIT)
    MVP = light_proj_mat * light_view_mat * model_mat
    glUniformMatrix4fv(uniform_locs["shadow_MVP"], 1, True, MVP.A)
    glBindVertexArray(vao["room"])
    glDrawArrays(GL_QUADS, 0, n_vertices["room"])
    model_mat = value["model_mat"] * Translate(10, -20, 10) * Scale(10, 10, 10)
    MVP = light_proj_mat * light_view_mat * model_mat
    glUniformMatrix4fv(uniform_locs["shadow_MVP"], 1, True, MVP.A)
    glBindVertexArray(vao["ball"])
    glDrawArrays(GL_TRIANGLES, 0, n_vertices["ball"])

    #draw objects (room)
    view_mat = value["view_mat"]

    glUseProgram(glsl_program["render_shadow"])
    model_mat = value["model_mat"] * Scale(50, 50, 50)
    glUniformMatrix4fv(uniform_locs["render_shadow_proj_mat"], 1, True, value["proj_mat"].A)
    glUniformMatrix4fv(uniform_locs["render_shadow_view_mat"], 1, True, view_mat.A)
    glUniformMatrix4fv(uniform_locs["render_shadow_model_mat"], 1, True, model_mat.A)
    glUniform1i(uniform_locs["render_shadow_cube_map"], 0)
    glUniform1i(uniform_locs["render_shadow_has_cube_map"], 1)
    for attrib in light_attrib:
        glUniform3f(uniform_locs["render_shadow_" + attrib], *value[attrib])
    glUniform1f(uniform_locs["render_shadow_m_shininess"], value["m_shininess"])
    glUniform1i(uniform_locs["render_shadow_isLight1Open"], isLight1Open)
    glUniform1i(uniform_locs["render_shadow_isLight2Open"], isLight2Open)
    glUniformMatrix4fv(uniform_locs["render_shadow_shadow_mat1"], 1, True, (shadow_mat1 * Scale(50, 50, 50)).A)
    glUniformMatrix4fv(uniform_locs["render_shadow_shadow_mat2"], 1, True, (shadow_mat2 * Scale(50, 50, 50)).A)
    glBindTexture(GL_TEXTURE_2D, tex_id["light1"])
    glUniform1i(uniform_locs["render_shadow_shadow_map1"], 1)
    glBindTexture(GL_TEXTURE_2D, tex_id["light2"])
    glUniform1i(uniform_locs["render_shadow_shadow_map2"], 2)


    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDrawBuffer(GL_BACK)
    glViewport(0, 0, screenWidth, screenHeight)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glBindVertexArray(vao["room"])
    glDrawArrays(GL_QUADS, 0, n_vertices["room"])

    #draw mirror ball
    
    if isToon:
        prog_id = "reflection"
        glUseProgram(glsl_program[prog_id])
        model_mat = value["model_mat"] * Translate(10, -20, 10) * Scale(10, 10, 10)
        glUniformMatrix4fv(uniform_locs["reflection_proj_mat"], 1, True, value["proj_mat"].A)
        glUniformMatrix4fv(uniform_locs["reflection_view_mat"], 1, True, view_mat.A)
        glUniformMatrix4fv(uniform_locs["reflection_model_mat"], 1, True, model_mat.A)
        glUniform1i(uniform_locs["reflection_cube_map"], 0)
        glUniform3f(uniform_locs["reflection_ePos"], *value["e_pos"])
    else:
        prog_id = "toon"
        glUseProgram(glsl_program[prog_id])
        model_mat = value["model_mat"] * Translate(10, -20, 10) * Scale(10, 10, 10)
        glUniformMatrix4fv(uniform_locs["toon_proj_mat"], 1, True, value["proj_mat"].A)
        glUniformMatrix4fv(uniform_locs["toon_view_mat"], 1, True, view_mat.A)
        glUniformMatrix4fv(uniform_locs["toon_model_mat"], 1, True, model_mat.A)
        for attrib in light_attrib:
            glUniform3f(uniform_locs["toon_" + attrib], *value[attrib])
        glUniform1i(uniform_locs["toon_isLight1Open"], isLight1Open)
        glUniform1i(uniform_locs["toon_isLight2Open"], isLight2Open)
        

    glBindVertexArray(vao["ball"])
    glDrawArrays(GL_TRIANGLES, 0, n_vertices["ball"])


    # test light view

    # li_pos = value["l1_pos"]
    # li_at = value["e_at"]
    # light_proj_mat = Perspective(90, screenWidth/screenHeight, 0.1, 200)
    # light_view_mat = LookAt(*li_pos, *li_at, 0, 1, 0)
    # model_mat = value["model_mat"] * Scale(50, 50, 50)
    # glBindFramebuffer(GL_FRAMEBUFFER, 0)
    # glDrawBuffer(GL_BACK)
    # glViewport(0, 0, shadow_map_size, shadow_map_size)
    # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # glUseProgram(glsl_program["test"])
    # MVP = light_proj_mat * light_view_mat * model_mat
    # glUniformMatrix4fv(uniform_locs["MVP1"], 1, True, MVP.A)
    # glBindVertexArray(vao["room"])
    # glDrawArrays(GL_QUADS, 0, n_vertices["room"])
    # model_mat = value["model_mat"] * Translate(10, -20, 10) * Scale(10, 10, 10)
    # MVP = light_proj_mat * light_view_mat * model_mat
    # glUniformMatrix4fv(uniform_locs["MVP1"], 1, True, MVP.A)
    # glBindVertexArray(vao["ball"])
    # glDrawArrays(GL_TRIANGLES, 0, n_vertices["ball"])
    

    glutSwapBuffers()

def init():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_CUBE_MAP)
    glEnable(GL_TEXTURE_2D)
    load_tri("models/room2.tri", "room")
    load_tri("models/ball.tri", "ball")
    load_cube_map("cube_map/street_HCmap.jpg")
    init_texture()
    init_glsl()
    init_value()

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(screenWidth, screenHeight)
    glutInitWindowPosition(50, 50)    
    glutCreateWindow(b"Reflection map")
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    init()
    glutMainLoop()

if __name__ == "__main__":
    main()   