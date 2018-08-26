uniform float time;
uniform vec2 resolution;
uniform mat4 inverseWorld;

uniform sampler2D textureEnv;

varying vec2 vUv;
varying mat4 model;

#define STEPS 96
#define STEP_SIZE 1.0 / 96.0
#define EPSILON 0.0001
#define MIN_DIST 0.0
#define MAX_DIST 100.0
#define MAX_MARCHING_STEPS 255


vec3 glow = vec3(0.);
float glow_intensity = .015;
vec3 glow_color = vec3(1.0);



vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
float permute(float x){return floor(mod(((x*34.0)+1.0)*x, 289.0));}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
float taylorInvSqrt(float r){return 1.79284291400159 - 0.85373472095314 * r;}

vec4 grad4(float j, vec4 ip){
  const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
  vec4 p,s;

  p.xyz = floor( fract (vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
  p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
  s = vec4(lessThan(p, vec4(0.0)));
  p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www;

  return p;
}

float snoise(vec4 v){
  const vec2  C = vec2( 0.138196601125010504,  // (5 - sqrt(5))/20  G4
                        0.309016994374947451); // (sqrt(5) - 1)/4   F4
// First corner
  vec4 i  = floor(v + dot(v, C.yyyy) );
  vec4 x0 = v -   i + dot(i, C.xxxx);

// Other corners

// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
  vec4 i0;

  vec3 isX = step( x0.yzw, x0.xxx );
  vec3 isYZ = step( x0.zww, x0.yyz );
//  i0.x = dot( isX, vec3( 1.0 ) );
  i0.x = isX.x + isX.y + isX.z;
  i0.yzw = 1.0 - isX;

//  i0.y += dot( isYZ.xy, vec2( 1.0 ) );
  i0.y += isYZ.x + isYZ.y;
  i0.zw += 1.0 - isYZ.xy;

  i0.z += isYZ.z;
  i0.w += 1.0 - isYZ.z;

  // i0 now contains the unique values 0,1,2,3 in each channel
  vec4 i3 = clamp( i0, 0.0, 1.0 );
  vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );
  vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );

  //  x0 = x0 - 0.0 + 0.0 * C
  vec4 x1 = x0 - i1 + 1.0 * C.xxxx;
  vec4 x2 = x0 - i2 + 2.0 * C.xxxx;
  vec4 x3 = x0 - i3 + 3.0 * C.xxxx;
  vec4 x4 = x0 - 1.0 + 4.0 * C.xxxx;

// Permutations
  i = mod(i, 289.0);
  float j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x);
  vec4 j1 = permute( permute( permute( permute (
             i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))
           + i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))
           + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))
           + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));
// Gradients
// ( 7*7*6 points uniformly over a cube, mapped onto a 4-octahedron.)
// 7*7*6 = 294, which is close to the ring size 17*17 = 289.

  vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;

  vec4 p0 = grad4(j0,   ip);
  vec4 p1 = grad4(j1.x, ip);
  vec4 p2 = grad4(j1.y, ip);
  vec4 p3 = grad4(j1.z, ip);
  vec4 p4 = grad4(j1.w, ip);

// Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  p4 *= taylorInvSqrt(dot(p4,p4));

// Mix contributions from the five corners
  vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
  vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)            ), 0.0);
  m0 = m0 * m0;
  m1 = m1 * m1;
  return 49.0 * ( dot(m0*m0, vec3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 )))
               + dot(m1*m1, vec2( dot( p3, x3 ), dot( p4, x4 ) ) ) ) ;

}







float rand(vec2 n) {
    return fract(sin(dot(n, vec2(12.9898,12.1414))) * 83758.5453);
}

float noise(vec2 n) {
    const vec2 d = vec2(0.0, 1.0);
    vec2 b = floor(n);
    vec2 f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
    return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}


float fire(vec2 n) {
    return noise(n) + noise(n * 2.1) * .6 + noise(n * 5.4) * .42;
}





vec3 transformed(vec3 p)
{
    return (inverseWorld * vec4(p, 1.0)).xyz;
}

float sdSphere(vec3 p, float s)
{
	return length(p) - s;
}

float sdBox( vec3 p, vec3 b )
{
    return length(max(abs(transformed(p))-b,0.0));
}

//vec3 sphereNormal(vec3 p, float s) {
//    return normalize(vec3(
//        simplex3d_fractal(vec3(p.x + EPSILON, p.y, p.z)) - simplex3d_fractal(vec3(p.x - EPSILON, p.y, p.z)),
//        simplex3d_fractal(vec3(p.x, p.y + EPSILON, p.z)) - simplex3d_fractal(vec3(p.x, p.y - EPSILON, p.z)),
//        simplex3d_fractal(vec3(p.x, p.y, p.z + EPSILON)) - simplex3d_fractal(vec3(p.x, p.y, p.z - EPSILON))
//    ));
//}

float rmSphere(vec3 p, float s) {
    float color = sdSphere(p, s);
    if (color > 0.0001) {
        return 1.0;
    }

    return 0.0;
}


vec3 sphereNormal(vec3 p, float s) {
    return normalize(vec3(
        sdSphere(vec3(p.x + EPSILON, p.y, p.z), s) - sdSphere(vec3(p.x - EPSILON, p.y, p.z), s),
        sdSphere(vec3(p.x, p.y + EPSILON, p.z), s) - sdSphere(vec3(p.x, p.y - EPSILON, p.z), s),
        sdSphere(vec3(p.x, p.y, p.z + EPSILON), s) - sdSphere(vec3(p.x, p.y, p.z - EPSILON), s)
    ));
}

vec3 boxNormal(vec3 p, vec3 s) {
    return normalize(vec3(
        sdBox(vec3(p.x + EPSILON, p.y, p.z), s) - sdBox(vec3(p.x - EPSILON, p.y, p.z), s),
        sdBox(vec3(p.x, p.y + EPSILON, p.z), s) - sdBox(vec3(p.x, p.y - EPSILON, p.z), s),
        sdBox(vec3(p.x, p.y, p.z + EPSILON), s) - sdBox(vec3(p.x, p.y, p.z - EPSILON), s)
    ));
}



float tri( float x ){
  return abs( fract(x) - .5 );
}

vec3 tri3( vec3 p ){

  return vec3(
      tri( p.z + tri( p.y * 1. ) ),
      tri( p.z + tri( p.x * 1. ) ),
      tri( p.y + tri( p.x * 1. ) )
  );

}


float triNoise3D( vec3 p, float spd , float time){

  float z  = 1.4;
	float rz =  0.;
  vec3  bp =   p;

	for( float i = 0.; i <= 3.; i++ ){

    vec3 dg = tri3( bp * 2. );
    p += ( dg + time * .1 * spd );

    bp *= 1.8;
		z  *= 1.5;
		p  *= 1.2;

    float t = tri( p.z + tri( p.x + tri( p.y )));
    rz += t / z;
    bp += 0.14;

	}

	return rz;

}

vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size / 2.0;
    float z = size.y / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

mat3 viewMatrixBuild(vec3 eye, vec3 center, vec3 up) {
    // Based on gluLookAt man page
    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    return mat3(s, u, -f);
}

float smin( float a, float b, float k )
{
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

float sceneSDF(vec3 samplePoint) {
    vec3 p = transformed(samplePoint);

    float r = 0.2 + pow(triNoise3D(samplePoint * 2.0 + time * vec3(-0.05, 0.1, 0), 1.5, time) * 0.02, 1.0);
    float dt = length(p) - r;

    float d = dt;

    glow += glow_color * .025 / (.01 + d*d);

    return d;
}

vec3 sceneNormal(vec3 p) {
    //vec3 p = transformed(pos);
    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
        sceneSDF(vec3(p.x, p.y, p.z + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

float shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start, float end) {
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = sceneSDF(eye + depth * marchingDirection);
        if (dist < EPSILON) {
            return depth;
        }
        depth += dist;
        if (depth >= end) {
            return end;
        }
    }
    return end;
}

float fresnel(float bias, float scale, float power, vec3 I, vec3 N)
{
    return bias + scale * pow(1.0 + dot(I, N), power);
}

vec3 raymarch(vec2 p, float s) {
    vec3 viewDir = rayDirection(70.0, resolution.xy, gl_FragCoord.xy);
    vec3 eye = cameraPosition;

    mat3 viewToWorld = viewMatrixBuild(eye, vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));

    vec3 worldDir = viewToWorld * viewDir;

    float dist = shortestDistanceToSurface(eye, viewDir, MIN_DIST, MAX_DIST);

    if (dist > MAX_DIST - EPSILON) {
        // Didn't hit anything
        return vec3(0.0, 0.0, 0.0);
    }

    vec3 pos = eye + dist * worldDir;
    //vec3 pos = eye + dist * worldDir;viewDir

    vec3 normal = sceneNormal(pos);

    vec3 diff1Color = vec3 ( 1.0, 0.1, 0.1 );
    vec3 diff2Color = vec3 ( 0.0, 0.5, 1.0 );

    vec3 light1Dir = normalize(vec3(0.2, -0.2, 0.4));
    float diffuse1 = max ( dot ( normal, light1Dir ), 0.0 );

    vec3 light2Dir = normalize(vec3(-0.2, 0.2, 0.4));
    float diffuse2 = max ( dot ( normal, light2Dir ), 0.0 );

    float spec1 = pow(diffuse1, 128.);
    float spec2 = pow(diffuse2, 128.);

    float diffuseMain = max ( dot ( normal, -worldDir ), 0.0 );

    //float specular = pow(diffuse, 256.);

    vec3 I = normalize(pos - eye);
    float R = fresnel(0.05, 4.0, 4.0, I, normal);

    vec3 light_color = vec3(0.99, 0.8, 0.4);

    vec3 diffResult = vec3(light_color*diffuse1 + light_color*diffuse2) * 0.2;


    vec3 specRay = reflect(transformed(worldDir), normal);

    vec3 colorTexture = texture2D(textureEnv, transformed(specRay).xy).rgb;


    vec3 result = vec3(
        glow*glow_intensity*diff2Color*diffuse2 +
        glow*glow_intensity*diff1Color*diffuse1 +
        diffResult*diffuse1*diffuse2 +
        spec1 * colorTexture * 0.99 +
        spec2 * colorTexture * 0.99 +
        R*0.3
    );// + specular*0.1

    return result; //noiseNormal(pos, vec3(0.1));
}

void main( void ) {
    vec2 position = vUv;



    vec2 p = (-resolution.xy + 2.0*gl_FragCoord.xy) / resolution.y;

    vec3 color = raymarch(p, 0.4);



    //float specular = pow(diffuse, 32.0);



    gl_FragColor = vec4(color, 1.0);
}