varying vec2 vUv;

varying mat4 model;

void main()
{
    vUv = uv;
    model = modelMatrix;
    vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
    gl_Position = projectionMatrix * mvPosition;
}