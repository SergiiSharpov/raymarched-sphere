import * as THREE from 'three';

import vertex from './vertex.glsl';
import fragment from './fragment.glsl';

export default {
    uniforms:
        {
            time: { value: 1.0 },
            resolution: { value: new THREE.Vector2() },
            inverseWorld: { value: new THREE.Matrix4() },
            textureEnv: { type: "t", value: new THREE.TextureLoader().load( "textures/env.jpg" ) }
        },
    vertexShader: vertex,
    fragmentShader: fragment
};