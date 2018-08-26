import * as THREE from 'three';
import raySphere from './shaders/marchedsphere/index';
import * as _ from 'lodash';
import './OrbitControls';

let camera, scene, renderer, controls;
let geometry, material, mesh;
let shaderData;
let clock;

init();
animate();

function init() {

    camera = new THREE.PerspectiveCamera( 70, window.innerWidth / window.innerHeight, 0.01, 10 );
    camera.position.z = 1;

    clock = new THREE.Clock();

    scene = new THREE.Scene();

    controls = new THREE.OrbitControls( camera );
    controls.autoRotate = true;
    controls.enableDamping = true;
    controls.enablePan = false;

    shaderData = _.extend({}, raySphere);

    geometry = new THREE.BoxGeometry( 0.9, 0.9, 0.9 );
    material = new THREE.ShaderMaterial(shaderData);

    mesh = new THREE.Mesh( geometry, material );
    scene.add( mesh );

    renderer = new THREE.WebGLRenderer( { antialias: true } );
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );


    window.addEventListener( 'resize', onWindowResize, false );

}

function onWindowResize() {
    const SCREEN_WIDTH = window.innerWidth;
    const SCREEN_HEIGHT = window.innerHeight;

    let aspect = SCREEN_WIDTH / SCREEN_HEIGHT;

    renderer.setSize( SCREEN_WIDTH, SCREEN_HEIGHT );
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
}

function animate() {

    requestAnimationFrame( animate );

    controls.update();

    clock.getDelta();

    shaderData.uniforms.resolution.value.x = window.innerWidth;
    shaderData.uniforms.resolution.value.y = window.innerHeight;

    shaderData.uniforms.inverseWorld.value = mesh.matrixWorld.getInverse(mesh.matrixWorld);

    shaderData.uniforms.time.value = clock.elapsedTime;

    mesh.rotation.x += 0.01;
    mesh.rotation.y += 0.02;

    renderer.render( scene, camera );

}