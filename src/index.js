import * as THREE from 'three';
import raySphere from './shaders/marchedsphere/index';
import * as _ from 'lodash';

let camera, scene, renderer;
let geometry, material, mesh, texture;
let shaderMaterial, shaderData;
let clock;

init();
animate();

//console.log(_.assign);

function init() {

    camera = new THREE.PerspectiveCamera( 70, window.innerWidth / window.innerHeight, 0.01, 10 );
    camera.position.z = 1;

    clock = new THREE.Clock();

    scene = new THREE.Scene();

    // texture = new THREE.TextureLoader().load( "textures/env.jpg" );

    shaderData = _.extend({}, raySphere);

    shaderMaterial = new THREE.ShaderMaterial(shaderData);

    geometry = new THREE.BoxGeometry( 0.9, 0.9, 0.9 );//new THREE.PlaneGeometry( 1, 1, 32 );
    material = shaderMaterial;//new THREE.MeshNormalMaterial();

    mesh = new THREE.Mesh( geometry, material );
    scene.add( mesh );

    // mesh.rotation.y = 0.5;
    // mesh.rotation.x = 0.4;

    renderer = new THREE.WebGLRenderer( { antialias: true } );
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );


    window.addEventListener( 'resize', onWindowResize, false );

}

function onWindowResize( event ) {
    const SCREEN_WIDTH = window.innerWidth;
    const SCREEN_HEIGHT = window.innerHeight;

    let aspect = SCREEN_WIDTH / SCREEN_HEIGHT;

    renderer.setSize( SCREEN_WIDTH, SCREEN_HEIGHT );
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
}

function animate() {

    requestAnimationFrame( animate );

    clock.getDelta();

    shaderData.uniforms.resolution.value.x = window.innerWidth;
    shaderData.uniforms.resolution.value.y = window.innerHeight;

    shaderData.uniforms.inverseWorld.value = mesh.matrixWorld.getInverse(mesh.matrixWorld);
    //getInverse

    shaderData.uniforms.time.value = clock.elapsedTime;

    mesh.rotation.x += 0.01;
    mesh.rotation.y += 0.02;

    renderer.render( scene, camera );

}