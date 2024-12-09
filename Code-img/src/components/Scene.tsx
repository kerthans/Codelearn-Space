import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { SMAAPass } from 'three/examples/jsm/postprocessing/SMAAPass';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

// Mouse particle system
const createParticleSystem = () => {
  const particleCount = 1000;
  const positions = new Float32Array(particleCount * 3);
  const velocities = new Float32Array(particleCount * 3);

  for (let i = 0; i < particleCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 50;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 50;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 50;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  
  const material = new THREE.PointsMaterial({
    size: 0.1,
    color: 0xffffff,
    transparent: true,
    opacity: 0.6,
    blending: THREE.AdditiveBlending
  });

  return {
    points: new THREE.Points(geometry, material),
    velocities
  };
};

const vertexShader = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vPosition;
  uniform float time;
  uniform float deformStrength;
  
  void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    vPosition = position;
    
    vec3 transformed = position;
    transformed.x += sin(position.y * 2.0 + time) * deformStrength;
    transformed.y += cos(position.x * 2.0 + time) * deformStrength;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);
  }
`;

const fragmentShader = `
  uniform float time;
  uniform vec3 baseColor;
  uniform sampler2D texture1;
  uniform float hologramStrength;
  uniform float fresnelStrength;
  
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vPosition;
  
  void main() {
    vec4 texColor = texture2D(texture1, vUv);
    float hologram = abs(sin(vPosition.y * 10.0 + time)) * hologramStrength;
    float fresnel = pow(1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0))), 3.0) * fresnelStrength;
    vec3 finalColor = baseColor * texColor.rgb + vec3(hologram) + vec3(fresnel);
    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

interface SceneControls {
  deformStrength: number;
  hologramStrength: number;
  fresnelStrength: number;
  bloomStrength: number;
  lightIntensity: number;
}

export default function Scene() {
  const mountRef = useRef<HTMLDivElement>(null);
  const mouseRef = useRef({ x: 0, y: 0 });
  const [controls, setControls] = useState<SceneControls>({
    deformStrength: 0.2,
    hologramStrength: 0.3,
    fresnelStrength: 1.0,
    bloomStrength: 1.5,
    lightIntensity: 1.0
  });

  const sceneRef = useRef<{
    renderer: THREE.WebGLRenderer | null;
    composer: EffectComposer | null;
    geometry: THREE.BufferGeometry | null;
    material: THREE.ShaderMaterial | null;
    controls: OrbitControls | null;
    lights: THREE.PointLight[];
    particleSystem: { points: THREE.Points; velocities: Float32Array } | null;
    bloomPass: UnrealBloomPass | null;
  }>({
    renderer: null,
    composer: null,
    geometry: null,
    material: null,
    controls: null,
    lights: [],
    particleSystem: null,
    bloomPass: null
  });

  const handleMouseMove = (event: MouseEvent) => {
    mouseRef.current = {
      x: (event.clientX / window.innerWidth) * 2 - 1,
      y: -(event.clientY / window.innerHeight) * 2 + 1
    };
  };

  useEffect(() => {
    if (!mountRef.current) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      powerPreference: "high-performance"
    });
    
    sceneRef.current.renderer = renderer;
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mountRef.current.appendChild(renderer.domElement);

    const textureLoader = new THREE.TextureLoader();
    const texture = textureLoader.load('/textures/pic.jpg');

    const geometry = new THREE.TorusKnotGeometry(10, 3, 200, 32);
    sceneRef.current.geometry = geometry;
    
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        time: { value: 0 },
        baseColor: { value: new THREE.Color(0.5, 0.8, 1.0) },
        texture1: { value: texture },
        deformStrength: { value: controls.deformStrength },
        hologramStrength: { value: controls.hologramStrength },
        fresnelStrength: { value: controls.fresnelStrength }
      }
    });
    sceneRef.current.material = material;

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const lights = [];
    for(let i = 0; i < 3; i++) {
      const light = new THREE.PointLight(0xffffff, controls.lightIntensity, 100);
      light.position.set(
        Math.sin(i * Math.PI * 2/3) * 20,
        Math.cos(i * Math.PI * 2/3) * 20,
        20
      );
      scene.add(light);
      lights.push(light);
    }
    sceneRef.current.lights = lights;

    // Particle system
    const particleSystem = createParticleSystem();
    scene.add(particleSystem.points);
    sceneRef.current.particleSystem = particleSystem;

    const composer = new EffectComposer(renderer);
    sceneRef.current.composer = composer;
    composer.addPass(new RenderPass(scene, camera));

    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      controls.bloomStrength, 0.4, 0.85
    );
    sceneRef.current.bloomPass = bloomPass;
    composer.addPass(bloomPass);

    const smaaPass = new SMAAPass(
      window.innerWidth * renderer.getPixelRatio(),
      window.innerHeight * renderer.getPixelRatio()
    );
    composer.addPass(smaaPass);

    camera.position.z = 30;
    const orbitControls = new OrbitControls(camera, renderer.domElement);
    sceneRef.current.controls = orbitControls;
    orbitControls.enableDamping = true;
    orbitControls.dampingFactor = 0.05;

    let lastTime = 0;
    const frameRateLimit = 1000 / 60;

    function updateParticles() {
      if (!sceneRef.current.particleSystem) return;
      
      const positions = sceneRef.current.particleSystem.points.geometry.attributes.position.array;
      const velocities = sceneRef.current.particleSystem.velocities;

      for (let i = 0; i < positions.length; i += 3) {
        // Update velocities based on mouse position
        velocities[i] += (mouseRef.current.x * 0.1 - velocities[i]) * 0.01;
        velocities[i + 1] += (mouseRef.current.y * 0.1 - velocities[i + 1]) * 0.01;

        // Update positions
        positions[i] += velocities[i];
        positions[i + 1] += velocities[i + 1];
        positions[i + 2] += velocities[i + 2];

        // Boundary check
        if (Math.abs(positions[i]) > 25) velocities[i] *= -0.9;
        if (Math.abs(positions[i + 1]) > 25) velocities[i + 1] *= -0.9;
        if (Math.abs(positions[i + 2]) > 25) velocities[i + 2] *= -0.9;
      }

      sceneRef.current.particleSystem.points.geometry.attributes.position.needsUpdate = true;
    }

    function animate(currentTime: number) {
      const deltaTime = currentTime - lastTime;
      
      if (deltaTime > frameRateLimit) {
        material.uniforms.time.value = currentTime * 0.001;
        
        lights.forEach((light, i) => {
          const time = currentTime * 0.001;
          light.position.x = Math.sin(time + i * Math.PI * 2/3) * 20;
          light.position.y = Math.cos(time + i * Math.PI * 2/3) * 20;
        });

        updateParticles();
        orbitControls.update();
        composer.render();
        lastTime = currentTime;
      }
      
      requestAnimationFrame(animate);
    }

    animate(0);

    window.addEventListener('mousemove', handleMouseMove);

    function handleResize() {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
      composer.setSize(window.innerWidth, window.innerHeight);
    }

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
      mountRef.current?.removeChild(renderer.domElement);
      
      sceneRef.current.geometry?.dispose();
      sceneRef.current.material?.dispose();
      sceneRef.current.renderer?.dispose();
      sceneRef.current.composer?.dispose();
      sceneRef.current.controls?.dispose();
      sceneRef.current.particleSystem?.points.geometry.dispose();
      sceneRef.current.particleSystem?.points.material.dispose();
      
      sceneRef.current = {
        renderer: null,
        composer: null,
        geometry: null,
        material: null,
        controls: null,
        lights: [],
        particleSystem: null,
        bloomPass: null
      };
    };
  }, []);

  // Update uniforms and parameters when controls change
  useEffect(() => {
    if (sceneRef.current.material) {
      sceneRef.current.material.uniforms.deformStrength.value = controls.deformStrength;
      sceneRef.current.material.uniforms.hologramStrength.value = controls.hologramStrength;
      sceneRef.current.material.uniforms.fresnelStrength.value = controls.fresnelStrength;
    }

    if (sceneRef.current.bloomPass) {
      sceneRef.current.bloomPass.strength = controls.bloomStrength;
    }

    sceneRef.current.lights.forEach(light => {
      light.intensity = controls.lightIntensity;
    });
  }, [controls]);

  return (
    <div className="relative w-full h-full">
      <div ref={mountRef} className="w-full h-full" />
      <div className="absolute top-4 left-4 bg-black/50 p-4 rounded-lg text-white">
        <div className="space-y-2">
          <div>
            <label className="block text-sm">Deform Strength</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={controls.deformStrength}
              onChange={(e) => setControls(prev => ({ ...prev, deformStrength: parseFloat(e.target.value) }))}
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-sm">Hologram Strength</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={controls.hologramStrength}
              onChange={(e) => setControls(prev => ({ ...prev, hologramStrength: parseFloat(e.target.value) }))}
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-sm">Fresnel Strength</label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.01"
              value={controls.fresnelStrength}
              onChange={(e) => setControls(prev => ({ ...prev, fresnelStrength: parseFloat(e.target.value) }))}
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-sm">Bloom Strength</label>
            <input
              type="range"
              min="0"
              max="3"
              step="0.1"
              value={controls.bloomStrength}
              onChange={(e) => setControls(prev => ({ ...prev, bloomStrength: parseFloat(e.target.value) }))}
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-sm">Light Intensity</label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={controls.lightIntensity}
              onChange={(e) => setControls(prev => ({ ...prev, lightIntensity: parseFloat(e.target.value) }))}
              className="w-full"
            />
          </div>
        </div>
      </div>
    </div>
  );
}