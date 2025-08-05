
import * as THREE from "three";

const webcam = document.getElementById("webcam");
const startBtn = document.getElementById("startBtn");
const threeContainer = document.getElementById("three-container");

let poseLandmarker;
let videoStream;

let scene, camera, renderer;

let points, particlePositions, basePositions;
let particleVelocities;

let cohetes = [];
let trailLines = [];

const springFactor = 0.1; // Fuerza del resorte
const damping = 0.8;      // Amortiguación

let leftHandWasRaised = false;
let rightHandWasRaised = false;

startBtn.onclick = async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Cargando modelo...";
  await initPoseLandmarker();
  await enableWebcam();
  await initThreeJS();
  startBtn.style.display = "none";
};

async function initPoseLandmarker() {
  const visionModule = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0"
  );
  const { FilesetResolver, PoseLandmarker } = visionModule;

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

async function enableWebcam() {
  // Recomendado: 640x480 para mejor detección y performance decente
  const constraints = { video: { width: 640, height: 480 } };
  videoStream = await navigator.mediaDevices.getUserMedia(constraints);
  webcam.srcObject = videoStream;
  return new Promise((res) => (webcam.onloadedmetadata = res));
}

async function initThreeJS() {
  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    1,
    1000
  );
  camera.position.z = 320;

  renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x000000, 0);

  threeContainer.appendChild(renderer.domElement);

  const geometry = new THREE.BufferGeometry();
  particlePositions = new Float32Array(33 * 3);
  basePositions = new Float32Array(33 * 3);
  particleVelocities = new Float32Array(33 * 3);

  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(particlePositions, 3)
  );

  const material = new THREE.PointsMaterial({
    color: 0x36d1c4,
    size: 16,
    map: createCircleTexture(),
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.95,
  });

  points = new THREE.Points(geometry, material);
  scene.add(points);

  window.addEventListener("resize", onResize);

  animateScene();
  predictWebcam();
}

function createCircleTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext("2d");
  context.beginPath();
  context.arc(32, 32, 30, 0, 2 * Math.PI);
  context.fillStyle = "#FFFFFF";
  context.fill();
  return new THREE.CanvasTexture(canvas);
}


function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

const landmarkMap = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftWrist: 15,
  rightWrist: 16,
};

// Variables para tracking de velocidad y estado de la mano
const handState = {
  left: {
    lastPos: null,
    lastTime: null,
    lastVelocity: { x: 0, y: 0 },
    isMoving: false,
  },
  right: {
    lastPos: null,
    lastTime: null,
    lastVelocity: { x: 0, y: 0 },
    isMoving: false,
  },
};

function detectFlingAndStop(landmarks, wristIndex, handKey) {
  const wrist = landmarks[wristIndex];
  const now = performance.now();

  const px = (1 - wrist.x - 0.5) * 2 * (webcam.videoWidth / 3);
  const py = -(wrist.y - 0.5) * 2 * (webcam.videoHeight / 3);

  const state = handState[handKey];

  if (state.lastPos && state.lastTime) {
    const dt = (now - state.lastTime) / 1000;
    const dx = px - state.lastPos.x;
    const dy = py - state.lastPos.y;
    const vx = dx / dt;
    const vy = dy / dt;
    const speed = Math.sqrt(vx * vx + vy * vy);

    const MOVING_THRESHOLD = 900;
    const STOPPED_THRESHOLD = 180;

    // Detecta inicio de movimiento rápido
    if (!state.isMoving && speed > MOVING_THRESHOLD) {
      state.isMoving = true;
    }
    // Detecta parada después del movimiento rápido: lanza cohete
    if (state.isMoving && speed < STOPPED_THRESHOLD) {
      // Normaliza dirección, mantiene potencia constante
      const norm =
        Math.sqrt(state.lastVelocity.x ** 2 + state.lastVelocity.y ** 2) || 1;
      const factor = 20;
      launchCohete(
        px,
        py,
        (state.lastVelocity.x / norm) * factor,
        (state.lastVelocity.y / norm) * factor
      );
      state.isMoving = false;
    }
    // Actualiza dirección de velocidad mientras se mueve rápido
    if (speed > MOVING_THRESHOLD) {
      state.lastVelocity = { x: vx, y: vy };
    }
  }

  state.lastPos = { x: px, y: py };
  state.lastTime = now;
}


async function predictWebcam() {
  if (!poseLandmarker || webcam.readyState < 2) {
    requestAnimationFrame(predictWebcam);
    return;
  }

  const result = await poseLandmarker.detectForVideo(webcam, performance.now());
  if (result.landmarks.length > 0) {
    const landmarks = result.landmarks[0];

    for (let i = 0; i < landmarks.length; i++) {
      const p = landmarks[i];
      const idx = i * 3;
      const px = (1 - p.x - 0.5) * 2 * (webcam.videoWidth / 3);
      const py = -(p.y - 0.5) * 2 * (webcam.videoHeight / 3);
      const pz = -p.z * 100;
      basePositions[idx] = px;
      basePositions[idx + 1] = py;
      basePositions[idx + 2] = pz;
    }
    detectFlingAndStop(landmarks, landmarkMap.rightWrist, "right");
    detectFlingAndStop(landmarks, landmarkMap.leftWrist, "left");
  }
  requestAnimationFrame(predictWebcam);
}

/*
 * 
function detectHandRaise(landmarks) {
    const leftShoulder = landmarks[landmarkMap.leftShoulder];
    const rightShoulder = landmarks[landmarkMap.rightShoulder];
    const leftWrist = landmarks[landmarkMap.leftWrist];
    const rightWrist = landmarks[landmarkMap.rightWrist];

    // Mano sobre el hombro: Y más arriba que el hombro
    const isLeftHandRaised = leftWrist.y < leftShoulder.y;
    const isRightHandRaised = rightWrist.y < rightShoulder.y;

    // Dispara transición de abajo a arriba
    if (isLeftHandRaised && !leftHandWasRaised) {
        const x = (1 - leftWrist.x - 0.5) * 2 * (webcam.videoWidth / 3);
        const y = -(leftWrist.y - 0.5) * 2 * (webcam.videoHeight / 3);
        const vx = (Math.random() - 0.5) * 100;
        const vy = 300;
        launchCohete(x, y, vx, vy);
    }
    if (isRightHandRaised && !rightHandWasRaised) {
        const x = (1 - rightWrist.x - 0.5) * 2 * (webcam.videoWidth / 3);
        const y = -(rightWrist.y - 0.5) * 2 * (webcam.videoHeight / 3);
        const vx = (Math.random() - 0.5) * 100;
        const vy = 300;
        launchCohete(x, y, vx, vy);
    }
    leftHandWasRaised = isLeftHandRaised;
    rightHandWasRaised = isRightHandRaised;
}
*/


function animateScene() {
  const dt = 0.016;

  for (let i = 0; i < 33; i++) {
    const idx = i * 3;

    // Calcula fuerza de resorte hacia basePositions
    const dx = basePositions[idx] - particlePositions[idx];
    const dy = basePositions[idx + 1] - particlePositions[idx + 1];
    const dz = basePositions[idx + 2] - particlePositions[idx + 2];

    // Aplica fuerza a la velocidad
    particleVelocities[idx] += dx * springFactor;
    particleVelocities[idx + 1] += dy * springFactor;
    particleVelocities[idx + 2] += dz * springFactor;

    // Aplica amortiguación (damping)
    particleVelocities[idx] *= damping;
    particleVelocities[idx + 1] *= damping;
    particleVelocities[idx + 2] *= damping;

    // Actualiza la posición
    particlePositions[idx] += particleVelocities[idx];
    particlePositions[idx + 1] += particleVelocities[idx + 1];
    particlePositions[idx + 2] += particleVelocities[idx + 2];
  }
  points.geometry.attributes.position.needsUpdate = true;

  for (let i = cohetes.length - 1; i >= 0; i--) {
    const cohete = cohetes[i];

    // Actualiza posición por velocidad
    cohete.x += cohete.vx * dt;
    cohete.y += cohete.vy * dt;
    cohete.mesh.position.set(cohete.x, cohete.y, 0);

    // Actualiza la estela (trail)
    cohete.trailPoints.push(new THREE.Vector3(cohete.x, cohete.y, 0));
    if (cohete.trailPoints.length > 40) cohete.trailPoints.shift();

    // Elimina y reemplaza el mesh de la estela si existe
    if (cohete.trailMesh) scene.remove(cohete.trailMesh);
    if (cohete.trailPoints.length > 1) {
      const trailGeom = new THREE.BufferGeometry().setFromPoints(
        cohete.trailPoints
      );
      const trailMat = new THREE.LineBasicMaterial({
        color: 0xff3e7c,
        linewidth: 4,
      });
      cohete.trailMesh = new THREE.Line(trailGeom, trailMat);
      scene.add(cohete.trailMesh);
    }

    // Si el cohete sale de límites, explota y lo elimina
    if (Math.abs(cohete.x) > 320 || Math.abs(cohete.y) > 240) {
      if (cohete.trailMesh) scene.remove(cohete.trailMesh);
      scene.remove(cohete.mesh);
      explodeAt(cohete.x, cohete.y);
      cohetes.splice(i, 1); // Elimina del array
    }
  }

  // Renderiza la escena
  renderer.render(scene, camera);
  requestAnimationFrame(animateScene);
}

function launchCohete(x, y, vx, vy) {
  const geometry = new THREE.SphereGeometry(9, 16, 16);
  const material = new THREE.MeshBasicMaterial({ color: 0xff3e7c });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(x, y, 0);
  scene.add(mesh);

  cohetes.push({ x, y, vx, vy, mesh, trailMesh: null, trailPoints: [] });
}


function explodeAt(x, y) {
  const particles = [];
  const geometry = new THREE.BufferGeometry();
  const positions = [];
  const numParticles = 24;

  // Genera partículas en círculo
  for (let i = 0; i < numParticles; i++) {
    const angle = (2 * Math.PI * i) / numParticles;
    const speed = 6 + Math.random() * 3;
    positions.push(x, y, 0);
    particles.push({
      x: x,
      y: y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      alpha: 1,
    });
  }
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );
  const mat = new THREE.PointsMaterial({
    color: 0xffff00,
    size: 10,
    transparent: true,
    opacity: 1,
    map: createCircleTexture(),
  });
  const pts = new THREE.Points(geometry, mat);
  scene.add(pts);

  let ticks = 0;
  function animateExplosion() {
    ticks++;
    for (let i = 0; i < particles.length; i++) {
      particles[i].x += particles[i].vx;
      particles[i].y += particles[i].vy;
      geometry.attributes.position.setXYZ(i, particles[i].x, particles[i].y, 0);
    }
    geometry.attributes.position.needsUpdate = true;
    mat.opacity = Math.max(0, 1 - ticks / 28);
    if (ticks < 28) {
      requestAnimationFrame(animateExplosion);
    } else {
      scene.remove(pts);
      geometry.dispose();
      mat.dispose();
    }
  }
  animateExplosion();
}
