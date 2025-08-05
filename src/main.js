import * as THREE from "three";

const webcam = document.getElementById("webcam");
const startBtn = document.getElementById("startBtn");
const threeContainer = document.getElementById("three-container");

let handLandmarker;
let videoStream;
let scene, camera, renderer;
let points, particlePositions, basePositions;
let cohetes = [];
let trailLines = [];

let particleVelocities;
const springFactor = 0.1;
const damping = 0.8;

// MODIFICADO: Ahora el estado de las manos se inicializa en un objeto
const handStates = {
  Left: { isClosed: false },
  Right: { isClosed: false },
};

// ----------- Inicialización -----------

startBtn.onclick = async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Cargando modelo...";
  await initHandLandmarker();
  await enableWebcam();
  await initThreeJS();
  startBtn.style.display = "none";
};

async function initHandLandmarker() {
  const visionModule = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0"
  );
  const { FilesetResolver, HandLandmarker } = visionModule;

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

async function enableWebcam() {
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

  // Landmarks
  const numHandLandmarks = 21 * 2;
  const geometry = new THREE.BufferGeometry();
  particlePositions = new Float32Array(numHandLandmarks * 3);
  basePositions = new Float32Array(numHandLandmarks * 3);
  particleVelocities = new Float32Array(numHandLandmarks * 3);

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

// ----------- Hand Detection y Animación -----------

const fingerTipIndices = [4, 8, 12, 16, 20];
const knuckleIndices = [2, 6, 10, 14, 18];
const wristIndex = 0;

// MODIFICADO: Función más robusta para detectar si la mano está cerrada
function isHandClosed(landmarks) {
  // Verificamos si la punta de cada dedo está por debajo (más arriba en el plano Y)
  // de su nudillo correspondiente. Usamos el plano XZ para mayor robustez
  // frente a la orientación de la mano.
  
  // No verificamos el pulgar, ya que su movimiento es diferente
  let closedFingers = 0;
  for (let i = 1; i < fingerTipIndices.length; i++) {
    const tip = new THREE.Vector3(landmarks[fingerTipIndices[i]].x, landmarks[fingerTipIndices[i]].y, landmarks[fingerTipIndices[i]].z);
    const knuckle = new THREE.Vector3(landmarks[knuckleIndices[i]].x, landmarks[knuckleIndices[i]].y, landmarks[knuckleIndices[i]].z);
    
    // Si la punta del dedo está por debajo de la posición del nudillo (en el eje Y)
    // considera la mano cerrada
    if (tip.y > knuckle.y) {
      closedFingers++;
    }
  }

  // Consideramos la mano cerrada si la mayoría de los dedos están cerrados
  return closedFingers >= 3;
}

// NUEVO: Función para obtener la dirección de la mano, ahora con más robustez
function getHandDirection(landmarks) {
  const wrist = new THREE.Vector3(landmarks[0].x, landmarks[0].y, landmarks[0].z);
  const indexFingerBase = new THREE.Vector3(landmarks[5].x, landmarks[5].y, landmarks[5].z);
  const pinkyBase = new THREE.Vector3(landmarks[17].x, landmarks[17].y, landmarks[17].z);
  
  // Vector de la base del índice a la base del meñique
  const handDirection = new THREE.Vector3().subVectors(indexFingerBase, pinkyBase).normalize();

  // Obtenemos un vector que va desde la base del índice al nudillo del dedo medio
  const middleFingerKnuckle = new THREE.Vector3(landmarks[10].x, landmarks[10].y, landmarks[10].z);
  const handNormal = new THREE.Vector3().subVectors(middleFingerKnuckle, wrist).normalize();

  // La dirección final es una combinación de estos vectores
  const direction = new THREE.Vector3().crossVectors(handDirection, handNormal).normalize();
  return direction;
}

async function predictWebcam() {
  if (!handLandmarker || webcam.readyState < 2) {
    requestAnimationFrame(predictWebcam);
    return;
  }

  const result = await handLandmarker.detectForVideo(webcam, performance.now());
  const numLandmarksPerHand = 21;
  
  // Limpiamos las posiciones base para que los puntos desaparezcan si no hay manos
  basePositions.fill(0);
  
  if (result.landmarks.length > 0) {
    for (let i = 0; i < result.landmarks.length; i++) {
      const landmarks = result.landmarks[i];
      const handedness = result.handednesses[i][0].categoryName;
      
      // Actualiza las posiciones objetivo de los landmarks
      for (let j = 0; j < landmarks.length; j++) {
        const p = landmarks[j];
        const idx = (i * numLandmarksPerHand + j) * 3;
        const px = (1 - p.x - 0.5) * 2 * (webcam.videoWidth / 3);
        const py = -(p.y - 0.5) * 2 * (webcam.videoHeight / 3);
        const pz = -p.z * 100;
        basePositions[idx] = px;
        basePositions[idx + 1] = py;
        basePositions[idx + 2] = pz;
      }
      
      // Lógica para detectar el gesto
      const isHandCurrentlyClosed = isHandClosed(landmarks);
      
      if (handStates[handedness]) {
        // Disparamos solo en la transición de "cerrado" a "abierto"
        if (!isHandCurrentlyClosed && handStates[handedness].isClosed) {
            const wristLandmark = landmarks[wristIndex];
            const x = (1 - wristLandmark.x - 0.5) * 2 * (webcam.videoWidth / 3);
            const y = -(wristLandmark.y - 0.5) * 2 * (webcam.videoHeight / 3);
            
            const direction = getHandDirection(landmarks);
            const factor = 400; // Ajusta este valor para la velocidad del cohete
            
            launchCohete(x, y, direction.x * factor, direction.y * factor);
        }
        // Actualizamos el estado de la mano para la próxima iteración
        handStates[handedness].isClosed = isHandCurrentlyClosed;
      }
    }
  }
  
  requestAnimationFrame(predictWebcam);
}

function animateScene() {
  const dt = 0.016;
  const numHandLandmarks = 21 * 2;

  for (let i = 0; i < numHandLandmarks; i++) {
    const idx = i * 3;

    const dx = basePositions[idx] - particlePositions[idx];
    const dy = basePositions[idx + 1] - particlePositions[idx + 1];
    const dz = basePositions[idx + 2] - particlePositions[idx + 2];

    particleVelocities[idx] += dx * springFactor;
    particleVelocities[idx + 1] += dy * springFactor;
    particleVelocities[idx + 2] += dz * springFactor;

    particleVelocities[idx] *= damping;
    particleVelocities[idx + 1] *= damping;
    particleVelocities[idx + 2] *= damping;

    particlePositions[idx] += particleVelocities[idx];
    particlePositions[idx + 1] += particleVelocities[idx + 1];
    particlePositions[idx + 2] += particleVelocities[idx + 2];
  }
  points.geometry.attributes.position.needsUpdate = true;

  for (let i = cohetes.length - 1; i >= 0; i--) {
    const cohete = cohetes[i];

    cohete.x += cohete.vx * dt;
    cohete.y += cohete.vy * dt;
    cohete.mesh.position.set(cohete.x, cohete.y, 0);

    cohete.trailPoints.push(new THREE.Vector3(cohete.x, cohete.y, 0));
    if (cohete.trailPoints.length > 40) cohete.trailPoints.shift();

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

    if (Math.abs(cohete.x) > 320 || Math.abs(cohete.y) > 240) {
      if (cohete.trailMesh) scene.remove(cohete.trailMesh);
      scene.remove(cohete.mesh);
      explodeAt(cohete.x, cohete.y);
      cohetes.splice(i, 1);
    }
  }

  renderer.render(scene, camera);
  requestAnimationFrame(animateScene);
}
// ----------- Cohete y Explosión -----------

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