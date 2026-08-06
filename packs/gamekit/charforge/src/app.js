// charforge viewer — turntable stage for compiled characters.
// The build inlines this bundle plus the character specs into one HTML
// file: no network, no external assets, headless-verifiable.

import * as THREE from 'three';
import { buildCharacter, makeAnimator, buildStage, frameCamera, measureCharacter } from './charforge.js';

const SPECS = window.__CHARFORGE_SPECS__ ?? [];

const state = {
  scene: null, camera: null, renderer: null,
  character: null, animator: null, turn: 0, dragging: false, lastX: 0,
  frames: 0, poseSamples: [],
};

function mount(specIndex) {
  const spec = SPECS[specIndex];
  if (state.character) state.scene.remove(state.character.group);
  state.scene = new THREE.Scene();
  state.character = buildCharacter(spec);
  state.animator = makeAnimator(state.character);
  state.scene.add(state.character.group);
  buildStage(state.scene, state.character);
  frameCamera(state.camera, state.character, innerWidth / innerHeight);
  document.getElementById('name').textContent = state.character.spec.name;
  // Gate hook: expose measurements + live pose so the harness asserts
  // the ASSEMBLED artifact, not the source.
  window.__forge = {
    spec: state.character.spec,
    measure: measureCharacter(state.character),
    frames: () => state.frames,
    // Animation probe: a joint that moves in BOTH modes for either
    // skeleton (tail for creatures, arm swing + body bob for kids).
    pose: () => {
      const j = state.character.joints;
      return (j.tail?.rotation.y ?? 0) + (j.armL?.rotation.x ?? 0) + j.body.position.y;
    },
    setMode: (m) => state.animator.setMode(m),
    mode: () => state.animator.getMode(),
  };
}

function init() {
  // preserveDrawingBuffer: the RENDERS_CONTENT gate reads pixels back
  // outside the frame loop; without this the readback is legally blank
  // and the gate lies (found on the first gate run).
  state.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  state.renderer.setSize(innerWidth, innerHeight);
  state.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  state.renderer.shadowMap.enabled = true;
  state.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  state.renderer.toneMapping = THREE.ACESFilmicToneMapping;
  document.body.appendChild(state.renderer.domElement);
  state.camera = new THREE.PerspectiveCamera(34, innerWidth / innerHeight, 0.1, 100);

  const picker = document.getElementById('picker');
  SPECS.forEach((s, i) => {
    const b = document.createElement('button');
    b.textContent = s.name;
    b.onclick = () => mount(i);
    picker.appendChild(b);
  });
  document.getElementById('walk').onclick = () => {
    const next = state.animator.getMode() === 'walk' ? 'idle' : 'walk';
    state.animator.setMode(next);
    document.getElementById('walk').textContent = next === 'walk' ? 'idle' : 'walk';
  };

  addEventListener('pointerdown', (e) => { state.dragging = true; state.lastX = e.clientX; });
  addEventListener('pointerup', () => { state.dragging = false; });
  addEventListener('pointermove', (e) => {
    if (state.dragging) { state.turn += (e.clientX - state.lastX) * 0.008; state.lastX = e.clientX; }
  });
  addEventListener('resize', () => {
    state.renderer.setSize(innerWidth, innerHeight);
    frameCamera(state.camera, state.character, innerWidth / innerHeight);
  });

  mount(0);
  const clock = new THREE.Clock();
  state.renderer.setAnimationLoop(() => {
    const t = clock.getElapsedTime();
    state.animator.tick(t);
    state.character.group.rotation.y = state.turn + (state.dragging ? 0 : Math.sin(t * 0.25) * 0.12);
    state.renderer.render(state.scene, state.camera);
    state.frames += 1;
  });
}

init();
