import './style.css'
import * as THREE from 'three'
import { ParametricGeometry } from 'three/examples/jsm/geometries/ParametricGeometry.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import {
  surfaces,
  type SurfaceFn,
  type WorkerInbound,
  type WorkerOutbound,
} from './surfaces.ts'

const app = document.querySelector<HTMLDivElement>('#app')!

const scene = new THREE.Scene()
scene.background = new THREE.Color(0x16171d)

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100)

const renderer = new THREE.WebGLRenderer({ antialias: true })
renderer.setPixelRatio(window.devicePixelRatio)
app.appendChild(renderer.domElement)

const controls = new OrbitControls(camera, renderer.domElement)
controls.enableDamping = true
controls.autoRotate = true
controls.autoRotateSpeed = 1.0

let activeSurface: SurfaceFn = surfaces.ripple
const surfaceFn: SurfaceFn = (u, v, t) => activeSurface(u, v, t)

// Ground truth — translucent wireframe reference.
const truthMat = new THREE.MeshBasicMaterial({
  color: 0x6b6375,
  wireframe: true,
  transparent: true,
  opacity: 0.35,
})
const truthMesh = new THREE.Mesh(new ParametricGeometry(surfaceFn, 80, 80), truthMat)
scene.add(truthMesh)

// Predicted mesh — vertex positions are streamed in from the worker.
const SEG = 60
const VERTS = (SEG + 1) * (SEG + 1)
const predGeom = new THREE.BufferGeometry()
const predPositions = new Float32Array(VERTS * 3)
predGeom.setAttribute('position', new THREE.BufferAttribute(predPositions, 3))
const indices: number[] = []
for (let j = 0; j < SEG; j++) {
  for (let i = 0; i < SEG; i++) {
    const a = j * (SEG + 1) + i
    const b = a + 1
    const c = a + (SEG + 1)
    const d = c + 1
    indices.push(a, c, b, b, c, d)
  }
}
predGeom.setIndex(indices)

const predMat = new THREE.MeshStandardMaterial({
  color: 0xc084fc,
  roughness: 0.4,
  metalness: 0.1,
  side: THREE.DoubleSide,
})
scene.add(new THREE.Mesh(predGeom, predMat))

scene.add(new THREE.AmbientLight(0xffffff, 0.4))
const key = new THREE.DirectionalLight(0xffffff, 1.2)
key.position.set(5, 8, 6)
scene.add(key)
const fill = new THREE.DirectionalLight(0xaa3bff, 0.6)
fill.position.set(-6, 2, -4)
scene.add(fill)

const hud = document.createElement('div')
hud.id = 'hud'
const status = document.createElement('span')
const select = document.createElement('select')
for (const name of Object.keys(surfaces)) {
  const opt = document.createElement('option')
  opt.value = name
  opt.textContent = name
  select.appendChild(opt)
}
select.value = 'klein'
const startBtn = document.createElement('button')
startBtn.textContent = 'Start'
const stopBtn = document.createElement('button')
stopBtn.textContent = 'Stop'
const infoBtn = document.createElement('button')
infoBtn.textContent = '?'
infoBtn.title = 'About'
hud.append(status, select, startBtn, stopBtn, infoBtn)
app.appendChild(hud)

const info = document.createElement('div')
info.id = 'info'
info.hidden = false
info.innerHTML = `
  <p>A small neural network is learning a <strong>parametric surface</strong> —
  a function <code>(u,&nbsp;v) → (x,&nbsp;y,&nbsp;z)</code>.</p>
  <p>The <span class="truth">gray wireframe</span> is the target.
  The <span class="pred">purple mesh</span> is what the network currently predicts.</p>
`
app.appendChild(info)
infoBtn.onclick = () => {
  info.hidden = !info.hidden
}

const worker = new Worker(
  new URL('./training-worker.ts', import.meta.url),
  { type: 'module' },
)
const send = (msg: WorkerInbound) => worker.postMessage(msg)

let step = 0
let loss = NaN

worker.onmessage = (e: MessageEvent<WorkerOutbound>) => {
  const msg = e.data
  if (msg.type === 'prediction') {
    if (msg.positions.length === predPositions.length) {
      predPositions.set(msg.positions)
      predGeom.attributes.position.needsUpdate = true
      predGeom.computeVertexNormals()
    }
    step = msg.step
    loss = msg.loss
  }
}

const setRunning = (r: boolean) => {
  startBtn.disabled = r
  stopBtn.disabled = !r
  send({ type: 'setRunning', running: r })
}
startBtn.onclick = () => setRunning(true)
stopBtn.onclick = () => setRunning(false)
setRunning(false)

select.onchange = () => applySurfaceChange(select.value)

function getTargetStats() {
  truthMesh.geometry.computeBoundingSphere()
  const bs = truthMesh.geometry.boundingSphere!
  return { center: bs.center.clone(), scale: Math.max(bs.radius, 0.1) }
}

function fitCamera() {
  const bs = truthMesh.geometry.boundingSphere!
  controls.target.copy(bs.center)
  camera.position.set(
    bs.center.x + bs.radius * 1.6,
    bs.center.y + bs.radius * 1.0,
    bs.center.z + bs.radius * 1.6,
  )
  camera.lookAt(bs.center)
  controls.update()
}

function applySurfaceChange(name: string) {
  activeSurface = surfaces[name]
  truthMesh.geometry.dispose()
  truthMesh.geometry = new ParametricGeometry(surfaceFn, 80, 80)
  const stats = getTargetStats()
  fitCamera()
  send({
    type: 'setSurface',
    surface: name,
    center: { x: stats.center.x, y: stats.center.y, z: stats.center.z },
    scale: stats.scale,
  })
}

applySurfaceChange('klein')

const resize = () => {
  const w = window.innerWidth
  const h = window.innerHeight
  renderer.setSize(w, h)
  camera.aspect = w / h
  camera.updateProjectionMatrix()
}
resize()
window.addEventListener('resize', resize)

renderer.setAnimationLoop(() => {
  status.textContent = `step ${step}  ·  loss ${Number.isFinite(loss) ? loss.toExponential(3) : '—'}`
  controls.update()
  renderer.render(scene, camera)
})
