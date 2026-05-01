import * as tf from '@tensorflow/tfjs'
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm'
import wasmUrl from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm?url'
import wasmSimdUrl from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm?url'
import wasmThreadedSimdUrl from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm?url'
import {
  surfaces,
  type SurfaceFn,
  type Vec3Plain,
  type WorkerInbound,
  type WorkerOutbound,
} from './surfaces.ts'

setWasmPaths({
  'tfjs-backend-wasm.wasm': wasmUrl,
  'tfjs-backend-wasm-simd.wasm': wasmSimdUrl,
  'tfjs-backend-wasm-threaded-simd.wasm': wasmThreadedSimdUrl,
})

const ctx = self as unknown as {
  onmessage: ((e: MessageEvent<WorkerInbound>) => void) | null
  postMessage(message: WorkerOutbound, transfer?: Transferable[]): void
}

class Vec3 {
  x = 0
  y = 0
  z = 0
  set(x: number, y: number, z: number) {
    this.x = x
    this.y = y
    this.z = z
  }
}

const SEG = 60
const VERTS = (SEG + 1) * (SEG + 1)

const grid = new Float32Array(VERTS * 2)
for (let j = 0; j <= SEG; j++) {
  for (let i = 0; i <= SEG; i++) {
    const idx = j * (SEG + 1) + i
    grid[idx * 2] = i / SEG
    grid[idx * 2 + 1] = j / SEG
  }
}
const gridTensor = tf.tensor2d(grid, [VERTS, 2])

const BATCH = 1024
const STEPS_PER_POST = 1
const tmp = new Vec3()

let activeSurface: SurfaceFn = surfaces.ripple
let model: tf.Sequential | null = null
let optimizer: tf.Optimizer | null = null
let running = false
let step = 0
let loss = NaN

let pendingSurface: { surface: string; center: Vec3Plain; scale: number } | null = null

function buildModel(center: Vec3Plain, scale: number) {
  const m = tf.sequential()
  m.add(tf.layers.dense({
    inputShape: [2],
    units: 128,
    activation: 'tanh',
    kernelInitializer: tf.initializers.randomUniform({ minval: -15, maxval: 15 }),
    biasInitializer: tf.initializers.randomUniform({ minval: -10, maxval: 10 }),
  }))
  m.add(tf.layers.dense({ units: 128, activation: 'tanh' }))
  m.add(tf.layers.dense({ units: 128, activation: 'tanh' }))
  m.add(tf.layers.dense({ units: 128, activation: 'tanh' }))
  m.add(tf.layers.dense({
    units: 3,
    kernelInitializer: tf.initializers.randomUniform({
      minval: -scale * 0.2,
      maxval: scale * 0.2,
    }),
    biasInitializer: 'zeros',
  }))
  tf.tidy(() => {
    m.predict(tf.zeros([1, 2]))
  })
  const last = m.layers[m.layers.length - 1]
  const [kernel] = last.getWeights()
  const newBias = tf.tensor1d([center.x, center.y, center.z])
  last.setWeights([kernel, newBias])
  newBias.dispose()
  return m
}

async function trainStep(): Promise<number> {
  const us = new Float32Array(BATCH * 2)
  const ys = new Float32Array(BATCH * 3)
  for (let i = 0; i < BATCH; i++) {
    const u = Math.random()
    const v = Math.random()
    us[i * 2] = u
    us[i * 2 + 1] = v
    activeSurface(u, v, tmp)
    ys[i * 3] = tmp.x
    ys[i * 3 + 1] = tmp.y
    ys[i * 3 + 2] = tmp.z
  }
  const lossT = tf.tidy(() => {
    const xT = tf.tensor2d(us, [BATCH, 2])
    const yT = tf.tensor2d(ys, [BATCH, 3])
    return optimizer!.minimize(() => {
      const pred = model!.predict(xT) as tf.Tensor
      return tf.losses.meanSquaredError(yT, pred) as tf.Scalar
    }, true) as tf.Scalar
  }) as tf.Scalar
  const value = (await lossT.data())[0]
  lossT.dispose()
  return value
}

async function predictGrid(): Promise<Float32Array> {
  const out = tf.tidy(() => model!.predict(gridTensor) as tf.Tensor)
  const data = (await out.data()) as Float32Array
  out.dispose()
  return data
}

let needsRepaint = false

function applyPendingSurface() {
  if (!pendingSurface) return false
  const p = pendingSurface
  pendingSurface = null
  activeSurface = surfaces[p.surface]
  if (model) model.dispose()
  model = buildModel(p.center, p.scale)
  optimizer = tf.train.adam(0.005)
  step = 0
  loss = NaN
  return true
}

async function loop() {
  while (true) {
    if (applyPendingSurface()) needsRepaint = true
    if (!model) {
      await new Promise((r) => setTimeout(r, 60))
      continue
    }
    if (!running) {
      if (needsRepaint) {
        const data = await predictGrid()
        ctx.postMessage(
          { type: 'prediction', positions: data, step, loss },
          [data.buffer],
        )
        needsRepaint = false
      }
      await new Promise((r) => setTimeout(r, 60))
      continue
    }
    for (let i = 0; i < STEPS_PER_POST; i++) {
      loss = await trainStep()
      step++
    }
    const data = await predictGrid()
    ctx.postMessage(
      { type: 'prediction', positions: data, step, loss },
      [data.buffer],
    )
    needsRepaint = false
    // Yield to the macrotask queue so onmessage (setRunning, setSurface) can fire.
    await new Promise((r) => setTimeout(r, 0))
  }
}

ctx.onmessage = (e) => {
  const msg = e.data
  if (msg.type === 'setSurface') {
    pendingSurface = {
      surface: msg.surface,
      center: msg.center,
      scale: msg.scale,
    }
  } else if (msg.type === 'setRunning') {
    running = msg.running
  }
}

async function init() {
  await tf.setBackend('wasm')
  await tf.ready()
  console.log('[worker] tf backend:', tf.getBackend())
  loop()
}
init()
