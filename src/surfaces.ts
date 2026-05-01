export interface Vec3Target {
  set(x: number, y: number, z: number): void
}
export type SurfaceFn = (u: number, v: number, t: Vec3Target) => void

// (u, v) ∈ [0, 1]² → (x, y, z)
export const surfaces: Record<string, SurfaceFn> = {
  klein: (u, v, t) => {
    const U = u * Math.PI * 2
    const V = v * Math.PI * 2
    const s = 0.2
    let x, z
    if (U < Math.PI) {
      x = 3 * Math.cos(U) * (1 + Math.sin(U)) + 2 * (1 - Math.cos(U) / 2) * Math.cos(U) * Math.cos(V)
      z = -8 * Math.sin(U) - 2 * (1 - Math.cos(U) / 2) * Math.sin(U) * Math.cos(V)
    } else {
      x = 3 * Math.cos(U) * (1 + Math.sin(U)) + 2 * (1 - Math.cos(U) / 2) * Math.cos(V + Math.PI)
      z = -8 * Math.sin(U)
    }
    const y = -2 * (1 - Math.cos(U) / 2) * Math.sin(V)
    t.set(x * s, y * s, z * s)
  },
  ripple: (u, v, t) => {
    const U = (u - 0.5) * 4
    const V = (v - 0.5) * 4
    t.set(U, Math.sin(U * 1.5) * Math.cos(V * 1.5) * 0.8, V)
  },
  saddle: (u, v, t) => {
    const U = (u - 0.5) * 3
    const V = (v - 0.5) * 3
    t.set(U, (U * U - V * V) * 0.25, V)
  },
  sphere: (u, v, t) => {
    const phi = v * Math.PI
    const theta = u * Math.PI * 2
    const r = 1.4
    t.set(
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.cos(phi),
      r * Math.sin(phi) * Math.sin(theta),
    )
  },
  torus: (u, v, t) => {
    const R = 1.3
    const r = 0.45
    const U = u * Math.PI * 2
    const V = v * Math.PI * 2
    t.set(
      (R + r * Math.cos(V)) * Math.cos(U),
      r * Math.sin(V),
      (R + r * Math.cos(V)) * Math.sin(U),
    )
  },
  cone: (u, v, t) => {
    const theta = u * Math.PI * 2
    const h = (v - 0.5) * 3
    const r = (1 - v) * 1.5
    t.set(r * Math.cos(theta), h, r * Math.sin(theta))
  },

  plane: (u, v, t) => {
    t.set((u - 0.5) * 4, 0, (v - 0.5) * 4)
  },
  mobius: (u, v, t) => {
    const offset = u - 0.5
    const angle = 2 * Math.PI * v
    const a = 2
    t.set(
      Math.cos(angle) * (a + offset * Math.cos(angle / 2)),
      Math.sin(angle) * (a + offset * Math.cos(angle / 2)),
      offset * Math.sin(angle / 2),
    )
  },
  mobius3d: (u, v, t) => {
    const U = u * Math.PI * 2
    const T = v * 2 * Math.PI
    const phi = U / 2
    const major = 2.25, a = 0.125, b = 0.65
    let x = a * Math.cos(T) * Math.cos(phi) - b * Math.sin(T) * Math.sin(phi)
    const z = a * Math.cos(T) * Math.sin(phi) + b * Math.sin(T) * Math.cos(phi)
    const y = (major + x) * Math.sin(U)
    x = (major + x) * Math.cos(U)
    t.set(x, y, z)
  },
}

export interface Vec3Plain {
  x: number
  y: number
  z: number
}

export type WorkerInbound =
  | { type: 'setSurface'; surface: string; center: Vec3Plain; scale: number }
  | { type: 'setRunning'; running: boolean }

export type WorkerOutbound = {
  type: 'prediction'
  positions: Float32Array
  step: number
  loss: number
}
