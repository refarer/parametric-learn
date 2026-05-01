# parametric-learn

A small neural network learns to fit a parametric surface `(u, v) → (x, y, z)` in real time. The gray wireframe is the target; the purple mesh is the network's current prediction.

Training runs in a Web Worker (TensorFlow.js); rendering is Three.js.

## Run

```sh
pnpm install
pnpm dev
```

Pick a surface from the dropdown, hit **Start**, and watch it converge.
