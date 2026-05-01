# parametric-learn

A small neural network learns to fit a parametric surface `(u, v) → (x, y, z)` in real time. The gray wireframe is the target; the purple mesh is the network's current prediction.

Training runs in a Web Worker (TensorFlow.js); rendering is Three.js.

[parametric-learn.webm](https://github.com/user-attachments/assets/ed705225-2449-4d98-8bb0-38778cfc2f77)

## Run

```sh
pnpm install
pnpm dev
```

Pick a surface from the dropdown, hit **Start**, and watch it converge.
