import { defineConfig } from "vite";
import { viteStaticCopy } from 'vite-plugin-static-copy';


export default defineConfig({
    base: './',
    server: {
        headers: {
            // Это важно!
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp",
        },
    },

    // говорим Vite обрабатывать wasm как бинарник
    assetsInclude: [
        "**/*.wasm",
        "**/*.onnx"
    ],
    optimizeDeps: {
        exclude: ["onnxruntime-web"]
    },

    build: {
        target: "esnext",
        assetsDir: './',
        assetsInlineLimit: 0,
    },
    plugins: [
        viteStaticCopy({
            targets: [
                {
                    src: 'node_modules/onnxruntime-web/dist/*.wasm',
                    dest: '.'
                },
                {
                    src: 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs',
                    dest: '.'
                }
            ]
        }),
    ],
});