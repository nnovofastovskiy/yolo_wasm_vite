import * as ort from "onnxruntime-web";
ort.env.wasm.wasmPaths = "/wasm/";
export async function loadModelWithFallback() {
    const providers = ['webgpu', 'webgl', 'wasm'];

    for (const provider of providers) {
        try {
            const session = await ort.InferenceSession.create(
                '/models/best_nms.onnx',
                { executionProviders: [provider] }
            );

            console.log(`Модель загружена с ${provider}`);
            return session;
        } catch (err: any) {
            console.warn(`${provider} не сработал:`, err.message);
        }
    }

    throw new Error('Не удалось загрузить модель ни с одним провайдером');
}