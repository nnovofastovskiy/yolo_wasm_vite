import * as ort from "onnxruntime-web";
ort.env.wasm.wasmPaths = "./";
export async function loadModelWithFallback(statusEl: HTMLParagraphElement) {
    const providers = ['webgl', 'webgpu', 'wasm'];
    const modelUrl = './models/best_nms.onnx';
    const response = await fetch(modelUrl);
    if (!response.ok) {
        throw new Error(`Не удалось загрузить модель по адресу ${modelUrl}: ${response.statusText}`);
    }

    const totalSize = Number(response.headers.get('Content-Length') ?? 0);
    console.log(totalSize);

    let downloadedSize = 0;

    const reader = response.body?.getReader();
    const chunks: Uint8Array[] = [];

    while (true) {
        const { done, value } = await (reader?.read() ?? Promise.resolve({ done: true, value: undefined }));
        if (done) break;

        chunks.push(value!);
        downloadedSize += value!.length;

        if (totalSize > 0) {
            const percent = ((downloadedSize / totalSize) * 100).toFixed(2);
            statusEl.innerText = `Загрузка модели... ${percent}%`;
        }
    }

    const modelData = concatenateChunks(chunks);

    for (const provider of providers) {
        try {
            const session = await ort.InferenceSession.create(
                modelData,
                { executionProviders: [provider] }
            );
            statusEl.innerText += `\nМодель загружена с ${provider}`;
            console.log(`Модель загружена с ${provider}`);
            return session;
        } catch (err: any) {
            statusEl.innerText += `\n${provider} не сработал:`;
            console.warn(`${provider} не сработал:`, err.message);
        }
    }

    throw new Error('Не удалось загрузить модель ни с одним провайдером');
}

function concatenateChunks(chunks: Uint8Array[]): Uint8Array {
    let total = 0;
    for (const c of chunks) total += c.length;

    const out = new Uint8Array(total);
    let offset = 0;
    for (const c of chunks) {
        out.set(c, offset);
        offset += c.length;
    }
    return out;
}