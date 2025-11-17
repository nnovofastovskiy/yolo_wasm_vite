import * as ort from "onnxruntime-web";

/**
 * Прогревает модель YOLO перед реальным инференсом.
 * @param session — созданная InferenceSession
 * @param inputName — имя входного тензора модели (обычно "images")
 * @param size — размер входа, например 640
 */
export async function warmupModel(
    session: ort.InferenceSession,
    inputName: string,
    size: number
): Promise<void> {
    // создаём пустой tензор (1,3,H,W)
    const dummy = new ort.Tensor("float32", new Float32Array(1 * 3 * size * size).map(_ => Math.random()), [1, 3, size, size]);
    console.log(dummy);

    const feeds: Record<string, ort.Tensor> = {};
    feeds[inputName] = dummy;

    // запускаем "холостой" прогон
    await session.run(feeds);

    console.log("🔥 Модель прогрета");
}