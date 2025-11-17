export function loadImageFromFile(): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
        const input = document.getElementById('upload-image') as HTMLInputElement;
        const inputImgWrapper = document.getElementById('input-img-wrapper') as HTMLDivElement;
        input.type = "file";
        input.accept = "image/*"; // только изображения

        input.onchange = () => {
            if (!input.files || input.files.length === 0) {
                reject(new Error("Файл не выбран"));
                return;
            }
            console.log('image change');

            const file = input.files[0];
            console.log(file);

            const url = URL.createObjectURL(file);

            const img = new Image();
            img.onload = () => {
                URL.revokeObjectURL(url); // освобождаем память
                if (!inputImgWrapper) {
                    reject(new Error(`Контейнер с id="${inputImgWrapper}" не найден`));
                    return;
                }

                // очищаем контейнер и вставляем изображение
                img.style.maxWidth = "100%";
                inputImgWrapper.innerHTML = "";
                inputImgWrapper.appendChild(img);

                resolve(img);
            };
            img.onerror = (err) => reject(err);
            img.src = url;
            console.log(url);

        };

        input.click();
    });
}