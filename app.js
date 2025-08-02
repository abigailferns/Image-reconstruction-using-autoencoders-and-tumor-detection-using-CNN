document.addEventListener("DOMContentLoaded", function () {
    const imageInput = document.getElementById("imageInput");
    const resultsSection = document.getElementById("resultsSection");
    const loadingIndicator = document.getElementById("loading");

    window.processImage = function (organ) {
        const file = imageInput.files[0];
        if (!file) {
            alert("Please select an image first!");
            return;
        }

        const formData = new FormData();
        formData.append("image", file);

        loadingIndicator.classList.remove("hidden");
        resultsSection.classList.add("hidden");

        fetch(`/process/${organ}`, {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingIndicator.classList.add("hidden");

            document.getElementById("originalImage").src = data.original;
            document.getElementById("originalImage").classList.remove("hidden");

            document.getElementById("denoisedImage").src = data.denoised;
            document.getElementById("denoisedImage").classList.remove("hidden");

            const predictionBox = document.getElementById("predictionBox");
            predictionBox.innerText = `Predicted: ${data.prediction}`;
            predictionBox.classList.remove("hidden");

            resultsSection.classList.remove("hidden");
        })
        .catch(error => {
            loadingIndicator.classList.add("hidden");
            alert("Error processing image!");
            console.error(error);
        });
    };
});