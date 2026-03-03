/* ── state ── */
let video, canvas, ctx, overlay, overlayCtx;
let stream = null;
let intervalId = null;
const INTERVAL_MS = 150;

/* ── prediction buffer ── */
const BUFFER_SIZE = 5;
const STABILITY_THRESHOLD = 3;
let predictionBuffer = [];
let lastStableLetter = "";
let sentence = "";

/* ── element refs ── */
const $ = (id) => document.getElementById(id);

/* ── init ── */
window.addEventListener("DOMContentLoaded", () => {
    video = $("video");
    overlay = $("overlay");
    overlayCtx = overlay.getContext("2d");

    // off-screen canvas for capture
    canvas = document.createElement("canvas");
    ctx = canvas.getContext("2d");

    $("startBtn").addEventListener("click", startCamera);
    $("stopBtn").addEventListener("click", stopCamera);
    $("spaceBtn").addEventListener("click", addSpace);
    $("backspaceBtn").addEventListener("click", backspace);
    $("clearBtn").addEventListener("click", clearSentence);
    $("speakBtn").addEventListener("click", speak);
});

/* ── camera ── */
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: 640, height: 480 },
        });
        video.srcObject = stream;
        await video.play();

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        overlay.width = canvas.width;
        overlay.height = canvas.height;

        if (intervalId) clearInterval(intervalId);
        intervalId = setInterval(captureFrame, INTERVAL_MS);
    } catch (err) {
        console.error("Camera error:", err);
        alert("Could not access the camera. Please allow camera permissions.");
    }
}

function stopCamera() {
    if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
    }
    if (stream) {
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
    }
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    $("letter").textContent = "-";
    $("confidence").textContent = "Confidence: --";
    $("top3").innerHTML = "";
}

/* ── capture & predict ── */
async function captureFrame() {
    if (!video || video.readyState < 2) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.8);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataUrl }),
        });
        const data = await response.json();
        handlePrediction(data);
    } catch (err) {
        console.error("Prediction error:", err);
    }
}

/* ── handle response ── */
function handlePrediction(data) {
    const letterEl = $("letter");
    const confEl = $("confidence");
    const top3El = $("top3");

    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

    if (!data.hand_detected) {
        letterEl.textContent = "-";
        confEl.textContent = "Confidence: --";
        top3El.innerHTML = "";
        predictionBuffer = [];
        return;
    }

    // main prediction
    letterEl.textContent = data.label;
    confEl.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

    // top 3
    if (data.top3) {
        top3El.innerHTML = data.top3
            .map(
                (p) =>
                    `<span class="chip">${p.label} ${(p.confidence * 100).toFixed(
                        0
                    )}%</span>`
            )
            .join(" ");
    }

    // bounding box
    if (data.bbox) {
        const [x1, y1, x2, y2] = data.bbox;
        overlayCtx.strokeStyle = "#e67e22";
        overlayCtx.lineWidth = 3;
        overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        overlayCtx.fillStyle = "#e67e22";
        overlayCtx.font = "bold 18px sans-serif";
        overlayCtx.fillText(
            `${data.label} ${(data.confidence * 100).toFixed(0)}%`,
            x1,
            y1 - 6
        );
    }

    // stability buffer
    predictionBuffer.push(data.label);
    if (predictionBuffer.length > BUFFER_SIZE) predictionBuffer.shift();

    if (predictionBuffer.length >= BUFFER_SIZE && $("autoAppend").checked) {
        const counts = {};
        predictionBuffer.forEach((l) => (counts[l] = (counts[l] || 0) + 1));
        const best = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
        if (best[1] >= STABILITY_THRESHOLD && best[0] !== lastStableLetter) {
            lastStableLetter = best[0];
            sentence += best[0];
            $("sentence").textContent = sentence;
            predictionBuffer = [];
        }
    }
}

/* ── text controls ── */
function addSpace() {
    sentence += " ";
    $("sentence").textContent = sentence;
    lastStableLetter = "";
}

function backspace() {
    sentence = sentence.slice(0, -1);
    $("sentence").textContent = sentence;
    lastStableLetter = "";
}

function clearSentence() {
    sentence = "";
    $("sentence").textContent = "";
    lastStableLetter = "";
    predictionBuffer = [];
}

async function speak() {
    if (!sentence) return;
    const lang = document.getElementById("langSelect").value;
    const btn = document.getElementById("speakBtn");
    btn.textContent = "Speaking...";
    btn.disabled = true;

    try {
        const res = await fetch("/speak", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: sentence, lang: lang }),
        });
        const data = await res.json();
        if (data.audio) {
            const audio = new Audio(data.audio);
            audio.onended = () => { btn.textContent = "Speak"; btn.disabled = false; };
            audio.onerror = () => { btn.textContent = "Speak"; btn.disabled = false; };
            audio.play();
        } else {
            btn.textContent = "Speak";
            btn.disabled = false;
        }
    } catch (err) {
        console.error("Speak failed:", err);
        btn.textContent = "Speak";
        btn.disabled = false;
    }
}
