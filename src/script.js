import './style.css'
import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

// Wacht totdat de DOM volledig geladen is
document.addEventListener("DOMContentLoaded", function() {
    const enableWebcamButton = document.getElementById("webcamButton")
    const checkPoseButton = document.getElementById("checkAnwser")
    const resultBox = document.getElementById("predictionResult");

    const video = document.getElementById("webcam")
    const canvasElement = document.getElementById("output_canvas")
    const canvasCtx = canvasElement.getContext("2d")

    const drawUtils = new DrawingUtils(canvasCtx)
    let handLandmarker = undefined;
    let webcamRunning = false;
    let results = undefined;

    let image = document.querySelector("#myimage")



    ml5.setBackend("webgl");
    const nn = ml5.neuralNetwork({ task: 'classification', debug: true })
    const modelDetails = {
        model: 'model/model.json',
        metadata: 'model/model_meta.json',
        weights: 'model/model.weights.bin'
    }
    nn.load(modelDetails, () => {
        console.log("het model is geladen!");
        showNextSum();
    });
//functie die data in array pusht
    function liveHandData(handLandmarks) {
        let handData = [];
        for (let hand of handLandmarks) {
            for (let landmark of hand) {
                handData.push(landmark.x);
                handData.push(landmark.y);
                handData.push(landmark.z);
            }
        }
        return handData;
    }

    const gestureToValue = {
        "🤟": 10,
        "🖖": 2,
        "👎": 20
    };

    const valueToGesture = {
        10: "🤟",
        2: "🖖",
        20: "👎"
    };

    const sums = [
        { question: "🤟 + 🤟", answer: "👎" },      // 10 + 10 = 20
        { question: "🖖 x 🤟", answer: "👎" },      // 2 x 5 = 10
        { question: "👎 - 🤟", answer: "🤟" },      // 20 - 10 = 10
        { question: "👎 : 🤟", answer: "🖖" },      // 20 : 10 = 10
        { question: "🤟 x 🖖", answer: "👎" },      // 10 x 2 = 20
        { question: "👎 : 🤟", answer: "🖖" },      // 20 : 10 = 2
    ];

    let currentQuestion = null;
    const allSums = sums.filter(s => gestureToValue[s.answer]);

    function showNextSum() {
        const index = Math.floor(Math.random() * allSums.length);
        currentQuestion = allSums[index];
        document.getElementById("sumDisplay").innerText = currentQuestion.question + " = ?";
    }

    async function posePrediction() {
        if (!results || !results.landmarks || results.landmarks.length === 0) {
            console.warn("Geen handlandmarks beschikbaar!");
            return;
        }

        const input = liveHandData([results.landmarks[0]]);
        const prediction = await nn.classify(input);
        const label = prediction[0].label;
        const confidence = prediction[0].confidence;

        console.log(`Gebaar herkend als: ${label} met een nauwkeurigheid van ${(confidence * 100).toFixed(2)}%`);

        if (label === currentQuestion.answer) {
            resultBox.innerText = "✅ Goed zo!";
            setTimeout(() => {
                resultBox.innerText = "";
                showNextSum();
            }, 1500);
        } else {
            resultBox.innerText = "❌ Probeer opnieuw!";
            setTimeout(() => {
                resultBox.innerText = "";
            }, 1500);
        }
    }





    // create pose detector
    const createHandLandmarker = async () => {
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 2
        });
        console.log("model loaded, you can start webcam");

        enableWebcamButton.addEventListener("click", (e) => enableCam(e));
        checkPoseButton.addEventListener("click", (e) => posePrediction(e));
    }

    // start the webcam
    async function enableCam() {
        webcamRunning = true;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            video.srcObject = stream;
            video.addEventListener("loadeddata", () => {
                canvasElement.style.width = video.videoWidth;
                canvasElement.style.height = video.videoHeight;
                canvasElement.width = video.videoWidth;
                canvasElement.height = video.videoHeight;
                document.querySelector(".videoView").style.height = video.videoHeight + "px";
                predictWebcam();
            });
        } catch (error) {
            console.error("Error accessing webcam:", error);
        }
    }

    // START PREDICTIONS
    async function predictWebcam() {
        results = await handLandmarker.detectForVideo(video, performance.now());

        let hand = results.landmarks[0];
        if (hand) {
            let thumb = hand[4];
            image.style.transform = `translate(${video.videoWidth - thumb.x * video.videoWidth}px, ${thumb.y * video.videoHeight}px)`;
        }

        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        for (let hand of results.landmarks) {
            drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5, });
            drawUtils.drawLandmarks(hand, { radius: 4, color: "#FF0000", lineWidth: 2 });
        }

        if (webcamRunning) {
            window.requestAnimationFrame(predictWebcam);
        }
    }

    // Start the app
    if (navigator.mediaDevices?.getUserMedia) {
        createHandLandmarker();
    }
});
