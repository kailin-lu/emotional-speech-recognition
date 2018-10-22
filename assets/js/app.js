/*
 AUDIO RECORDING CODE REFERENCES
 https://developer.mozilla.org/en-US/docs/Web/API/MediaStream_Recording_API/Using_the_MediaStream_Recording_API
 https://developers.google.com/web/fundamentals/media/recording-audio/
 https://github.com/addpipe/Media-Recorder-API-Demo/blob/master/js/main.js
 https://github.com/mdn/web-dictaphone
*/

navigator.getUserMeia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;

var constraints = { audio: true }

var record = document.querySelector('button#recordButton');
var stop= document.querySelector('button#stopButton');
var canvas = document.getElementById('visualizer');
var startDrawing = true;
WIDTH = canvas.width;
HEIGHT = canvas.height;

var audioCtx = new (window.AudioContext || websiteAudioContext)();
var canvasCtx = canvas.getContext('2d');

function errorCallback(error){
	console.log('navigator.getUserMedia error: ', error);
}

var mediaRecorder;
var chunks = [];
var count = 0;

function startRecording(stream) {
    mediaRecorder = new MediaRecorder(stream);
    visualize(stream);
	mediaRecorder.start();

	console.log(mediaRecorder.state);
	console.log('Recorder started');

	var url = window.URL || window.webkitURL;

	mediaRecorder.ondataavailable = function(e) {
		chunks.push(e.data);
	};

	mediaRecorder.onerror = function(e){
		console.log('Error: ', e);
	};

	mediaRecorder.onstop = function(){
        console.log('Recording Stopped');
		var blob = new Blob(chunks, {type: "audio/webm"});
		chunks = [];

        var audio = document.createElement('audio');
        audio.setAttribute('controls', '');
        var audioURL = window.URL.createObjectURL(blob);
        audio.src = audioURL;
        audio.play()
        stream.getTracks()
        .forEach( track => track.stop() );

	};
}

function onRecordClicked (){
    startDrawing = true;
    navigator.getUserMedia(constraints, startRecording, errorCallback);
    record.disabled = true;
    stop.disabled = false;
}


function onStopClicked(){
	mediaRecorder.stop();
	startDrawing = false;
    console.log(mediaRecorder.state);
	record.disabled = false;
	stop.disabled = true;
}


function visualize(stream) {

    if (startDrawing) {
        var source = audioCtx.createMediaStreamSource(stream);
        var analyser = audioCtx.createAnalyser();

        analyser.fftSize = 2048;
        var bufferLength = analyser.frequencyBinCount;
        var dataArray = new Uint8Array(bufferLength);

        source.connect(analyser);
        draw();

        function draw() {
            var drawVisual = requestAnimationFrame(draw);

            analyser.getByteTimeDomainData(dataArray);

            canvasCtx.clearRect(0, 0, WIDTH, HEIGHT);

            canvasCtx.lineWidth = 1;
            canvasCtx.strokeStyle = 'rgb(77, 129, 170)';
            canvasCtx.beginPath();

            var sliceWidth = WIDTH * 1.0 / bufferLength;
            var x = 0;
            for(var i = 0; i < bufferLength; i++) {

            var v = dataArray[i] / 128.0;
            var y = v * HEIGHT/2;

            if(i === 0) {
              canvasCtx.moveTo(x, y);
            } else {
              canvasCtx.lineTo(x, y);
            }

            x += sliceWidth;
          }
           canvasCtx.lineTo(canvas.width, canvas.height/2);
           canvasCtx.stroke();
        };
    }
}


/*
Tensorflow model
*/

const WEIGHTS_URL = 'https://storage.cloud.google.com/klu-models/web_prediction_model/weights_manifest.json';
const MODEL_URL = 'https://storage.cloud.google.com/klu-models/web_prediction_model/tensorflowjs_model.pb';

const model = tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);



