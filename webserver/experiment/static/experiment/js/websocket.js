// Connecting to the server's websocket
const websocket = new WebSocket(
    ws_setting
    + '://'
    + window.location.host
    + window.location.pathname
);




websocket.onopen = function(e) {
    console.log("WebSocket connection established.");
};


websocket.onerror = function(e) {
    console.log("WebSocket connection error.");
    console.log(e);
}



var startTime = Date.now();
var averageFPS = [];
function average(arr) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
    }
    return sum / arr.length;
}

function decodeImage(image){
    // Decode base64 image bytes
    const byteCharacters = atob(image);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'image/jpeg' });
    return URL.createObjectURL(blob);
}

// When the server finishes a step and replies
websocket.onmessage = function(e) {
    const data = JSON.parse(e.data);

    if(data.error !== undefined){
        document.getElementById("sub-title").innerText = "Error: " + data.error;
        document.getElementById("sub-title").style.color = 'red';
        document.getElementById("loading_div").style.visibility = 'hidden';
        return;
    }
    if(data.message !== undefined){
        document.getElementById("sub-title").innerText = data.message;
        document.getElementById("sub-title").style.color = 'red';
        document.getElementById("loading_div").style.visibility = 'hidden';
        return;
    }

    const image_scr = decodeImage(data.image);

    document.getElementById("image").src = image_scr;
    // We set the image visible and hide the loading icon
    document.getElementById("image").style.visibility = 'visible';
    document.getElementById("loading_div").style.visibility = 'hidden';
    // Replace the subtitle text by the new step number received
    document.getElementById("sub-title").innerText = "Step " + data.step;

    // Logging what is the actual frame rate on the browser
    let currentTime = Date.now();
    averageFPS.push(parseInt(1000.0 / (currentTime - startTime)));
    startTime = Date.now();

    // We send back the inputs
    websocket.send(JSON.stringify({actions: inputsForwarded}))
    console.log(inputsForwarded);
    // For reward-based experiments, we clear the inputs after sending them
    if (experiment_type === 'reward') {
        inputsForwarded = [];
    }

    // If the game is over
    if(data.terminated){
        console.log("Game over, average FPS: ", average(averageFPS));
        // Replace the subtitle text by adding "game over" and a restart button
        restart_button = '<a href="' + window.location.href + '" class="btn btn-primary"><i class="bi bi-bootstrap-reboot"></i> Restart</a>';
        document.getElementById("sub-title").innerHTML = document.getElementById("sub-title").innerText + " (game over) " + restart_button;
        // Adding an evaluate button
        if (experiment_train === 'True')
            document.getElementById("evaluate-col").hidden = false;
        websocket.close();
    } 
};




// When the server closes the connection
websocket.onclose = function(e) {
    console.log('Websocket closed by the server');
};