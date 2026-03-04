// Connecting to the server's websocket
export const websocket = new WebSocket(
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

// Web Worker for async image decoding (disabled - using data URL directly for reliability)
// The data URL approach is simpler and works across all browsers without COOP issues

function decodeImageSync(image){
    // Use data URL directly - browser handles decoding natively
    return 'data:image/jpeg;base64,' + image;
}

function decodeImage(image){
    // Use data URL directly - simple and reliable
    if (!image) {
        console.error("decodeImage: image is undefined or null");
        return;
    }
    const startTime = performance.now();
    const url = decodeImageSync(image);
    document.getElementById("image").src = url;
    document.getElementById("image").style.display = "block";
    document.getElementById("loading_div").style.display = "none";
    console.log(`Image set (data URL), size: ${(image.length/1024).toFixed(1)}KB, elapsed: ${(performance.now()-startTime).toFixed(1)}ms`);
}

function inputsMappingFunction(inputs){
    let result = [];
    for(let i in inputs){
        result.push(inputsMapping[inputs[i]]);
    }
    if(result.length == 0){
        result.push(inputsMapping['default']);
    }
    if(multipleInputs)
        return result;
    else
        return result[0];
}

function sendAction(action) {
    websocket.send(JSON.stringify({type: 'broadcast', action: action}));
}

// When the server finishes a step and replies
websocket.onmessage = function(e) {
    const data = JSON.parse(e.data);

    // Stop the experiment if there is an error
    if(data.error){
        document.getElementById("image").style.display = "none";
        document.getElementById("loading_div").style.display = "block";
        var config_button = '<a href="config" class="btn btn-sm btn-outline-primary"><i class="bi bi-bootstrap-reboot"></i> configuration </a>';
        document.getElementById("loading_div").innerHTML = "<h3>Error: "+data.error+"</h3><br><h4>Go to back the " + config_button + " and try again</h4>";
        websocket.close();
    }

    // Debug: log received data keys
    if (!data.image) {
        console.error("No image in message. Keys:", Object.keys(data));
        return;
    }

    // Decode and display image
    decodeImage(data.image);

    // Logging what is the actual frame rate on the browser
    let currentTime = Date.now();
    averageFPS.push(parseInt(1000.0 / (currentTime - startTime)));
    startTime = Date.now();

    // Showing detailled info
    document.getElementById("details").innerHTML += "<li>Step "+data.step;

    // We send back the inputs
    // If waitForInputs is True, wait until a non-default action is provided
    const action = inputsMappingFunction(inputsForwarded);
    const defaultAction = inputsMapping['default'];

    if (waitForInputs) {
        // Show waiting indicator
        document.getElementById("waiting_for_input").style.display = "block";
        // Wait for user to provide a non-default action
        const checkInterval = setInterval(() => {
            const currentAction = inputsMappingFunction(inputsForwarded);
            if (currentAction !== defaultAction) {
                // Send the action
                sendAction(currentAction);
                // Hide waiting indicator
                document.getElementById("waiting_for_input").style.display = "none";
                // Clear the interval and input forward list
                clearInterval(checkInterval);
                clearAllVisualFeedback();
                inputsForwarded = [];
            }
        }, 100); // Check every 100ms
    } else {
        // Send action immediately
        sendAction(action);
    }

    // If the game is over
    if(data.terminated || data.truncated){
        console.log("Game over, average FPS: ", average(averageFPS));
        if(data.redirect){
            // Wait and redirect to the experiment page
            document.getElementById("image").style.display = "none";
            document.getElementById("loading_div").style.display = "block";
            document.getElementById("loading_div").innerHTML = "<h3>Thanks for participating! You will now be redirected to complete the experiment.</h3>";
            setTimeout(function() {
                window.location.href = data.redirect;
            }, 5000);
        }
        else if(data.completed){
            document.getElementById("image").style.display = "none";
            document.getElementById("loading_div").style.display = "block";
            document.getElementById("loading_div").innerHTML = "<h3>Thanks for participating! This experiment is completed.</h3>";
        }
        else {
            // Replace the subtitle text by adding "game over" and a restart button
            document.getElementById("image").style.display = "none";
            document.getElementById("loading_div").style.display = "block";
            var restart_button = '<a href="' + window.location.href + '" class="btn btn-sm btn-outline-primary"><i class="bi bi-bootstrap-reboot"></i> Restart</a>';
            document.getElementById("loading_div").innerHTML = "<h3>Episode finished!</h3><br><h4>Ready to " + restart_button + " ?</h4>";
        }
        websocket.close();
    }
};




// When the server closes the connection
websocket.onclose = function(e) {
    console.log('Websocket closed by the server');
};