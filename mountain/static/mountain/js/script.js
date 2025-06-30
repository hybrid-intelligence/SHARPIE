// Capturing the inputs from the user
var left = 0;
var right = 0;
// If a key is pressed
document.addEventListener('keydown', (event)=> {
    if(event.key == 'ArrowLeft'){
        left = 1;
        right = 0;
    } else if(event.key == 'ArrowRight') {
        left = 0;
        right = 1;
    } else if(event.key == 'ArrowDown') {
        left = 0;
        right = 0;
    }
});

// Reset controls when keys are released
document.addEventListener('keyup', (event)=> {
    if(event.key == 'ArrowLeft' || event.key == 'ArrowRight' || event.key == 'ArrowDown') {
        left = 0;
        right = 0;
    }
});

// Connecting to the server's websocket
const chatSocket = new WebSocket(
    ws_setting + '://' + window.location.host + window.location.pathname
);

// Flag used to avoid querying the server too fast if it did not yet reply
let isInUse = false;
// Setting up an infinite loop to perform 'queryLoop', here every 41 milliseconds (~ 24 frames/second)
const refreshRate = 41;
const interval = setInterval(queryLoop, refreshRate);

// Asking the server to perform a step and sending the user's inputs
function queryLoop() {
    if (isInUse) return;

    try {
        // Get control state from KeyPressDisplay
        const controls = window.keyPressDisplay.getControlState();
        
        // Send the user's inputs and set isInUse
        chatSocket.send(JSON.stringify(controls));
        // Set isInUse 
        isInUse = true;
    }
    catch(err) {
        // If the websocket is still connecting, we catch the error and wait another round
        if (err.message === "Failed to execute 'send' on 'WebSocket': Still in CONNECTING state.") {
            console.log('Waiting for the websocket to connect');
        }
        // Else we report the error to the browser console
        else {
            console.error(err.message);
        }
    }
}

// When the server finishes a step and replies
chatSocket.onmessage = function(e) {
    // We refresh the image on the page by taking the new one from the server
    // "?"+new Date().getTime() is used here to force the browser to re-download the image and not use a cached version
    document.getElementById("image").src = image_scr + "?" + new Date().getTime();
    // We set the image visible and hide the loading icon
    document.getElementById("image").style.visibility = 'visible';
    document.getElementById("loading_div").style.visibility = 'hidden';
    // Parse the message from the server
    const data = JSON.parse(e.data);
    // Replace the subtitle text by the new step number received
    document.getElementById("sub-title").innerText = "Step " + data.step;
    // If the game is over
    if (data.message === 'done') {
        // Stop the infinite loop
        clearInterval(interval);
        console.log("Game over");
        // Update the key press display
        window.keyPressDisplay.setGameOver(true);
        // Replace the subtitle text by adding "game over" and a restart button
        document.getElementById("sub-title").innerHTML = document.getElementById("sub-title").innerText + " (game over) " + restart_button;
    } 
    // Unset isInUse 
    isInUse = false;
};

// When the server closes the connection
chatSocket.onclose = function() {
    // We stop the infinite loop
    clearInterval(interval);
    console.log('Websocket closed by the server');
};
