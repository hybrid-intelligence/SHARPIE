// Capturing the inputs from the user
var left = 0;
var right = 0;
var up = 0;
var down = 0;
// If a key is pressed
document.addEventListener('keydown', (event)=> {
    if(event.key == 'ArrowLeft'){
        left = 1;
        right = 0;
        up = 0;
        down = 0;
    } else if(event.key == 'ArrowRight') {
        left = 0;
        right = 1;
        up = 0;
        down = 0;
    } else if(event.key == 'ArrowDown') {
        left = 0;
        right = 0;
        up = 0;
        down = 1;
    } else if(event.key == 'ArrowUp') {
        left = 0;
        right = 0;
        up = 1;
        down = 0;
    } else if(event.key == ' ') {
        left = 0;
        right = 0;
        up = 0;
        down = 0;
    }
});





// Connecting to the server's websocket
const chatSocket = new WebSocket(
    'ws://'
    + window.location.host
    + window.location.pathname
);



// Flag used to avoid querying the server to fast
isInUse = false;
// Setting up an infinite loop to perform 'queryLoop', here every 41 milliseconds (~ 24 frames/second)
refreshRate = 150;
interval = setInterval(queryLoop, refreshRate);
// Asking the server to perform a step and sending the user's inputs
function queryLoop() {
    try {
        // Send the user's inputs and set isInUse
        chatSocket.send(JSON.stringify({
            'left': left,
            'right': right,
            'down': down,
            'up': up
        }));
        isInUse = true;
    }
    catch(err){
        // If the websocket is still connecting, we catch the error and wait another round
        if(err.message == "Failed to execute 'send' on 'WebSocket': Still in CONNECTING state."){
            console.log('Waiting for the websocket to connect');
        }
        else{
            console.error(err.message);
        }
    }
};




// When the server finishes a step and replies
chatSocket.onmessage = function(e) {
    // We refresh the image on the page by taking the new one from the server
    // "?"+new Date().getTime() is used here to force the browser to re-download the image and not use a cached version
    document.getElementById("image").src = image_scr+"?"+new Date().getTime();
    // Focus on the image
    document.getElementById("image").focus();
    // Parse the message from the server
    const data = JSON.parse(e.data);
    document.getElementById("sub-title").innerText = "Step " + data.step;
    // If the game is over, we stop the infinite loop
    if(data.message == 'done'){
        clearInterval(interval);
        console.log("Game over");
        document.getElementById("sub-title").innerHTML = document.getElementById("sub-title").innerText + " (game over) " + restart_button;
    } else {
        clearInterval(interval);
	interval = setInterval(queryLoop, refreshRate);
    }
    // Unset isInUse 
    isInUse = false;
};




// When the server closes the connection
chatSocket.onclose = function(e) {
    // We stop the infinite loop
    clearInterval(interval);
    console.error('Chat socket closed unexpectedly');
};