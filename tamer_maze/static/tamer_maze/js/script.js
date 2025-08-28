// Capturing the inputs from the user
var left = 0;
var right = 0;
var up = 0;
var down = 0;
var button_d = 0;
let keyPressed = {};
// If a key is pressed
document.addEventListener('keydown', (event)=> {
    if(!keyPressed[event.code]){
        keyPressed[event.code] = true;
        //send(event.code);

    // print destroys everything, spent hours after hours trying to find the reason of the bug
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
    } else if(event.key == 'D' || event.key == 'd') {
        button_d = !button_d;
    } else if(event.key == ' ') {
        left = 0;
        right = 0;
        up = 0;
        down = 0;
    }
    }
});


document.addEventListener("keyup", (event) => {
    keyPressed[event.code] = false;
        left = 0;
        right = 0;
        up = 0;
        down = 0;
});



function sendFeedback(value) {
    if (chatSocket.readyState === WebSocket.OPEN) {
        chatSocket.send(JSON.stringify({
            'feedback': value
        }));
    }
}


// Flag used to avoid querying the server to fast if it did not yet reply
isInUse = false;
// Asking the server to perform a step and sending the user's inputs
function queryLoop() {
    console.log("query loop i")
    // If we already asked the server but it still did not reply, we skip and wait another round
    if(isInUse){
        return;
    }

    try {

        // Send the user's inputs and set isInUse
        chatSocket.send(JSON.stringify({
            'action': 'next'
        }));
        // Set isInUse 
        isInUse = true;
    }
    catch(err){
        // If the websocket is still connecting, we catch the error and wait another round
        if(err.message == "Failed to execute 'send' on 'WebSocket': Still in CONNECTING state."){
            console.log('Waiting for the websocket to connect');
        }
        // Else we repport the error to the browser console
        else{
            console.error(err.message);
        }
    }
};


// Connecting to the server's websocket
const chatSocket = new WebSocket(
    ws_setting
    + '://'
    + window.location.host
    + window.location.pathname
);



// Flag used to avoid querying the server to fast if it did not yet reply
isInUse = false;
// Setting up an infinite loop to perform 'queryLoop'
refreshRate = 150;
interval = setInterval(queryLoop, refreshRate);
// Asking the server to perform a step and sending the user's inputs
function queryLoop() {
    // If we already asked the server but it still did not reply, we skip and wait another round
    if(isInUse){
        return;
    }

    try {
        // Send the user's inputs and set isInUse
        chatSocket.send(JSON.stringify({
            'left': left,
            'right': right,
            'down': down,
            'up': up,
            'button_d' : button_d
        }));
        // Set isInUse 
        isInUse = true;
    }
    catch(err){
        // If the websocket is still connecting, we catch the error and wait another round
        if(err.message == "Failed to execute 'send' on 'WebSocket': Still in CONNECTING state."){
            console.log('Waiting for the websocket to connect');
        }
        // Else we repport the error to the browser console
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
    // We set the image visible and hide the loading icon
    document.getElementById("image").style.visibility = 'visible';
    document.getElementById("loading_div").style.visibility = 'hidden';
    // Parse the message from the server
    const data = JSON.parse(e.data);
    // Replace the subtitle text by the new step number received
    document.getElementById("sub-title").innerText = "Step " + data.step;
    // If the game is over
    if(data.message == 'done'){
        // Stop the infinite loop
        clearInterval(interval);
        console.log("Game over");
        // Replace the subtitle text by adding "game over" and a restart button
        document.getElementById("sub-title").innerHTML = document.getElementById("sub-title").innerText + " (game over) " + restart_button;
    } 
    // Unset isInUse 
    isInUse = false;
};




// When the server closes the connection
chatSocket.onclose = function(e) {
    // We stop the infinite loop
    clearInterval(interval);
    console.log('Websocket closed by the server');
};