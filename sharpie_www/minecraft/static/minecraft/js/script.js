// Connecting to the server's websocket
const chatSocket = new WebSocket(
    'ws://'
    + window.location.host
    + window.location.pathname
);



// Flag used to avoid querying the server to fast
isInUse = false;
// Setting up an infinite loop to perform 'queryLoop', here every 41 milliseconds (~ 24 frames/second)
refreshRate = 41;
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
            'nothing': ''
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
    document.getElementById("image").style.visibility = 'visible';
    document.getElementById("loading_div").style.visibility = 'hidden';
    // Parse the message from the server
    const data = JSON.parse(e.data);
    document.getElementById("sub-title").innerText = "Step " + data.step;
    // If the game is over, we stop the infinite loop
    if(data.message == 'done'){
        clearInterval(interval);
        console.log("Game over");
        document.getElementById("sub-title").innerHTML = document.getElementById("sub-title").innerText + " (game over) " + restart_button;
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