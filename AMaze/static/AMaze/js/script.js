// Connecting to the server's websocket
const chatSocket = new WebSocket(
    ws_setting
    + '://'
    + window.location.host
    + window.location.pathname
);

// Ensure loading div is visible on page load
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById("loading_div").style.visibility = 'visible';
    document.getElementById("image").style.visibility = 'hidden';
});

// Flag used to avoid querying the server to fast if it did not yet reply
isInUse = false;
// Asking the server to perform a step and sending the user's inputs
function queryLoop() {
    // If we already asked the server but it still did not reply, we skip and wait another round
    if(isInUse){
        return;
    }

    try {
        // Send the user's inputs and set isInUse
        chatSocket.send(JSON.stringify({
            'reward': document.getElementById("id_reward").value
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

function sendFeedback(value) {
    if (chatSocket.readyState === WebSocket.OPEN) {
        chatSocket.send(JSON.stringify({
            'feedback': value
        }));
    }
}

// When the server finishes a step and replies
chatSocket.onmessage = function(e) {
    // Parse the message from the server
    const data = JSON.parse(e.data);
    
    // Set the image source and make it visible only after we receive the first response
    document.getElementById("image").src = image_scr+"?"+new Date().getTime();
    document.getElementById("image").style.visibility = 'visible';
    document.getElementById("loading_div").style.visibility = 'hidden';
    
    // Replace the subtitle text by the new step number received
    document.getElementById("sub-title").innerText = "Step " + data.step;
    document.getElementById("id_reward").value = data.reward;
    
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