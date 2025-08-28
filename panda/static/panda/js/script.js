var left = 0;
var right = 0;
var up = 0;
var down = 0;
var button_d = 0;
var z_up = 0;
var z_down = 0;
var button_q = 1.0;

let keyPressed = {};

function resetMovementFlags() {
    left = 0;
    right = 0;
    up = 0;
    down = 0;
    z_up = 0;
    z_down = 0;
}

document.addEventListener('keydown', (event)=> {
    if(!keyPressed[event.code]){
        keyPressed[event.code] = true;
        
        switch(event.code) {
            case 'ArrowLeft':
                //resetMovementFlags();
                left = 1;
                right = 0;
                break;
            case 'ArrowRight':
                //resetMovementFlags();
                right = 1;
                left = 0;
                break;
            case 'ArrowDown':
                //resetMovementFlags();
                down = 1;
                break;
            case 'ArrowUp':
                //resetMovementFlags();
                up = 1;
                down = 0;
                break;
            case 'KeyX':
                //resetMovementFlags();
                z_down = 1;
                z_up = 0;
                break;
            case 'KeyW':
                //resetMovementFlags();
                z_up = 1;
                z_down = 0;
                break;
            case 'KeyD':
                button_d = !button_d;
                break;
            case 'KeyQ':
                button_q = (-1) * button_q;
                break;
            case 'Space':
                //resetMovementFlags();
                break;
        }
    }
});

document.addEventListener("keyup", (event) => {
    keyPressed[event.code] = false;
    resetMovementFlags();
});


        // added ---------------------------------------------------
        let isAgentFrozen = false;

        function toggleFreezeAgent() {
            isAgentFrozen = !isAgentFrozen;
            const statusElement = document.getElementById('agent-status');
            if (isAgentFrozen) {
                statusElement.style.display = 'block';
                statusElement.classList.add('blink');
                console.log("Agent Frozen. User can demonstrate behavior.");
            } else {
                statusElement.style.display = 'none';
                statusElement.classList.remove('blink');
                console.log("Agent Unfrozen. Resuming normal behavior.");
            }
            // Here you can trigger WebSocket or backend call to freeze/unfreeze agent logic.
        }

        // Handle 'D' key press
        document.addEventListener('keydown', function(event) {
            if (event.key === 'd' || event.key === 'D') {
                toggleFreezeAgent();
            }
        // Handle button click
        document.getElementById('freeze-btn').addEventListener('click', toggleFreezeAgent);

        });





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
            'button_d': button_d,
            'z_up': z_up,
            'z_down': z_down,
            'button_q': button_q
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
    document.getElementById("sub-title").innerHTML = `
    <p style="color: #4CAF50; font-weight: bold;">Step ${data.step}</p>
    <p style="color: #2196F3;">Environment: ${data.environment}</p>
    <p style="color: #FF9800;">Episode: ${data.episode}</p>
    `;

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