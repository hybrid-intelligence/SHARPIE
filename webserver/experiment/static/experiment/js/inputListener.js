// Feedback CSS class constants
const FeedbackClass = Object.freeze({
    SHOW: 'show',
    ACTION: 'action'
});

const ALL_FEEDBACK_CLASSES = Object.freeze(Object.values(FeedbackClass));

// Key display uses keyDisplayMap from backend (config-driven, no hardcoding)

// Visual feedback functions
function showKeyPressed(key) {
    const keyElement = document.querySelector('.keyboard-key[data-key="' + key + '"]');
    if (keyElement) {
        keyElement.classList.add('pressed');
    }
}

function hideKeyPressed(key) {
    const keyElement = document.querySelector('.keyboard-key[data-key="' + key + '"]');
    if (keyElement) {
        keyElement.classList.remove('pressed');
    }
}

function showFeedback(key) {
    const feedbackElement = document.querySelector('.key-feedback[data-feedback="' + key + '"]');
    if (!feedbackElement) return;

    // Clear existing classes
    feedbackElement.classList.remove(...ALL_FEEDBACK_CLASSES);

    // Unified feedback: show label from config (works for both action and reward experiments)
    var displayLabel = key;
    if (typeof keyDisplayMap !== 'undefined' && keyDisplayMap && keyDisplayMap[key] && keyDisplayMap[key].label) {
        displayLabel = keyDisplayMap[key].label;
    } else if (key === ' ') {
        displayLabel = 'Space';
    }
    feedbackElement.textContent = displayLabel;
    feedbackElement.classList.add(FeedbackClass.ACTION);

    feedbackElement.classList.add(FeedbackClass.SHOW);

    // Auto-hide feedback after delay
    setTimeout(function() {
        feedbackElement.classList.remove(FeedbackClass.SHOW);
    }, 500);
}

// Clear all visual feedback (used when inputs are cleared)
function clearAllVisualFeedback() {
    var elements = document.querySelectorAll('.keyboard-key.pressed');
    for (var i = 0; i < elements.length; i++) {
        elements[i].classList.remove('pressed');
    }
}

// If a key is pressed
document.addEventListener('keydown', function(event) {
    // If the agent does not allow multiple keybord inputs
    if(!multipleInputs){
        inputsForwarded = [];
    }
    for (var i = 0; i < inputsListened.length; i++) {
        var input = inputsListened[i];
        if (event.key === input && inputsForwarded.indexOf(event.key) === -1) {
            inputsForwarded.push(event.key);
            showKeyPressed(event.key);
            showFeedback(event.key);
        }
    }
});

// If a key is up we remove it
document.addEventListener('keyup', function(event) {
    var index = inputsForwarded.indexOf(event.key);
    if (index > -1) {
        inputsForwarded.splice(index, 1);
        hideKeyPressed(event.key);
    }
});