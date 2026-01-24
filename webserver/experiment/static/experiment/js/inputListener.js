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
    feedbackElement.classList.remove('show', 'reward-positive', 'reward-negative', 'action');

    if (experiment_type === 'reward') {
        // Reward-based: Up = +1, Down = -1
        if (key === 'ArrowUp') {
            feedbackElement.textContent = '+1';
            feedbackElement.classList.add('reward-positive');
        } else if (key === 'ArrowDown') {
            feedbackElement.textContent = '-1';
            feedbackElement.classList.add('reward-negative');
        }
    } else {
        // Action-based: show key name
        var keyNames = {
            'ArrowLeft': 'Left',
            'ArrowRight': 'Right',
            'ArrowUp': 'Up',
            'ArrowDown': 'Down'
        };
        feedbackElement.textContent = keyNames[key] || key;
        feedbackElement.classList.add('action');
    }

    feedbackElement.classList.add('show');

    // Auto-hide feedback after delay
    setTimeout(function() {
        feedbackElement.classList.remove('show');
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
    for (var i = 0; i < inputsListened.length; i++) {
        var input = inputsListened[i];
        if (event.key === input && inputsForwarded.indexOf(event.key) === -1) {
            inputsForwarded.push(event.key);
            showKeyPressed(event.key);
            showFeedback(event.key);
        }
    }
});

// Only for action-based experiments. For reward-based experiments, the keys are reset on message sent
if (experiment_type === 'action') {
    // Reset controls when keys are released
    document.addEventListener('keyup', function(event) {
        var index = inputsForwarded.indexOf(event.key);
        if (index > -1) {
            inputsForwarded.splice(index, 1);
            hideKeyPressed(event.key);
        }
    });
}