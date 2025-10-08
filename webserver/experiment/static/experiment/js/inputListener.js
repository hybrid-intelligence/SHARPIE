// If a key is pressed
document.addEventListener('keydown', (event)=> {
    inputsListened.forEach(input => {
        if (event.key === input && !inputsForwarded.includes(event.key)) {
            inputsForwarded.push(event.key)
        }
    });
});

// Reset controls when keys are released
document.addEventListener('keyup', (event)=> {
    const index = inputsForwarded.indexOf(event.key);
    if (index > -1) { // only splice array when item is found
        inputsForwarded.splice(index, 1); // 2nd parameter means remove one item only
    }
});