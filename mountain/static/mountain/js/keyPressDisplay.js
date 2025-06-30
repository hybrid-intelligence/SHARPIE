class KeyPressDisplay {
    constructor() {
        this.keyPressedText = document.getElementById('key-pressed-text');
        this.leftPressed = false;
        this.rightPressed = false;
        this.setupEventListeners();
    }
  
    setupEventListeners() {
        document.addEventListener('keydown', (event) => this.handleKeyEvent(event, true));
        document.addEventListener('keyup', (event) => this.handleKeyEvent(event, false));
    }
  
    handleKeyEvent(event, isKeyDown) {
        if (event.key === 'ArrowLeft') {
            this.leftPressed = isKeyDown;
        } else if (event.key === 'ArrowRight') {
            this.rightPressed = isKeyDown;
        }
        
        this.updateDisplay();
    }
  
    updateDisplay() {
        if (!this.keyPressedText) {
            console.error('Key press display elements not found');
            return;
        }
        if (this.leftPressed) {
            this.keyPressedText.textContent = '← Left';
        } else if (this.rightPressed) {
            this.keyPressedText.textContent = '→ Right';
        } else {
            this.keyPressedText.textContent = 'No key pressed';
        }
    }
  
    getControlState() {
        return {
            left: this.leftPressed ? 1 : 0,
            right: this.rightPressed ? 1 : 0
        };
    }

    setGameOver(isOver) {
        if (isOver) {
            this.leftPressed = false;
            this.rightPressed = false;
            this.updateDisplay();
        }
    }
}
  
document.addEventListener('DOMContentLoaded', () => {
    window.keyPressDisplay = new KeyPressDisplay();
}); 