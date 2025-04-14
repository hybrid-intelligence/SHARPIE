class KeyPressDisplay {
    constructor() {
      this.keyPressedBox = document.getElementById('key-pressed-box');
      this.keyPressedText = document.getElementById('key-pressed-text');
      this.isGameOver = false;
      this.setupEventListeners();
    }
  
    setupEventListeners() {
      document.addEventListener('keydown', (event) => this.handleKeyDown(event));
      document.addEventListener('keyup', (event) => this.handleKeyUp(event));
    }
  
    handleKeyDown(event) {
      if (this.isGameOver) return;
      if (event.key === 'ArrowLeft') {
        this.updateDisplay('← Left');
      } else if (event.key === 'ArrowRight') {
        this.updateDisplay('→ Right');
      }
    }
  
    // Method to get current control state
    getControlState() {
      return {
        left: this.left,
        right: this.right
      };
    }
  
    // Method to set game over state
    setGameOver(isOver) {
      this.isGameOver = isOver;
    }
  }
  
  // Initialize when the DOM is loaded
  document.addEventListener('DOMContentLoaded', () => {
    const keyPressDisplay = new KeyPressDisplay();
    // Make the instance available globally if needed
    window.keyPressDisplay = keyPressDisplay;
  }); 