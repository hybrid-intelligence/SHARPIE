// Define MobileControls in the global scope
const MobileControls = function({ onLeftPress, onRightPress }) {
  return React.createElement('div', { className: 'mobile-controls' },
    React.createElement('style', null, `
      .mobile-controls {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        display: flex;
        justify-content: space-around;
        padding: 15px;
        background: rgba(255, 255, 255, 0.98);
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        gap: 15px;
      }
      .control-button {
        flex: 1;
        max-width: 140px;
        height: 50px;
        border-radius: 10px;
        color: white;
        border: none;
        font-size: 18px;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
        touch-action: manipulation;
        -webkit-tap-highlight-color: transparent;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        user-select: none;
      }
      .control-button:active {
        transform: translateY(2px);
        box-shadow: 0 2px 3px rgba(0, 0, 0, 0.1);
      }
      .control-button.left {
        background:rgb(228, 112, 124);
      }
      .control-button.left:active {
        background: #c82333;
      }
      .control-button.right {
        background:rgb(158, 233, 175);
      }
      .control-button.right:active {
        background: #218838;
      }
    `),
    React.createElement('button', {
      className: 'control-button left',
      onTouchStart: onLeftPress,
      onTouchEnd: () => document.dispatchEvent(new KeyboardEvent('keyup', { key: 'ArrowLeft' }))
    }, 'Left'),
    React.createElement('button', {
      className: 'control-button right',
      onTouchStart: onRightPress,
      onTouchEnd: () => document.dispatchEvent(new KeyboardEvent('keyup', { key: 'ArrowRight' }))
    }, 'Right')
  );
};

// Make MobileControls available globally
window.MobileControls = MobileControls; 