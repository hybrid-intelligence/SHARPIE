// Define MobileControls in the global scope
const MobileControls = function({ onUpPress, onDownPress, onLeftPress, onRightPress }) {
  return React.createElement('div', { className: 'mobile-controls' },
    React.createElement('style', null, `
      .mobile-controls {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        grid-template-rows: repeat(3, 1fr);
        gap: 6px;
        padding: 10px 8px 8px 8px;
        background: rgba(255, 255, 255, 0);
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.08);
        z-index: 1000;
        max-width: 400px;
        margin: 0 auto;
      }
      .control-button {
        width: 100%;
        height: 38px;
        border-radius: 8px;
        color: white;
        border: none;
        font-size: 16px;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
        touch-action: manipulation;
        -webkit-tap-highlight-color: transparent;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0);
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        user-select: none;
      }
      .control-button:active {
        transform: translateY(2px);
        box-shadow: 0 2px 3px rgba(0, 0, 0, 0.1);
      }
      .control-button.up {
        background: rgb(127, 222, 231);
        grid-column: 2;
        grid-row: 1;
      }
      .control-button.down {
        background: rgb(127, 222, 231);
        grid-column: 2;
        grid-row: 3;
      }
      .control-button.left {
        background: rgb(127, 222, 231);
        grid-column: 1;
        grid-row: 2;
      }
      .control-button.right {
        background: rgb(127, 222, 231);
        grid-column: 3;
        grid-row: 2;
      }
      .control-button.up:active {
        background: rgb(0, 204, 255);
      }
      .control-button.down:active {
        background: rgb(0, 204, 255);
      }
      .control-button.left:active {
        background: rgb(0, 204, 255);
      }
      .control-button.right:active {
        background: rgb(0, 204, 255);
      }
    `),
    React.createElement('button', {
      className: 'control-button up',
      onTouchStart: onUpPress,
      onTouchEnd: () => document.dispatchEvent(new KeyboardEvent('keyup', { key: 'ArrowUp' }))
    }, '↑'),
    React.createElement('button', {
      className: 'control-button down',
      onTouchStart: onDownPress,
      onTouchEnd: () => document.dispatchEvent(new KeyboardEvent('keyup', { key: 'ArrowDown' }))
    }, '↓'),
    React.createElement('button', {
      className: 'control-button left',
      onTouchStart: onLeftPress,
      onTouchEnd: () => document.dispatchEvent(new KeyboardEvent('keyup', { key: 'ArrowLeft' }))
    }, '←'),
    React.createElement('button', {
      className: 'control-button right',
      onTouchStart: onRightPress,
      onTouchEnd: () => document.dispatchEvent(new KeyboardEvent('keyup', { key: 'ArrowRight' }))
    }, '→')
  );
};

window.MobileControls = MobileControls; 