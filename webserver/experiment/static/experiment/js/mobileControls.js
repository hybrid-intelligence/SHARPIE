// Mobile Controls JavaScript for Experiment UI

// Mobile detection and controls
function isMobileDevice() {
  return (window.innerWidth <= 768) || ('ontouchstart' in window);
}

function showMobileControls() {
  if (isMobileDevice()) {
    document.getElementById('mobile-controls').style.display = 'block';
  }
}

function hideMobileControls() {
  document.getElementById('mobile-controls').style.display = 'none';
}

// Attach event listeners to mobile control buttons
document.addEventListener('DOMContentLoaded', function() {
  const mobileButtons = document.querySelectorAll('.mobile-control-btn');
  mobileButtons.forEach(button => {
    const key = button.getAttribute('data-key');
    
    // Touch events
    button.addEventListener('touchstart', function(e) {
      e.preventDefault();
      button.classList.add('pressed');
      document.dispatchEvent(new KeyboardEvent('keydown', { key: key }));
    });
    
    button.addEventListener('touchend', function(e) {
      e.preventDefault();
      button.classList.remove('pressed');
      document.dispatchEvent(new KeyboardEvent('keyup', { key: key }));
    });
    
    // Mouse events for desktop testing
    button.addEventListener('mousedown', function(e) {
      e.preventDefault();
      button.classList.add('pressed');
      document.dispatchEvent(new KeyboardEvent('keydown', { key: key }));
    });
    
    button.addEventListener('mouseup', function(e) {
      e.preventDefault();
      button.classList.remove('pressed');
      document.dispatchEvent(new KeyboardEvent('keyup', { key: key }));
    });
    
    // Handle touch cancel (when touch is interrupted)
    button.addEventListener('touchcancel', function(e) {
      e.preventDefault();
      button.classList.remove('pressed');
    });
  });
});

// Show/hide mobile controls based on screen size
window.addEventListener('resize', function() {
  if (isMobileDevice()) {
    showMobileControls();
  } else {
    hideMobileControls();
  }
});

// Initial check
if (isMobileDevice()) {
  showMobileControls();
}
