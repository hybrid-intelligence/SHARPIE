/**
 * History management for experiment interface.
 * Tracks user inputs and actions in a scrollable history panel.
 */

// History management
function addToHistory(type, content, result = null) {
  const historyDiv = document.getElementById('history');
  if (!historyDiv) return;

  const timestamp = new Date().toLocaleTimeString();

  // Remove "No history yet" message if present (only the specific paragraph)
  const noHistory = historyDiv.querySelector('p.text-muted');
  if (noHistory) {
    noHistory.remove();
  }

  const entry = document.createElement('div');
  entry.className = 'history-entry mb-2 pb-2 border-bottom';

  const typeColors = {
    'input': 'primary',
    'action': 'success',
    'error': 'danger',
    'info': 'secondary'
  };
  const color = typeColors[type] || 'secondary';

  entry.innerHTML = `
    <div class="d-flex justify-content-between align-items-start">
      <span class="badge bg-${color}">${type}</span>
      <small class="text-muted">${timestamp}</small>
    </div>
    <div class="mt-1">${content}</div>
    ${result ? `<div class="text-muted small mt-1">→ ${result}</div>` : ''}
  `;

  historyDiv.insertBefore(entry, historyDiv.firstChild);

  // Keep only last 50 entries
  while (historyDiv.children.length > 50) {
    historyDiv.removeChild(historyDiv.lastChild);
  }
}

// Submit button handler
document.addEventListener('DOMContentLoaded', function() {
  const submitBtn = document.getElementById('instruction-submit');
  const inputField = document.getElementById('instruction-input');

  if (submitBtn && inputField) {
    submitBtn.addEventListener('click', function() {
      const instruction = inputField.value.trim();
      if (instruction) {
        addToHistory('input', instruction);
        // Trigger the input event or call the appropriate handler
        const event = new CustomEvent('instruction-submit', { detail: { instruction: instruction } });
        document.dispatchEvent(event);
        inputField.value = '';
      }
    });

    // Also submit on Enter key
    inputField.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        submitBtn.click();
      }
    });
  }
});

// Export for use in other modules
window.addToHistory = addToHistory;