class ConnectionWarning {
    constructor() {
        this.modalId = 'connectionWarningModal';
        this.init();
    }

    init() {
        // Check if modal already exists
        if (document.getElementById(this.modalId)) {
            return;
        }

        // Create modal HTML
        const modalHTML = `
            <div class="modal fade" id="${this.modalId}" tabindex="-1" aria-labelledby="connectionWarningLabel" aria-hidden="true">
                <div class="modal-dialog modal-dialog-centered">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="connectionWarningLabel">
                                <i class="bi bi-exclamation-triangle-fill text-warning me-2"></i>
                                Connection Warning
                            </h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <p>Your connection is unstable. This may affect the experiment.</p>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Append modal to body
        document.body.insertAdjacentHTML('beforeend', modalHTML);
    }

    /**
     * Show the warning modal
     */
    show() {
        const modalElement = document.getElementById(this.modalId);
        if (modalElement) {
            const modal = new bootstrap.Modal(modalElement);
            modal.show();
        }
    }

    /**
     * Hide the warning modal
     */
    hide() {
        const modalElement = document.getElementById(this.modalId);
        if (modalElement) {
            const modal = bootstrap.Modal.getInstance(modalElement);
            if (modal) {
                modal.hide();
            }
        }
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConnectionWarning;
}
