// Web Worker for decoding base64 images off the main thread
// This prevents UI blocking during large image decoding

self.onmessage = function(e) {
    const startTime = performance.now();
    const { image } = e.data;

    try {
        // Use data URL directly - more reliable than blob URL
        // No need to decode base64 in worker - just pass through
        const dataUrl = 'data:image/jpeg;base64,' + image;

        const decodeTime = performance.now() - startTime;
        // Send back the data URL and timing
        self.postMessage({ url: dataUrl, decodeTime: decodeTime, size: image.length });
    } catch (error) {
        self.postMessage({ error: error.message });
    }
};