class ConnectionChecker {
    constructor() {
        this.bandwidthThreshold = 1.0; // Mbps - minimum acceptable bandwidth
        this.latencyThreshold = 200; // ms - maximum acceptable latency
        this.testImageSize = 100000; // bytes - size of test image for bandwidth test
    }

    /**
     * Measures network latency by pinging the server
     * @returns {Promise<number>} Latency in milliseconds
     */
    async measureLatency() {
        const startTime = performance.now();
        try {
            // Use a small fetch request to measure latency
            await fetch(window.location.origin + '/?t=' + Date.now(), {
                method: 'HEAD',
                cache: 'no-cache'
            });
            const endTime = performance.now();
            return endTime - startTime;
        } catch (error) {
            console.error('Latency measurement failed:', error);
            return Infinity; // Treat errors as very poor connection
        }
    }

    /**
     * Measures download bandwidth
     * @returns {Promise<number>} Bandwidth in Mbps
     */
    async measureBandwidth() {
        const startTime = performance.now();
        try {
            // Try to use a static asset for bandwidth testing
            // Add timestamp to prevent caching
            const testUrl = window.location.origin + '/static/home/logo.png?t=' + Date.now();
            
            const response = await fetch(testUrl, {
                cache: 'no-cache',
                method: 'GET'
            });
            
            if (!response.ok) {
                console.warn('Bandwidth test asset not available, using fallback');
                // Fallback: assume minimum acceptable bandwidth if test fails
                return this.bandwidthThreshold;
            }
            
            const blob = await response.blob();
            const endTime = performance.now();
            
            const duration = (endTime - startTime) / 1000; // Convert to seconds
            if (duration <= 0) {
                // If duration is too short, assume good connection
                return this.bandwidthThreshold * 2;
            }
            
            const sizeInBits = blob.size * 8; // Convert bytes to bits
            const bandwidthMbps = (sizeInBits / duration) / 1000000; // Convert to Mbps
            
            return bandwidthMbps;
        } catch (error) {
            console.error('Bandwidth measurement failed:', error);
            // On error, assume minimum acceptable bandwidth to avoid false positives
            return this.bandwidthThreshold;
        }
    }

    /**
     * Checks if connection quality is poor
     * @returns {Promise<{isPoor: boolean, metrics: object}>}
     */
    async checkConnectionQuality() {
        console.log('Checking connection quality...');
        
        const latency = await this.measureLatency();
        const bandwidth = await this.measureBandwidth();
        
        const metrics = {
            latency: latency,
            bandwidth: bandwidth,
            timestamp: new Date().toISOString()
        };
        
        // Determine if connection is poor
        const isPoor = latency > this.latencyThreshold || bandwidth < this.bandwidthThreshold;
        
        if (isPoor) {
            console.warn('Poor connection detected:', metrics);
        } else {
            console.log('Connection quality acceptable:', metrics);
        }
        
        return {
            isPoor: isPoor,
            metrics: metrics
        };
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConnectionChecker;
}
