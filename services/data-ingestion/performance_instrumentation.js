/**
 * Performance Instrumentation for Data Ingestion Service
 * Provides microsecond-precision timing and comprehensive metrics collection
 */

const { performance } = require('perf_hooks');
const client = require('prom-client');

// Prometheus metrics
const processingHistogram = new client.Histogram({
    name: 'message_processing_duration_microseconds',
    help: 'Time spent processing messages in microseconds',
    labelNames: ['stage', 'language', 'code_switched'],
    buckets: [100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000]
});

const requestCounter = new client.Counter({
    name: 'requests_total',
    help: 'Total number of requests processed',
    labelNames: ['status', 'language', 'error_type']
});

const activeRequestsGauge = new client.Gauge({
    name: 'active_requests',
    help: 'Number of currently active requests'
});

const queueDepthGauge = new client.Gauge({
    name: 'queue_depth',
    help: 'Current queue depth'
});

// System metrics
const cpuUsageGauge = new client.Gauge({
    name: 'cpu_usage_percent',
    help: 'CPU usage percentage'
});

const memoryUsageGauge = new client.Gauge({
    name: 'memory_usage_mb',
    help: 'Memory usage in MB'
});

// Register metrics
client.register.registerMetric(processingHistogram);
client.register.registerMetric(requestCounter);
client.register.registerMetric(activeRequestsGauge);
client.register.registerMetric(queueDepthGauge);
client.register.registerMetric(cpuUsageGauge);
client.register.registerMetric(memoryUsageGauge);

class PerformanceInstrumentation {
    constructor() {
        this.activeRequests = new Map();
        this.systemMonitorInterval = null;
        this.startSystemMonitoring();
    }

    startSystemMonitoring() {
        // Monitor system resources every second
        this.systemMonitorInterval = setInterval(() => {
            const usage = process.cpuUsage();
            const memUsage = process.memoryUsage();
            
            // Convert to percentages and MB
            const cpuPercent = (usage.user + usage.system) / 1000000 * 100; // Convert microseconds to percentage
            const memMB = memUsage.heapUsed / 1024 / 1024;
            
            cpuUsageGauge.set(cpuPercent);
            memoryUsageGauge.set(memMB);
        }, 1000);
    }

    startRequest(messageId, metadata = {}) {
        const requestData = {
            messageId,
            arrivalTime: performance.now() * 1000, // Convert to microseconds
            stages: {},
            metadata: {
                language: metadata.language || 'unknown',
                isCodeSwitched: metadata.isCodeSwitched || false,
                userAgent: metadata.userAgent,
                sourceIp: metadata.sourceIp,
                ...metadata
            },
            systemMetrics: {
                startCpuUsage: process.cpuUsage(),
                startMemoryUsage: process.memoryUsage()
            }
        };

        this.activeRequests.set(messageId, requestData);
        activeRequestsGauge.inc();
        
        console.log(`[METRICS] Request started: ${messageId} at ${requestData.arrivalTime}μs`);
        return requestData;
    }

    startStage(messageId, stageName) {
        const request = this.activeRequests.get(messageId);
        if (!request) {
            console.warn(`[METRICS] Request not found: ${messageId}`);
            return;
        }

        const stageStartTime = performance.now() * 1000;
        request.stages[stageName] = {
            startTime: stageStartTime,
            endTime: null,
            duration: null
        };

        console.log(`[METRICS] Stage ${stageName} started for ${messageId} at ${stageStartTime}μs`);
    }

    endStage(messageId, stageName, result = {}) {
        const request = this.activeRequests.get(messageId);
        if (!request || !request.stages[stageName]) {
            console.warn(`[METRICS] Stage not found: ${stageName} for ${messageId}`);
            return;
        }

        const stageEndTime = performance.now() * 1000;
        const stage = request.stages[stageName];
        stage.endTime = stageEndTime;
        stage.duration = stageEndTime - stage.startTime;
        stage.result = result;

        // Record in Prometheus
        processingHistogram
            .labels(stageName, request.metadata.language, request.metadata.isCodeSwitched.toString())
            .observe(stage.duration);

        console.log(`[METRICS] Stage ${stageName} completed for ${messageId}: ${stage.duration}μs`);
    }

    completeRequest(messageId, status = 'success', error = null) {
        const request = this.activeRequests.get(messageId);
        if (!request) {
            console.warn(`[METRICS] Request not found for completion: ${messageId}`);
            return;
        }

        const completionTime = performance.now() * 1000;
        const totalDuration = completionTime - request.arrivalTime;

        // Capture final system metrics
        const endCpuUsage = process.cpuUsage();
        const endMemoryUsage = process.memoryUsage();

        request.completionTime = completionTime;
        request.totalDuration = totalDuration;
        request.status = status;
        request.error = error;
        request.systemMetrics.endCpuUsage = endCpuUsage;
        request.systemMetrics.endMemoryUsage = endMemoryUsage;

        // Calculate CPU usage during request
        const cpuDelta = {
            user: endCpuUsage.user - request.systemMetrics.startCpuUsage.user,
            system: endCpuUsage.system - request.systemMetrics.startCpuUsage.system
        };
        request.systemMetrics.cpuUsageDuringRequest = cpuDelta;

        // Record metrics
        requestCounter
            .labels(status, request.metadata.language, error ? error.split(':')[0] : 'none')
            .inc();

        processingHistogram
            .labels('total', request.metadata.language, request.metadata.isCodeSwitched.toString())
            .observe(totalDuration);

        // Log comprehensive metrics
        const metricsLog = {
            messageId,
            arrivalTime: request.arrivalTime,
            completionTime,
            totalDuration,
            status,
            error,
            language: request.metadata.language,
            isCodeSwitched: request.metadata.isCodeSwitched,
            stages: Object.keys(request.stages).reduce((acc, stageName) => {
                const stage = request.stages[stageName];
                acc[stageName] = {
                    duration: stage.duration,
                    startTime: stage.startTime,
                    endTime: stage.endTime
                };
                return acc;
            }, {}),
            systemMetrics: {
                memoryUsedMB: endMemoryUsage.heapUsed / 1024 / 1024,
                cpuTimeMicroseconds: cpuDelta.user + cpuDelta.system
            },
            activeRequestsCount: this.activeRequests.size
        };

        console.log(`[METRICS] Request completed:`, JSON.stringify(metricsLog, null, 2));

        // Send to external metrics collector if configured
        this.sendToMetricsCollector(metricsLog);

        // Clean up
        this.activeRequests.delete(messageId);
        activeRequestsGauge.dec();
    }

    async sendToMetricsCollector(metricsData) {
        try {
            // Send to metrics collector service
            const response = await fetch('http://metrics-collector:9090/metrics', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(metricsData)
            });

            if (!response.ok) {
                console.warn(`[METRICS] Failed to send metrics: ${response.status}`);
            }
        } catch (error) {
            console.warn(`[METRICS] Error sending metrics:`, error.message);
        }
    }

    getActiveRequestsCount() {
        return this.activeRequests.size;
    }

    getQueueDepth() {
        // This would integrate with your actual queue system
        // For now, return active requests as proxy
        return this.activeRequests.size;
    }

    updateQueueDepth(depth) {
        queueDepthGauge.set(depth);
    }

    generateReport() {
        const activeRequests = Array.from(this.activeRequests.values());
        const currentTime = performance.now() * 1000;

        return {
            timestamp: currentTime,
            activeRequestsCount: activeRequests.length,
            activeRequests: activeRequests.map(req => ({
                messageId: req.messageId,
                arrivalTime: req.arrivalTime,
                ageInMicroseconds: currentTime - req.arrivalTime,
                currentStages: Object.keys(req.stages).filter(stage => 
                    req.stages[stage].endTime === null
                ),
                completedStages: Object.keys(req.stages).filter(stage => 
                    req.stages[stage].endTime !== null
                ).map(stage => ({
                    name: stage,
                    duration: req.stages[stage].duration
                }))
            })),
            systemMetrics: {
                cpuUsage: process.cpuUsage(),
                memoryUsage: process.memoryUsage(),
                uptime: process.uptime()
            }
        };
    }

    shutdown() {
        if (this.systemMonitorInterval) {
            clearInterval(this.systemMonitorInterval);
        }
        console.log('[METRICS] Performance instrumentation shutdown');
    }
}

// Export metrics endpoint
function getMetricsEndpoint() {
    return async (req, res) => {
        try {
            res.set('Content-Type', client.register.contentType);
            res.end(await client.register.metrics());
        } catch (error) {
            res.status(500).end(error);
        }
    };
}

module.exports = {
    PerformanceInstrumentation,
    getMetricsEndpoint,
    processingHistogram,
    requestCounter,
    activeRequestsGauge,
    queueDepthGauge
}; 