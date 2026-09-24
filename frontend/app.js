/**
 * Sentrix AI - Model Testing Dashboard Logic
 * Team Sentrix - SIH 2026
 */

document.addEventListener('DOMContentLoaded', () => {
    // State
    const state = {
        currentImageFile: null,
        currentSamplePath: null,
        currentVideoSamplePath: 'test_videos/road.mp4',
        modelStatus: null,
    };

    // DOM Elements - Tabs
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    // DOM Elements - Status Header
    const deviceText = document.getElementById('deviceText');
    const deviceStatusChip = document.getElementById('deviceStatusChip');
    const modelStatusText = document.getElementById('modelStatusText');
    const modelStatusChip = document.getElementById('modelStatusChip');

    // DOM Elements - Image Tab
    const imageModelSelect = document.getElementById('imageModelSelect');
    const imageThresholdSlider = document.getElementById('imageThresholdSlider');
    const imageThresholdVal = document.getElementById('imageThresholdVal');
    const runImageDetectBtn = document.getElementById('runImageDetectBtn');
    const imageSpinner = document.getElementById('imageSpinner');
    const imageSamplesGrid = document.getElementById('imageSamplesGrid');

    const imageDropzone = document.getElementById('imageDropzone');
    const imageFileInput = document.getElementById('imageFileInput');
    const dropzoneEmpty = document.getElementById('dropzoneEmpty');
    const imagePreviewWrapper = document.getElementById('imagePreviewWrapper');
    const imageSourcePreview = document.getElementById('imageSourcePreview');
    const removeImageBtn = document.getElementById('removeImageBtn');
    const imageResolutionBadge = document.getElementById('imageResolutionBadge');

    const imageResultPlaceholder = document.getElementById('imageResultPlaceholder');
    const annotatedImageWrapper = document.getElementById('annotatedImageWrapper');
    const imageAnnotatedResult = document.getElementById('imageAnnotatedResult');
    const latencyPill = document.getElementById('latencyPill');
    const fpsPill = document.getElementById('fpsPill');
    const countPill = document.getElementById('countPill');

    const detectionsSummaryText = document.getElementById('detectionsSummaryText');
    const detectionsTableBody = document.getElementById('detectionsTableBody');

    // DOM Elements - Video Tab
    const videoModelSelect = document.getElementById('videoModelSelect');
    const videoThresholdSlider = document.getElementById('videoThresholdSlider');
    const videoThresholdVal = document.getElementById('videoThresholdVal');
    const frameSkipSlider = document.getElementById('frameSkipSlider');
    const frameSkipVal = document.getElementById('frameSkipVal');
    const runVideoDetectBtn = document.getElementById('runVideoDetectBtn');
    const videoSpinner = document.getElementById('videoSpinner');
    const videoSamplesGrid = document.getElementById('videoSamplesGrid');

    const sourceVideoPlayer = document.getElementById('sourceVideoPlayer');
    const detectedVideoWrapper = document.getElementById('detectedVideoWrapper');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const detectedVideoPlayer = document.getElementById('detectedVideoPlayer');
    const detectedVideoSource = document.getElementById('detectedVideoSource');
    const videoFpsPill = document.getElementById('videoFpsPill');
    const videoDetectionsPill = document.getElementById('videoDetectionsPill');

    // =========================================================================
    // 1. Tab Navigation
    // =========================================================================
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            tabButtons.forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            tabPanes.forEach(pane => pane.classList.remove('active'));

            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');
            const activePane = document.getElementById(targetTab);
            if (activePane) activePane.classList.add('active');
        });
    });

    // =========================================================================
    // 2. Sliders Synchronization
    // =========================================================================
    imageThresholdSlider.addEventListener('input', (e) => {
        imageThresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    videoThresholdSlider.addEventListener('input', (e) => {
        videoThresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    frameSkipSlider.addEventListener('input', (e) => {
        frameSkipVal.textContent = `Every ${e.target.value}${e.target.value === '1' ? 'st' : (e.target.value === '2' ? 'nd' : (e.target.value === '3' ? 'rd' : 'th'))}`;
    });

    imageModelSelect.addEventListener('change', (e) => {
        if (e.target.value === 'incident') {
            imageThresholdSlider.value = 0.30;
            imageThresholdVal.textContent = '0.30';
        }
    });

    videoModelSelect.addEventListener('change', (e) => {
        if (e.target.value === 'incident') {
            videoThresholdSlider.value = 0.35;
            videoThresholdVal.textContent = '0.35';
        }
    });

    // =========================================================================
    // 3. System Status & Sample Media Fetching
    // =========================================================================
    async function initSystemStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            state.modelStatus = data;

            // Device Chip
            deviceText.textContent = data.device_name || `Device: ${data.device.toUpperCase()}`;
            deviceStatusChip.classList.add('chip-online');

            // Model Chip
            const potholeModel = data.models.pothole;
            const incidentModel = data.models.incident;
            const waterloggingModel = data.models.waterlogging;

            const readyModels = [];
            if (potholeModel && potholeModel.available) readyModels.push('Potholes');
            if (incidentModel && incidentModel.available) readyModels.push('Incidents');
            if (waterloggingModel && waterloggingModel.available) readyModels.push('Waterlogging');

            if (readyModels.length > 0) {
                modelStatusText.textContent = `Models Ready: ${readyModels.join(', ')}`;
                modelStatusChip.classList.add('chip-online');
            } else {
                modelStatusText.textContent = 'Models Missing Checkpoints';
                modelStatusChip.style.borderColor = 'var(--accent-amber)';
                modelStatusChip.style.color = 'var(--accent-amber)';
            }
        } catch (err) {
            console.error('Failed to query status:', err);
            deviceText.textContent = 'Backend Offline';
            modelStatusText.textContent = 'Server Unavailable';
        }
    }

    async function loadSamples() {
        try {
            const res = await fetch('/api/samples');
            const samples = await res.json();

            // Render Image Samples
            if (samples.images && samples.images.length > 0) {
                imageSamplesGrid.innerHTML = '';
                samples.images.forEach((img, idx) => {
                    const chip = document.createElement('div');
                    chip.className = `sample-chip ${idx === 0 ? 'active' : ''}`;
                    chip.innerHTML = `
                        <span class="sample-badge">${img.folder}</span>
                        <span>${img.name}</span>
                    `;
                    chip.addEventListener('click', () => {
                        document.querySelectorAll('#imageSamplesGrid .sample-chip').forEach(c => c.classList.remove('active'));
                        chip.classList.add('active');
                        selectImageSample(img);
                    });
                    imageSamplesGrid.appendChild(chip);
                });

                // Pre-select first image
                selectImageSample(samples.images[0]);
            } else {
                imageSamplesGrid.innerHTML = '<span class="text-muted">No sample images found.</span>';
            }

            // Render Video Samples
            if (samples.videos && samples.videos.length > 0) {
                videoSamplesGrid.innerHTML = '';
                samples.videos.forEach((vid, idx) => {
                    const chip = document.createElement('div');
                    chip.className = `sample-chip ${idx === 0 ? 'active' : ''}`;
                    chip.innerHTML = `
                        <span class="sample-badge">Video</span>
                        <span>${vid.name}</span>
                    `;
                    chip.addEventListener('click', () => {
                        document.querySelectorAll('#videoSamplesGrid .sample-chip').forEach(c => c.classList.remove('active'));
                        chip.classList.add('active');
                        selectVideoSample(vid);
                    });
                    videoSamplesGrid.appendChild(chip);
                });
            } else {
                videoSamplesGrid.innerHTML = '<span class="text-muted">No sample videos found.</span>';
            }
        } catch (err) {
            console.error('Failed to load samples:', err);
        }
    }

    function selectImageSample(sample) {
        state.currentImageFile = null;
        state.currentSamplePath = sample.path;

        // Auto-select model type based on sample
        if (sample.name.toLowerCase().includes('bus')) {
            imageModelSelect.value = 'coco';
        } else if (sample.name.toLowerCase().includes('pothole')) {
            imageModelSelect.value = 'pothole';
        }

        imageSourcePreview.src = sample.url;
        imageSourcePreview.onload = () => {
            imageResolutionBadge.textContent = `${imageSourcePreview.naturalWidth} × ${imageSourcePreview.naturalHeight}`;
        };

        dropzoneEmpty.classList.add('hidden');
        imagePreviewWrapper.classList.remove('hidden');
    }

    function selectVideoSample(sample) {
        state.currentVideoSamplePath = sample.path;
        sourceVideoPlayer.src = sample.url;
        sourceVideoPlayer.load();
    }

    // =========================================================================
    // 4. Image Upload & Drag-and-Drop
    // =========================================================================
    imageDropzone.addEventListener('click', (e) => {
        if (e.target !== removeImageBtn && !removeImageBtn.contains(e.target)) {
            imageFileInput.click();
        }
    });

    imageFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleImageFile(file);
    });

    ['dragenter', 'dragover'].forEach(event => {
        imageDropzone.addEventListener(event, (e) => {
            e.preventDefault();
            imageDropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(event => {
        imageDropzone.addEventListener(event, (e) => {
            e.preventDefault();
            imageDropzone.classList.remove('dragover');
        });
    });

    imageDropzone.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleImageFile(file);
        }
    });

    function handleImageFile(file) {
        state.currentImageFile = file;
        state.currentSamplePath = null;

        // Clear sample active states
        document.querySelectorAll('#imageSamplesGrid .sample-chip').forEach(c => c.classList.remove('active'));

        const reader = new FileReader();
        reader.onload = (e) => {
            imageSourcePreview.src = e.target.result;
            imageSourcePreview.onload = () => {
                imageResolutionBadge.textContent = `${imageSourcePreview.naturalWidth} × ${imageSourcePreview.naturalHeight}`;
            };
            dropzoneEmpty.classList.add('hidden');
            imagePreviewWrapper.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        state.currentImageFile = null;
        state.currentSamplePath = null;
        imageFileInput.value = '';
        imageSourcePreview.src = '';
        imageResolutionBadge.textContent = 'No Image';
        imagePreviewWrapper.classList.add('hidden');
        dropzoneEmpty.classList.remove('hidden');
        document.querySelectorAll('#imageSamplesGrid .sample-chip').forEach(c => c.classList.remove('active'));
    });

    // =========================================================================
    // 5. Image Detection Inference
    // =========================================================================
    runImageDetectBtn.addEventListener('click', async () => {
        if (!state.currentImageFile && !state.currentSamplePath) {
            alert('Please select or upload an image first.');
            return;
        }

        const modelName = imageModelSelect.value;
        const threshold = parseFloat(imageThresholdSlider.value);

        const formData = new FormData();
        formData.append('model_name', modelName);
        formData.append('threshold', threshold);

        if (state.currentImageFile) {
            formData.append('file', state.currentImageFile);
        } else if (state.currentSamplePath) {
            formData.append('sample_path', state.currentSamplePath);
        }

        // UI Loading state
        runImageDetectBtn.disabled = true;
        imageSpinner.classList.remove('hidden');

        try {
            const res = await fetch('/api/detect/image', {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(errData.detail || 'Inference request failed');
            }

            const data = await res.json();
            renderImageResults(data);
        } catch (err) {
            alert(`Detection Error: ${err.message}`);
            console.error('Inference error:', err);
        } finally {
            runImageDetectBtn.disabled = false;
            imageSpinner.classList.add('hidden');
        }
    });

    function renderImageResults(data) {
        // Telemetry Pills
        latencyPill.textContent = `${data.latency_ms} ms`;
        fpsPill.textContent = `${data.fps_estimate} FPS`;
        countPill.textContent = `${data.total_detections} Detected`;

        // Render Annotated Image
        imageResultPlaceholder.classList.add('hidden');
        annotatedImageWrapper.classList.remove('hidden');
        imageAnnotatedResult.src = data.image_base64;

        // Render Table
        detectionsSummaryText.textContent = `${data.total_detections} item(s) found in ${data.latency_ms} ms (${data.model_used.toUpperCase()})`;
        detectionsTableBody.innerHTML = '';

        if (!data.detections || data.detections.length === 0) {
            const hint = data.model_used === 'pothole'
                ? '<br><span style="color: var(--accent-cyan); margin-top: 6px; display: inline-block;">💡 Tip: Looking for buses, cars, or people? The Pothole model only searches for potholes. Switch the Model Engine dropdown above to <strong>COCO General Model</strong>.</span>'
                : '<br><span class="text-muted">Try lowering the confidence threshold slider above.</span>';

            detectionsTableBody.innerHTML = `
                <tr>
                    <td colspan="5" class="table-empty">
                        No ${data.model_used === 'pothole' ? 'potholes' : 'objects'} detected at confidence threshold ≥ ${imageThresholdSlider.value}.
                        ${hint}
                    </td>
                </tr>
            `;
            return;
        }

        data.detections.forEach(item => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>#${item.id}</strong></td>
                <td><span class="tag-class">${item.class_name}</span></td>
                <td>
                    <div class="confidence-bar-wrapper">
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill" style="width: ${item.confidence_percent}%;"></div>
                        </div>
                        <span style="font-family: var(--font-mono); font-weight: 600;">${item.confidence_percent}%</span>
                    </div>
                </td>
                <td style="font-family: var(--font-mono); font-size: 0.82rem;">[${item.box.join(', ')}]</td>
                <td style="font-family: var(--font-mono); font-size: 0.82rem;">${item.width} × ${item.height} px</td>
            `;
            detectionsTableBody.appendChild(tr);
        });
    }

    // =========================================================================
    // 6. Video Detection Inference
    // =========================================================================
    runVideoDetectBtn.addEventListener('click', async () => {
        if (!state.currentVideoSamplePath) {
            alert('Please select a video first.');
            return;
        }

        const modelName = videoModelSelect.value;
        const threshold = parseFloat(videoThresholdSlider.value);
        const frameSkip = parseInt(frameSkipSlider.value, 10);

        const formData = new FormData();
        formData.append('model_name', modelName);
        formData.append('threshold', threshold);
        formData.append('frame_skip', frameSkip);
        formData.append('sample_path', state.currentVideoSamplePath);

        runVideoDetectBtn.disabled = true;
        videoSpinner.classList.remove('hidden');

        try {
            const res = await fetch('/api/detect/video', {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(errData.detail || 'Video processing failed');
            }

            const data = await res.json();
            renderVideoResults(data);
        } catch (err) {
            alert(`Video Detection Error: ${err.message}`);
            console.error('Video error:', err);
        } finally {
            runVideoDetectBtn.disabled = false;
            videoSpinner.classList.add('hidden');
        }
    });

    function renderVideoResults(data) {
        videoPlaceholder.classList.add('hidden');
        detectedVideoPlayer.classList.remove('hidden');

        // Append timestamp cache buster
        const url = `${data.video_url}?t=${Date.now()}`;
        if (detectedVideoSource) {
            detectedVideoSource.src = url;
        }
        detectedVideoPlayer.src = url;
        detectedVideoPlayer.load();
        detectedVideoPlayer.play().catch(err => {
            console.log('Autoplay deferred until user action:', err);
        });

        videoFpsPill.textContent = `${data.avg_fps} FPS`;
        if (data.model_used === 'incident') {
            videoDetectionsPill.textContent = `${data.unique_tracks || data.total_detections} Incident(s) Tracked`;
        } else if (data.model_used === 'waterlogging') {
            videoDetectionsPill.textContent = `${data.total_detections} Flood Patch(es) Found`;
        } else {
            videoDetectionsPill.textContent = `${data.total_detections} Pothole(s) Found`;
        }
    }

    // Init
    initSystemStatus();
    loadSamples();
});
