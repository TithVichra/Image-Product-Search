/**
 * FashionLens AI - Frontend Controller
 * Multimodal Product Search with YOLOv8 Detection and Dual-Modal Scoring
 */

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const tabTextMode = document.getElementById("tab-text-mode");
    const tabImageMode = document.getElementById("tab-image-mode");
    const textSearchSection = document.getElementById("text-search-section");
    const imageSearchSection = document.getElementById("image-search-section");

    const textQueryInput = document.getElementById("text-query-input");
    const btnTextSearch = document.getElementById("btn-text-search");
    const tagChips = document.querySelectorAll(".tag-chip");

    const imageDropzone = document.getElementById("image-dropzone");
    const imageFileInput = document.getElementById("image-file-input");
    const dropzonePlaceholder = document.getElementById("dropzone-placeholder");
    const detectionPreviewCard = document.getElementById("detection-preview-card");
    const detectionCanvas = document.getElementById("detection-canvas");
    const cropPreviewImg = document.getElementById("crop-preview-img");
    const detectedCountBadge = document.getElementById("detected-count-badge");
    const useCropCheckbox = document.getElementById("use-crop-checkbox");
    const btnReupload = document.getElementById("btn-reupload");
    const btnImageSearch = document.getElementById("btn-image-search");

    const noiseFilterToggle = document.getElementById("noise-filter-toggle");
    const gapThresholdSlider = document.getElementById("gap-threshold-slider");
    const gapValDisplay = document.getElementById("gap-val-display");

    const resultsContainer = document.querySelector(".results-container");
    const resultsGrid = document.getElementById("results-grid");
    const resultsEmpty = document.getElementById("results-empty");
    const resultsLoading = document.getElementById("results-loading");
    const resultsCountBadge = document.getElementById("results-count-badge");
    const noiseEliminatedBadge = document.getElementById("noise-eliminated-badge");
    const noiseEliminatedText = document.getElementById("noise-eliminated-text");
    const searchModeBadge = document.getElementById("search-mode-badge");
    const searchLatencyBadge = document.getElementById("search-latency-badge");

    const indexedCountDisplay = document.getElementById("indexed-count");
    const openDatasetModalBtn = document.getElementById("open-dataset-modal-btn");
    const settingsModal = document.getElementById("settings-modal");
    const settingsCloseBtn = document.getElementById("settings-close-btn");
    const statImageCount = document.getElementById("stat-image-count");
    const statChunkCount = document.getElementById("stat-chunk-count");
    const btnTriggerIndexing = document.getElementById("btn-trigger-indexing");
    const indexingFeedback = document.getElementById("indexing-feedback");

    // State
    let currentMode = "text"; // "text" or "image"
    let uploadedImageFile = null;
    let detectionData = null;

    // --- Mode Switching (Exclusive Choice) ---
    function switchMode(mode) {
        currentMode = mode;
        if (mode === "text") {
            tabTextMode.classList.add("active");
            tabTextMode.setAttribute("aria-checked", "true");
            tabImageMode.classList.remove("active");
            tabImageMode.setAttribute("aria-checked", "false");

            textSearchSection.classList.remove("hidden");
            imageSearchSection.classList.add("hidden");
            searchModeBadge.textContent = "Mode: Description (75% text / 25% img)";
        } else {
            tabImageMode.classList.add("active");
            tabImageMode.setAttribute("aria-checked", "true");
            tabTextMode.classList.remove("active");
            tabTextMode.setAttribute("aria-checked", "false");

            imageSearchSection.classList.remove("hidden");
            textSearchSection.classList.add("hidden");
            searchModeBadge.textContent = "Mode: Image (75% img / 25% text)";
        }
    }

    tabTextMode.addEventListener("click", () => switchMode("text"));
    tabImageMode.addEventListener("click", () => switchMode("image"));

    // --- Slider listener ---
    gapThresholdSlider.addEventListener("input", (e) => {
        gapValDisplay.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // --- Quick Suggestion Chips ---
    tagChips.forEach(chip => {
        chip.addEventListener("click", () => {
            const query = chip.dataset.query;
            textQueryInput.value = query;
            triggerTextSearch(query);
        });
    });

    // --- Text Search Execution ---
    async function triggerTextSearch(queryText) {
        const query = (queryText || textQueryInput.value || "").trim();
        if (!query) {
            textQueryInput.focus();
            return;
        }

        setLoading(true);
        const eliminateNoise = noiseFilterToggle.checked;
        const gapThreshold = parseFloat(gapThresholdSlider.value);

        try {
            const response = await fetch("/api/search/text", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: query,
                    eliminate_noise: eliminateNoise,
                    gap_threshold: gapThreshold,
                    max_results: 10
                })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Search failed");
            }

            const data = await response.json();
            renderResults(data, "text");
        } catch (err) {
            console.error("Text search error:", err);
            alert("Search error: " + err.message);
        } finally {
            setLoading(false);
        }
    }

    btnTextSearch.addEventListener("click", () => triggerTextSearch());
    textQueryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            triggerTextSearch();
        }
    });

    // --- Image Dropzone & File Handling ---
    imageDropzone.addEventListener("click", () => imageFileInput.click());

    imageDropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        imageDropzone.classList.add("dragover");
    });

    imageDropzone.addEventListener("dragleave", () => {
        imageDropzone.classList.remove("dragover");
    });

    imageDropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        imageDropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleSelectedImage(e.dataTransfer.files[0]);
        }
    });

    imageFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedImage(e.target.files[0]);
        }
    });

    btnReupload.addEventListener("click", () => {
        imageFileInput.value = "";
        uploadedImageFile = null;
        detectionData = null;
        detectionPreviewCard.classList.add("hidden");
        dropzonePlaceholder.classList.remove("hidden");
    });

    async function handleSelectedImage(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please upload a valid image file.");
            return;
        }

        uploadedImageFile = file;
        dropzonePlaceholder.classList.add("hidden");
        detectionPreviewCard.classList.remove("hidden");

        // Run detection preview
        const formData = new FormData();
        formData.append("file", file);

        try {
            const resp = await fetch("/api/detect", {
                method: "POST",
                body: formData
            });

            if (!resp.ok) throw new Error("Detection failed");
            detectionData = await resp.json();

            // Draw bounding boxes on canvas
            drawDetections(file, detectionData.detected_boxes);
            cropPreviewImg.src = detectionData.crop_url;
            detectedCountBadge.textContent = `${detectionData.detected_boxes.length} item(s) detected`;
        } catch (err) {
            console.error("Detection error:", err);
            // Fallback preview
            const reader = new FileReader();
            reader.onload = (e) => {
                cropPreviewImg.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    }

    function drawDetections(imageFile, boxes) {
        const img = new Image();
        const reader = new FileReader();
        reader.onload = (e) => {
            img.src = e.target.result;
            img.onload = () => {
                const ctx = detectionCanvas.getContext("2d");
                detectionCanvas.width = img.width;
                detectionCanvas.height = img.height;
                ctx.drawImage(img, 0, 0);

                // Draw bounding boxes
                boxes.forEach((item, idx) => {
                    const [x1, y1, x2, y2] = item.box;
                    const w = x2 - x1;
                    const h = y2 - y1;

                    // Neon stroke
                    ctx.lineWidth = Math.max(3, Math.round(img.width / 250));
                    ctx.strokeStyle = item.is_selected ? "#10b981" : "#6366f1";
                    ctx.strokeRect(x1, y1, w, h);

                    // Label badge
                    const label = `${item.class_name} ${(item.confidence * 100).toFixed(0)}%`;
                    ctx.font = `bold ${Math.max(14, Math.round(img.width / 40))}px 'Plus Jakarta Sans', sans-serif`;
                    const textWidth = ctx.measureText(label).width;

                    ctx.fillStyle = item.is_selected ? "#10b981" : "#6366f1";
                    ctx.fillRect(x1, Math.max(0, y1 - 26), textWidth + 12, 26);

                    ctx.fillStyle = "#ffffff";
                    ctx.fillText(label, x1 + 6, Math.max(18, y1 - 7));
                });
            };
        };
        reader.readAsDataURL(imageFile);
    }

    // --- Search with Uploaded Image ---
    btnImageSearch.addEventListener("click", async () => {
        if (!uploadedImageFile) {
            alert("Please select or drop an image first.");
            return;
        }

        setLoading(true);
        const formData = new FormData();
        formData.append("file", uploadedImageFile);
        formData.append("use_crop", useCropCheckbox.checked);
        formData.append("eliminate_noise", noiseFilterToggle.checked);
        formData.append("gap_threshold", parseFloat(gapThresholdSlider.value));
        formData.append("max_results", 10);

        try {
            const resp = await fetch("/api/search/image", {
                method: "POST",
                body: formData
            });

            if (!resp.ok) {
                const errData = await resp.json();
                throw new Error(errData.detail || "Image search failed");
            }

            const data = await resp.json();
            renderResults(data, "image");
        } catch (err) {
            console.error("Image search error:", err);
            alert("Search error: " + err.message);
        } finally {
            setLoading(false);
        }
    });

    // --- Render Top 10 Results ---
    function renderResults(data, mode) {
        const results = data.results || [];
        const noiseMeta = data.noise_metadata || {};
        const latency = data.latency_sec ? (data.latency_sec * 1000).toFixed(0) : "0";

        const resultsOutOfDomain = document.getElementById("results-out-of-domain");
        const oodTitle = document.getElementById("ood-title");
        const oodMessage = document.getElementById("ood-message");
        const intentFilterBadge = document.getElementById("intent-filter-badge");

        searchLatencyBadge.textContent = `${latency} ms`;
        searchModeBadge.textContent = mode === "image" 
            ? "Mode: Image (75% img / 25% text)" 
            : "Mode: Description (75% text / 25% img)";

        // Intent badge display
        if (data.intent && (data.intent.target_category || data.intent.target_gender)) {
            const parts = [];
            if (data.intent.target_gender) parts.push(`Gender: ${data.intent.target_gender}`);
            if (data.intent.target_category) parts.push(`Category: ${data.intent.target_category}`);
            if (intentFilterBadge) {
                intentFilterBadge.textContent = parts.join(" • ");
                intentFilterBadge.classList.remove("hidden");
            }
        } else if (intentFilterBadge) {
            intentFilterBadge.classList.add("hidden");
        }

        // Out of domain handling (e.g. dog or non-fashion image)
        if (data.is_out_of_domain) {
            if (resultsOutOfDomain) {
                resultsOutOfDomain.classList.remove("hidden");
                resultsOutOfDomain.style.display = "flex";
                if (oodTitle) {
                    oodTitle.textContent = data.detected_entity 
                        ? `Non-Fashion Entity Detected: ${data.detected_entity.toUpperCase()}`
                        : "Non-Fashion Image Detected";
                }
                if (oodMessage) {
                    oodMessage.textContent = data.out_of_domain_reason || "The uploaded image appears to be an animal or non-apparel item. FashionLens search only indexes clothing, footwear, and accessories.";
                }
            }
            resultsEmpty.classList.add("hidden");
            resultsGrid.innerHTML = "";
            resultsCountBadge.textContent = "0 Matches (Out of Domain)";
            noiseEliminatedBadge.classList.add("hidden");
            resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
            return;
        } else if (resultsOutOfDomain) {
            resultsOutOfDomain.classList.add("hidden");
            resultsOutOfDomain.style.display = "none";
        }

        // Noise Elimination Badge
        if (noiseMeta.eliminated_count > 0) {
            noiseEliminatedBadge.classList.remove("hidden");
            noiseEliminatedText.textContent = `Eliminated ${noiseMeta.eliminated_count} noise items (Cliff drop: ${noiseMeta.gap_size || 'low score'})`;
        } else {
            noiseEliminatedBadge.classList.add("hidden");
        }

        resultsCountBadge.textContent = `Top ${results.length} Matches`;

        if (results.length === 0) {
            resultsEmpty.classList.remove("hidden");
            resultsGrid.innerHTML = "";
            return;
        }

        resultsEmpty.classList.add("hidden");
        resultsGrid.innerHTML = "";

        results.forEach((item) => {
            const card = document.createElement("article");
            card.className = "product-card";

            const rank = item.rank || 1;
            const rankClass = rank === 1 ? "rank-1" : (rank === 2 ? "rank-2" : (rank === 3 ? "rank-3" : "rank-other"));
            const overallScorePct = (item.score * 100).toFixed(1);

            // Tags
            const tagsHtml = [
                item.gender ? `<span class="meta-tag">${item.gender}</span>` : "",
                item.articleType ? `<span class="meta-tag">${item.articleType}</span>` : "",
                item.baseColour ? `<span class="meta-tag">${item.baseColour}</span>` : "",
                item.usage ? `<span class="meta-tag">${item.usage}</span>` : "",
            ].filter(Boolean).join("");

            // Chunk highlight if matched
            const chunkHtml = item.best_matching_chunk ? `
                <div class="chunk-highlight" title="Best matching description chunk">
                    <strong>Match Chunk:</strong> "${escapeHtml(item.best_matching_chunk)}"
                </div>
            ` : "";

            const primaryLabel = mode === "image" ? "Image Sim (75%)" : "Text Sim (75%)";
            const secondaryLabel = mode === "image" ? "Text Sim (25%)" : "Image Sim (25%)";
            const primaryScore = mode === "image" ? (item.image_score || 0) : (item.text_score || 0);
            const secondaryScore = mode === "image" ? (item.text_score || 0) : (item.image_score || 0);
            const ngramHtml = (mode === "text" && item.ngram_score !== undefined) ? `
                        <div class="score-row" style="color: #60a5fa; font-weight: 500;">
                            <span>N-Gram & Context Match:</span>
                            <span class="score-val" style="color: #60a5fa;">${(item.ngram_score * 100).toFixed(1)}%</span>
                        </div>
            ` : "";

            card.innerHTML = `
                <div class="card-img-wrapper">
                    <div class="rank-badge ${rankClass}">#${rank}</div>
                    <img src="${item.image_url}" alt="${escapeHtml(item.display_name || item.name)}" class="card-img" onerror="this.onerror=null;this.src='/static/img/placeholder.svg';">
                    <div class="score-pill-overlay">
                        <span class="score-dot"></span>
                        <span>${overallScorePct}% Match</span>
                    </div>
                </div>
                <div class="card-body">
                    <h3 class="card-title">${escapeHtml(item.display_name || item.name)}</h3>
                    <div class="card-tags">${tagsHtml}</div>
                    ${chunkHtml}
                    <div class="score-breakdown-box">
                        <div class="score-row">
                            <span>${primaryLabel}:</span>
                            <span class="score-val">${primaryScore.toFixed(4)}</span>
                        </div>
                        <div class="score-bar-bg">
                            <div class="score-bar-fill" style="width: ${Math.min(100, Math.max(5, primaryScore * 100))}%;"></div>
                        </div>
                        <div class="score-row">
                            <span>${secondaryLabel}:</span>
                            <span class="score-val">${secondaryScore.toFixed(4)}</span>
                        </div>
                        ${ngramHtml}
                        <div class="score-row" style="margin-top: 0.25rem; font-size: 0.72rem; color: var(--secondary);">
                            <span>Dot Product Blended Score:</span>
                            <span class="score-val" style="color: var(--secondary);">${item.score.toFixed(4)}</span>
                        </div>
                    </div>
                </div>
            `;

            resultsGrid.appendChild(card);
        });

        // Smooth scroll to results
        resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // --- Loading UI helper ---
    function setLoading(isLoading) {
        if (isLoading) {
            resultsLoading.classList.remove("hidden");
            resultsEmpty.classList.add("hidden");
            btnTextSearch.querySelector(".btn-spinner").classList.remove("hidden");
            btnTextSearch.querySelector(".btn-text").textContent = "Searching...";
            btnImageSearch.querySelector(".btn-spinner").classList.remove("hidden");
            btnImageSearch.querySelector(".btn-text").textContent = "Analyzing...";
        } else {
            resultsLoading.classList.add("hidden");
            btnTextSearch.querySelector(".btn-spinner").classList.add("hidden");
            btnTextSearch.querySelector(".btn-text").textContent = "Search Catalog";
            btnImageSearch.querySelector(".btn-spinner").classList.add("hidden");
            btnImageSearch.querySelector(".btn-text").textContent = "Search with This Image";
        }
    }

    // --- Fetch System & Vector DB Status ---
    async function updateStatus() {
        try {
            const resp = await fetch("/api/status");
            if (resp.ok) {
                const data = await resp.json();
                const stats = data.vector_stats || {};
                const imgCount = stats.image_count || 0;
                const chunkCount = stats.chunk_count || 0;

                indexedCountDisplay.textContent = imgCount.toLocaleString();
                statImageCount.textContent = imgCount.toLocaleString();
                statChunkCount.textContent = chunkCount.toLocaleString();
            }
        } catch (err) {
            console.warn("Status fetch note:", err);
        }
    }

    // Settings Modal
    openDatasetModalBtn.addEventListener("click", () => {
        settingsModal.classList.remove("hidden");
        updateStatus();
    });

    settingsCloseBtn.addEventListener("click", () => {
        settingsModal.classList.add("hidden");
    });

    settingsModal.addEventListener("click", (e) => {
        if (e.target === settingsModal) {
            settingsModal.classList.add("hidden");
        }
    });

    // Batch Ingestion Trigger
    btnTriggerIndexing.addEventListener("click", async () => {
        const limit = parseInt(document.getElementById("index-limit-select").value);
        const batchSize = parseInt(document.getElementById("index-batch-select").value);

        btnTriggerIndexing.querySelector(".btn-spinner").classList.remove("hidden");
        btnTriggerIndexing.querySelector(".btn-text").textContent = "Starting...";
        indexingFeedback.classList.remove("hidden");
        indexingFeedback.textContent = `Triggering ingestion for ${limit} items with batch_size=${batchSize}...`;

        try {
            const resp = await fetch("/api/index", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ limit: limit, batch_size: batchSize })
            });
            const data = await resp.json();
            indexingFeedback.textContent = data.message || "Indexing task started in background!";
            setTimeout(updateStatus, 3000);
        } catch (err) {
            indexingFeedback.textContent = "Error starting indexing: " + err.message;
        } finally {
            btnTriggerIndexing.querySelector(".btn-spinner").classList.add("hidden");
            btnTriggerIndexing.querySelector(".btn-text").textContent = "Start Batch Ingestion";
        }
    });

    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Initial status poll
    updateStatus();
    setInterval(updateStatus, 15000);
});
