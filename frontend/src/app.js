/**
 * Heritage AI — Monument Decay Monitoring & Preservation System
 * Frontend Client Application
 * Connects to live FastAPI backend at /api/v1
 */

// Dynamic API base to handle both FastAPI host (port 8008) and external servers (e.g. Live Server port 5500 or file://)
const API_BASE = (typeof window !== 'undefined' && window.location.origin && window.location.origin.includes(':8008'))
  ? ''
  : 'http://127.0.0.1:8008';

// State Management
const state = {
  currentFile: null,
  currentImageBitmap: null,
  analysisResult: null,
  viewMode: 'annotated', // 'annotated' | 'side-by-side' | 'raw'
  activeFilters: {
    Crack: true,
    Moss: true,
    Seepage: true
  },
  hoveredBox: null
};

// DOM Element References
const elements = {
  // Navigation Tabs
  tabButtons: document.querySelectorAll('.nav-tab, .tab-btn'),
  tabPanes: document.querySelectorAll('.tab-pane'),

  // System Status
  statusPill: document.getElementById('system-status-pill'),
  statusText: document.getElementById('system-status-text'),

  // Dropzone & Inputs
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('file-input'),
  browseBtn: document.getElementById('browse-btn'),
  sampleChips: document.querySelectorAll('.sample-btn, .sample-chip'),

  // Controls & Tuning
  sliderScore: document.getElementById('slider-score-thresh'),
  valScore: document.getElementById('val-score-thresh'),
  sliderNms: document.getElementById('slider-nms-thresh'),
  valNms: document.getElementById('val-nms-thresh'),
  checkClahe: document.getElementById('check-clahe'),
  btnRunInference: document.getElementById('btn-run-inference'),
  btnRunText: document.getElementById('btn-run-text'),

  // Visualizer Viewport
  canvas: document.getElementById('detection-canvas'),
  viewport: document.getElementById('viewport'),
  emptyState: document.getElementById('empty-state'),
  loadingOverlay: document.getElementById('loading-overlay'),
  viewToggles: document.querySelectorAll('.mode-btn, .view-mode-btn'),

  // Defect Filters
  filterCrack: document.getElementById('filter-crack'),
  filterMoss: document.getElementById('filter-moss'),
  filterSeepage: document.getElementById('filter-seepage'),

  // Metadata & Action Downloads
  downloadBar: document.getElementById('download-bar'),
  inspectedFilename: document.getElementById('inspected-filename'),
  inspectedDims: document.getElementById('inspected-dims'),
  btnDownloadImg: document.getElementById('btn-download-img'),
  btnDownloadMd: document.getElementById('btn-download-md'),
  btnDownloadJson: document.getElementById('btn-download-json'),

  // Risk Assessment Banner
  riskBadge: document.getElementById('risk-badge'),
  coverageVal: document.getElementById('coverage-val'),
  riskProgressFill: document.getElementById('risk-progress-fill'),
  riskRationaleText: document.getElementById('risk-rationale-text'),
  riskTimelineText: document.getElementById('risk-timeline-text'),

  // Defect Counts & Confidence
  countCrack: document.getElementById('count-crack'),
  confCrack: document.getElementById('conf-crack'),
  countMoss: document.getElementById('count-moss'),
  confMoss: document.getElementById('conf-moss'),
  countSeepage: document.getElementById('count-seepage'),
  confSeepage: document.getElementById('conf-seepage'),

  // Dedicated Preservation Recommendation Elements
  recObservedCondition: document.getElementById('rec-observed-condition'),
  recRecommendedAction: document.getElementById('rec-recommended-action'),
  recPriorityTimeline: document.getElementById('rec-priority-timeline'),
  recPreventiveMeasure: document.getElementById('rec-preventive-measure'),
  dynamicProtocolsList: document.getElementById('dynamic-protocols-list'),
  authorityBadge: document.getElementById('authority-badge')
};

// Vibrant High-Contrast Defect Color Palette
const CLASS_COLORS = {
  Crack: { stroke: '#ff4d4f', fill: 'rgba(255, 77, 79, 0.22)', tagBg: '#ff4d4f' },
  Moss: { stroke: '#10b981', fill: 'rgba(16, 185, 129, 0.22)', tagBg: '#10b981' },
  Seepage: { stroke: '#00d2ff', fill: 'rgba(0, 210, 255, 0.22)', tagBg: '#00d2ff' }
};

// ==============================================================================
// 1. Initialization & Backend Health
// ==============================================================================

async function initApp() {
  setupNavigationTabs();
  setupDropzone();
  setupSliders();
  setupFilters();
  setupViewToggles();
  setupActionButtons();
  setupSampleButtons();

  await checkBackendHealth();
}

async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const dev = (data.model && data.model.device) ? data.model.device.toUpperCase() : 'CPU';
    elements.statusText.textContent = `Model Ready (${dev})`;
    elements.statusPill.classList.add('online');
  } catch (err) {
    console.warn('Backend connection issue:', err);
    elements.statusText.textContent = 'API Offline (Port 8008)';
    elements.statusPill.classList.remove('online');
  }
}

// ==============================================================================
// 2. Navigation Tabs
// ==============================================================================

function setupNavigationTabs() {
  elements.tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTabId = btn.dataset.tab;
      elements.tabButtons.forEach(b => b.classList.remove('active'));
      elements.tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPane = document.getElementById(targetTabId);
      if (targetPane) {
        targetPane.classList.add('active');
      }
    });
  });
}

// ==============================================================================
// 3. Dropzone & File Handling
// ==============================================================================

function setupDropzone() {
  if (elements.browseBtn) {
    elements.browseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      elements.fileInput.click();
    });
  }

  if (elements.dropzone) {
    elements.dropzone.addEventListener('click', () => {
      elements.fileInput.click();
    });

    ['dragenter', 'dragover'].forEach(name => {
      elements.dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        elements.dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      elements.dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        elements.dropzone.classList.remove('dragover');
      });
    });

    elements.dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files[0]) {
        handleFileSelected(dt.files[0]);
      }
    });
  }

  if (elements.fileInput) {
    elements.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelected(e.target.files[0]);
      }
    });
  }
}

function setupSampleButtons() {
  elements.sampleChips.forEach(chip => {
    chip.addEventListener('click', async () => {
      const sampleUrl = chip.dataset.sample;
      try {
        if (elements.loadingOverlay) elements.loadingOverlay.classList.add('active');
        const res = await fetch(sampleUrl);
        const blob = await res.blob();
        const filename = sampleUrl.split('/').pop();
        const file = new File([blob], filename, { type: 'image/jpeg' });
        await handleFileSelected(file);
        await runAnalysis();
      } catch (err) {
        console.error('Failed to load sample image:', err);
        alert('Could not load sample image: ' + err.message);
      } finally {
        if (elements.loadingOverlay) elements.loadingOverlay.classList.remove('active');
      }
    });
  });
}

async function handleFileSelected(file) {
  state.currentFile = file;
  if (elements.inspectedFilename) elements.inspectedFilename.textContent = file.name;
  
  // Load into Image bitmap for canvas rendering
  const imgUrl = URL.createObjectURL(file);
  const img = new Image();
  img.src = imgUrl;
  await new Promise(resolve => {
    img.onload = () => {
      state.currentImageBitmap = img;
      if (elements.inspectedDims) elements.inspectedDims.textContent = `${img.naturalWidth} x ${img.naturalHeight} px`;
      if (elements.emptyState) elements.emptyState.style.display = 'none';
      drawCanvas();
      resolve();
    };
  });
}

// ==============================================================================
// 4. Sliders, Filters & View Toggles
// ==============================================================================

function setupSliders() {
  if (elements.sliderScore && elements.valScore) {
    elements.sliderScore.addEventListener('input', (e) => {
      elements.valScore.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }
  if (elements.sliderNms && elements.valNms) {
    elements.sliderNms.addEventListener('input', (e) => {
      elements.valNms.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }
}

function setupFilters() {
  const filterInputs = [
    { el: elements.filterCrack, cls: 'Crack' },
    { el: elements.filterMoss, cls: 'Moss' },
    { el: elements.filterSeepage, cls: 'Seepage' }
  ];

  filterInputs.forEach(({ el, cls }) => {
    if (!el) return;
    el.addEventListener('change', () => {
      state.activeFilters[cls] = el.checked;
      el.parentElement.classList.toggle('active', el.checked);
      drawCanvas();
    });
  });
}

function setupViewToggles() {
  elements.viewToggles.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.viewToggles.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.viewMode = btn.dataset.view;
      drawCanvas();
    });
  });
}

// ==============================================================================
// 5. Run Inference Analysis API Call
// ==============================================================================

if (elements.btnRunInference) {
  elements.btnRunInference.addEventListener('click', runAnalysis);
}

async function runAnalysis() {
  if (!state.currentFile) {
    alert('Please select or upload a monument image first.');
    return;
  }

  const formData = new FormData();
  formData.append('file', state.currentFile);

  const scoreThresh = elements.sliderScore ? elements.sliderScore.value : 0.25;
  const nmsThresh = elements.sliderNms ? elements.sliderNms.value : 0.45;
  const enableEnhance = elements.checkClahe ? elements.checkClahe.checked : true;

  const url = `${API_BASE}/api/v1/analyze?score_thresh=${scoreThresh}&nms_thresh=${nmsThresh}&enable_enhancement=${enableEnhance}`;

  try {
    if (elements.loadingOverlay) elements.loadingOverlay.classList.add('active');
    if (elements.btnRunText) elements.btnRunText.textContent = 'Analyzing Monument...';

    const response = await fetch(url, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      throw new Error(errJson.detail || `Server error: ${response.status}`);
    }

    const data = await response.json();
    state.analysisResult = data;

    updateMetricsAndRiskUI(data);
    updatePreservationSection(data);
    drawCanvas();

  } catch (err) {
    console.error('Inference error:', err);
    alert('Monument inspection error: ' + err.message);
  } finally {
    if (elements.loadingOverlay) elements.loadingOverlay.classList.remove('active');
    if (elements.btnRunText) elements.btnRunText.textContent = 'Run Defect Inspection';
  }
}

// ==============================================================================
// 6. UI Updates: Risk Assessment & Defect Statistics
// ==============================================================================

function updateMetricsAndRiskUI(data) {
  const ra = data.risk_assessment;
  if (!ra) return;

  const riskLvl = (ra.risk_level || 'LOW').toUpperCase();

  // 1. Risk Status Badge
  if (elements.riskBadge) {
    elements.riskBadge.textContent = `${riskLvl} RISK`;
    elements.riskBadge.className = 'risk-pill';
    
    if (riskLvl === 'HIGH') {
      elements.riskBadge.classList.add('badge-high');
      if (elements.riskProgressFill) elements.riskProgressFill.style.backgroundColor = 'var(--risk-high)';
    } else if (riskLvl === 'MEDIUM' || riskLvl === 'MODERATE') {
      elements.riskBadge.classList.add('badge-medium');
      if (elements.riskProgressFill) elements.riskProgressFill.style.backgroundColor = 'var(--risk-med)';
    } else {
      elements.riskBadge.classList.add('badge-low');
      if (elements.riskProgressFill) elements.riskProgressFill.style.backgroundColor = 'var(--risk-low)';
    }
  }

  // 2. Coverage Statistic & Progress Bar
  const cov = typeof ra.coverage_percentage === 'number' ? ra.coverage_percentage : 0;
  if (elements.coverageVal) elements.coverageVal.textContent = `${cov.toFixed(1)}%`;
  if (elements.riskProgressFill) elements.riskProgressFill.style.width = `${Math.min(100, cov)}%`;

  // 3. Rationale & Timeline Text
  if (elements.riskRationaleText) elements.riskRationaleText.textContent = ra.risk_rationale || ra.description || 'Structural condition verified.';
  if (elements.riskTimelineText) elements.riskTimelineText.textContent = ra.action_timeline || 'Standard periodic monitoring.';

  // 4. Defect Breakdown Counts & Max Confidences
  const counts = ra.defect_counts || {};
  const confs = ra.defect_max_confidences || {};

  if (elements.countCrack) elements.countCrack.textContent = counts['Crack'] || 0;
  if (elements.confCrack) elements.confCrack.textContent = (confs['Crack'] || 0.0).toFixed(2);

  if (elements.countMoss) elements.countMoss.textContent = counts['Moss'] || 0;
  if (elements.confMoss) elements.confMoss.textContent = (confs['Moss'] || 0.0).toFixed(2);

  if (elements.countSeepage) elements.countSeepage.textContent = counts['Seepage'] || 0;
  if (elements.confSeepage) elements.confSeepage.textContent = (confs['Seepage'] || 0.0).toFixed(2);
}

// ==============================================================================
// 7. UI Updates: Dedicated Preservation Recommendation Section
// ==============================================================================

function updatePreservationSection(data) {
  const plan = data.preservation_plan;
  const ra = data.risk_assessment;

  if (!plan) return;

  const summary = plan.preservation_summary || {};
  const protocols = plan.recommended_protocols || [];

  // Pillar 1: Observed Condition
  if (elements.recObservedCondition) {
    const totalDetections = data.total_detections || 0;
    const primaryConcern = summary.primary_concern || 'None Identified';
    const condition = summary.monument_condition || ra.risk_level || 'Normal';
    
    elements.recObservedCondition.innerHTML = `
      <p><strong>Condition Level:</strong> <span class="text-${primaryConcern.toLowerCase()}">${condition}</span></p>
      <p><strong>Primary Concern:</strong> ${primaryConcern}</p>
      <p><strong>Total Defect Regions:</strong> ${totalDetections} detected across surface.</p>
      <p style="margin-top: 4px; font-size: 0.76rem; color: var(--text-muted);">${ra.risk_rationale || ''}</p>
    `;
  }

  // Pillar 2: Recommended Action
  if (elements.recRecommendedAction) {
    const actionText = summary.recommended_action_summary || 'Standard non-invasive monitoring and dry seasonal inspection.';
    elements.recRecommendedAction.innerHTML = `
      <p>${actionText}</p>
    `;
  }

  // Pillar 3: Priority & Timeline
  if (elements.recPriorityTimeline) {
    const timeline = ra.action_timeline || 'Periodic seasonal survey';
    const urgency = ra.risk_level === 'HIGH' ? 'Immediate / High Priority' : (ra.risk_level === 'MEDIUM' ? 'Medium Priority' : 'Standard Routine');
    
    elements.recPriorityTimeline.innerHTML = `
      <p><strong>Urgency Rating:</strong> ${urgency}</p>
      <p><strong>Intervention Window:</strong> ${timeline}</p>
      <p style="margin-top: 4px; font-size: 0.74rem; color: var(--text-muted);">Supervising: ${summary.supervising_authority || 'ASI / State Archaeology'}</p>
    `;
  }

  // Pillar 4: Preventive Measures & Monitoring
  if (elements.recPreventiveMeasure) {
    let contraindicationSnippet = 'Avoid Ordinary Portland Cement and impermeable polymer resin coatings on heritage stone.';
    if (protocols.length > 0 && protocols[0].contraindications && protocols[0].contraindications.length > 0) {
      contraindicationSnippet = protocols[0].contraindications[0];
    }
    
    elements.recPreventiveMeasure.innerHTML = `
      <p><strong>Preventive Protocol:</strong> Implement environmental moisture barriers and clean drainage cornices.</p>
      <p style="margin-top: 6px; font-size: 0.76rem; color: var(--color-crack);"><strong>Warning:</strong> ${contraindicationSnippet}</p>
    `;
  }

  // Dynamic Detailed Protocol Cards
  if (elements.dynamicProtocolsList) {
    if (protocols.length === 0) {
      elements.dynamicProtocolsList.innerHTML = `
        <div class="active-protocol-item">
          <h4 class="active-protocol-title">Baseline Conservation Protocol</h4>
          <p style="font-size: 0.82rem; color: var(--text-secondary); margin-top: 4px;">
            No critical structural fractures or active biocolonization were localized above threshold. Maintain periodic cyclic inspections and monitor microclimate moisture.
          </p>
        </div>
      `;
      return;
    }

    elements.dynamicProtocolsList.innerHTML = protocols.map(p => {
      const defectType = p.defect_type || 'Defect';
      const themeClass = `border-${defectType.toLowerCase()}`;
      const diagList = (p.diagnostic_steps || []).map(s => `<li>${s}</li>`).join('');
      const treatList = (p.treatment_steps || []).map(s => `<li>${s}</li>`).join('');
      const contraList = (p.contraindications || []).join(' ');

      return `
        <div class="active-protocol-item ${themeClass}">
          <div class="active-protocol-header">
            <div>
              <span class="sub-caption">${p.urgency || 'Standard'} Urgency • ${p.count || 0} Regions</span>
              <h4 class="active-protocol-title">${p.summary || defectType}</h4>
            </div>
            <span class="header-pill">Max Conf: ${(p.max_confidence || 0).toFixed(2)}</span>
          </div>

          <div class="protocol-steps-grid">
            <div class="protocol-steps-col">
              <h5>Diagnostic Procedures</h5>
              <ul>${diagList}</ul>
            </div>
            <div class="protocol-steps-col">
              <h5>Intervention & Treatment</h5>
              <ul>${treatList}</ul>
            </div>
          </div>

          ${contraList ? `
            <div class="warning-callout" style="margin-top: 12px;">
              <div class="warning-head">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon>
                </svg>
                <span>Contraindication</span>
              </div>
              <p>${contraList}</p>
            </div>
          ` : ''}
        </div>
      `;
    }).join('');
  }
}

// ==============================================================================
// 8. Interactive Canvas Drawing
// ==============================================================================

function drawCanvas() {
  const canvas = elements.canvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const img = state.currentImageBitmap;

  if (!img) return;

  const width = img.naturalWidth || img.width;
  const height = img.naturalHeight || img.height;

  canvas.width = width;
  canvas.height = height;

  // Clear canvas
  ctx.clearRect(0, 0, width, height);

  // 1. Draw Original Image
  ctx.drawImage(img, 0, 0, width, height);

  // If Raw view mode, stop here
  if (state.viewMode === 'raw') return;

  // If Side-by-Side view mode, draw clean divider
  if (state.viewMode === 'side-by-side') {
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = Math.max(3, width / 400);
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.stroke();

    // Side-by-side header label banners
    const badgeW = Math.max(130, Math.round(width / 8));
    const badgeH = Math.max(28, Math.round(height / 45));

    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.fillRect(12, 12, badgeW, badgeH);
    ctx.fillStyle = '#f8fafc';
    ctx.font = `700 ${Math.max(12, Math.round(width / 160))}px Plus Jakarta Sans, sans-serif`;
    ctx.fillText('RAW ORIGINAL', 22, 12 + badgeH * 0.68);

    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.fillRect(width / 2 + 12, 12, badgeW + 16, badgeH);
    ctx.fillStyle = '#38bdf8';
    ctx.fillText('AI DETECTIONS', width / 2 + 22, 12 + badgeH * 0.68);
  }

  // If no detections yet, return
  if (!state.analysisResult || !state.analysisResult.detections) return;

  const detections = state.analysisResult.detections;
  const lineWidth = Math.max(2, Math.round(Math.min(width, height) / 340));

  detections.forEach(det => {
    const cls = det.class_name;
    if (!state.activeFilters[cls]) return; // filtered out by user

    const [x1, y1, x2, y2] = det.box;
    const boxW = x2 - x1;
    const boxH = y2 - y1;

    // For side-by-side, only draw detections on the right half
    if (state.viewMode === 'side-by-side' && x2 < width / 2) {
      return;
    }

    const colorConfig = CLASS_COLORS[cls] || { stroke: '#dc2626', fill: 'rgba(220, 38, 38, 0.16)', tagBg: '#dc2626' };

    // Bounding box fill & outline
    ctx.fillStyle = colorConfig.fill;
    ctx.fillRect(x1, y1, boxW, boxH);

    ctx.strokeStyle = colorConfig.stroke;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x1, y1, boxW, boxH);

    // Label banner above box
    const label = `${cls} ${(det.confidence * 100).toFixed(0)}%`;
    const fontSize = Math.max(12, Math.round(width / 160));
    ctx.font = `700 ${fontSize}px Plus Jakarta Sans, sans-serif`;
    const textMetrics = ctx.measureText(label);
    const tagH = Math.max(18, Math.round(width / 120));
    const tagW = textMetrics.width + 12;

    const tagY = Math.max(0, y1 - tagH);
    ctx.fillStyle = colorConfig.tagBg;
    ctx.fillRect(x1, tagY, tagW, tagH);

    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, x1 + 6, tagY + tagH * 0.72);
  });
}

// ==============================================================================
// 9. Report & Asset Downloads
// ==============================================================================

function setupActionButtons() {
  // 1. Download High-Res Annotated Image
  if (elements.btnDownloadImg) {
    elements.btnDownloadImg.addEventListener('click', async () => {
      if (!state.currentFile) return;
      const formData = new FormData();
      formData.append('file', state.currentFile);

      const scoreThresh = elements.sliderScore ? elements.sliderScore.value : 0.25;
      const nmsThresh = elements.sliderNms ? elements.sliderNms.value : 0.45;
      const enableEnhance = elements.checkClahe ? elements.checkClahe.checked : true;

      const url = `${API_BASE}/api/v1/analyze/visualize?score_thresh=${scoreThresh}&nms_thresh=${nmsThresh}&enable_enhancement=${enableEnhance}`;

      try {
        const res = await fetch(url, { method: 'POST', body: formData });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `annotated_${state.currentFile.name}`;
        a.click();
      } catch (err) {
        alert('Annotated image download error: ' + err.message);
      }
    });
  }

  // 2. Download Markdown Conservation Report
  if (elements.btnDownloadMd) {
    elements.btnDownloadMd.addEventListener('click', async () => {
      if (!state.currentFile) return;
      const formData = new FormData();
      formData.append('file', state.currentFile);

      const url = `${API_BASE}/api/v1/analyze/report?format=markdown`;
      try {
        const res = await fetch(url, { method: 'POST', body: formData });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const mdText = await res.text();
        const blob = new Blob([mdText], { type: 'text/markdown' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `ASI_Conservation_Protocol_${state.currentFile.name.replace(/\.[^/.]+$/, "")}.md`;
        a.click();
      } catch (err) {
        alert('Report download error: ' + err.message);
      }
    });
  }

  // 3. Export Raw JSON Data
  if (elements.btnDownloadJson) {
    elements.btnDownloadJson.addEventListener('click', () => {
      if (!state.analysisResult) {
        alert('No inspection results to export yet.');
        return;
      }
      const jsonStr = JSON.stringify(state.analysisResult, null, 2);
      const blob = new Blob([jsonStr], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `decay_inspection_${state.currentFile ? state.currentFile.name.replace(/\.[^/.]+$/, "") : "data"}.json`;
      a.click();
    });
  }
}

// Start application when DOM is ready
document.addEventListener('DOMContentLoaded', initApp);
