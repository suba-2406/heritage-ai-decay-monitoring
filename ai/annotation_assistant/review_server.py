#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - Human Review Web Server
Provides an interactive, local, zero-crash web UI for inspecting AI proposals,
modifying/drawing bounding boxes, marking images as Normal, and exporting Pascal VOC XML.

Usage:
    python review_server.py [--port 5500] [--host 127.0.0.1]
"""

import os
import sys
import json
import csv
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template_string
from typing import Dict, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from annotation_schema import BoundingBox, ImageAnnotationRecord, ReviewStatus, CLASS_MAP
from export_pascal_voc import export_record_to_voc_xml

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_ROOT = os.environ.get("DATASET_ROOT", r"c:\Users\subaj\OneDrive\Desktop\intern\TEMPLE")
MANIFEST_PATH = os.path.join(BASE_DIR, "ai", "dataset", "annotation_manifest.json")
PROPOSALS_DIR = os.path.join(BASE_DIR, "ai", "annotations", "proposals")
REVIEW_DIR = os.path.join(BASE_DIR, "ai", "annotations", "review")
RAW_XML_DIR = os.path.join(BASE_DIR, "ai", "annotations", "raw")
FLAGGED_CSV = os.path.join(BASE_DIR, "ai", "annotations", "flagged_for_review.csv")

os.makedirs(PROPOSALS_DIR, exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)
os.makedirs(RAW_XML_DIR, exist_ok=True)

def get_image_catalog():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("images", [])
    return []

# HTML UI Template
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Heritage AI — Monument Decay Annotation Reviewer</title>
<style>
  :root {
    --bg-dark: #0f172a;
    --card-bg: #1e293b;
    --card-border: #334155;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
    --crack-color: #ef4444;
    --moss-color: #22c55e;
    --seepage-color: #0284c7;
    --normal-color: #10b981;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg-dark);
    color: var(--text-main);
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }
  header {
    background: var(--card-bg);
    border-bottom: 1px solid var(--card-border);
    padding: 10px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .title-group h1 { font-size: 1.15rem; font-weight: 700; color: #fff; }
  .title-group p { font-size: 0.78rem; color: var(--text-muted); }
  .stats-bar { display: flex; gap: 15px; font-size: 0.82rem; }
  .stat-badge {
    background: #0f172a;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid var(--card-border);
  }
  .stat-badge strong { color: var(--accent); }

  .main-container {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* Sidebar */
  .sidebar {
    width: 320px;
    background: var(--card-bg);
    border-right: 1px solid var(--card-border);
    display: flex;
    flex-direction: column;
  }
  .sidebar-filter {
    padding: 10px;
    border-bottom: 1px solid var(--card-border);
    display: flex;
    gap: 6px;
  }
  .filter-btn {
    flex: 1;
    padding: 6px;
    font-size: 0.75rem;
    background: #0f172a;
    border: 1px solid var(--card-border);
    color: var(--text-muted);
    border-radius: 4px;
    cursor: pointer;
  }
  .filter-btn.active {
    background: var(--accent);
    color: #0f172a;
    font-weight: 600;
  }
  .image-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }
  .image-item {
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 4px;
    cursor: pointer;
    border: 1px solid transparent;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
  }
  .image-item:hover { background: #334155; }
  .image-item.active { background: #1e3a5f; border-color: var(--accent); }
  .status-tag {
    font-size: 0.65rem;
    padding: 2px 6px;
    border-radius: 4px;
    text-transform: uppercase;
    font-weight: 600;
  }
  .tag-review { background: #7c2d12; color: #fdba74; }
  .tag-proposed-normal { background: #064e3b; color: #6ee7b7; }
  .tag-verified-defect { background: #1e3a8a; color: #93c5fd; }
  .tag-verified-normal { background: #047857; color: #a7f3d0; }
  .tag-flagged { background: #831843; color: #fbcfe8; }

  /* Workspace */
  .workspace {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: #090d16;
    position: relative;
  }
  .toolbar {
    background: var(--card-bg);
    border-bottom: 1px solid var(--card-border);
    padding: 8px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
  }
  .tool-group { display: flex; gap: 8px; align-items: center; }
  .btn {
    padding: 6px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    border-radius: 6px;
    cursor: pointer;
    border: 1px solid transparent;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .btn-primary { background: var(--accent); color: #0f172a; }
  .btn-primary:hover { filter: brightness(1.1); }
  .btn-success { background: var(--normal-color); color: #fff; }
  .btn-danger { background: var(--crack-color); color: #fff; }
  .btn-warning { background: #d97706; color: #fff; }
  .btn-secondary { background: #334155; color: var(--text-main); border-color: var(--card-border); }
  .btn-secondary:hover { background: #475569; }

  /* Canvas Area */
  .canvas-container {
    flex: 1;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  #annotationCanvas {
    cursor: crosshair;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  }

  /* Right Inspection Panel */
  .control-panel {
    width: 300px;
    background: var(--card-bg);
    border-left: 1px solid var(--card-border);
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    overflow-y: auto;
  }
  .panel-section {
    background: #0f172a;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 12px;
  }
  .panel-section h3 {
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .box-list { display: flex; flex-direction: column; gap: 6px; max-height: 250px; overflow-y: auto; }
  .box-row {
    padding: 6px 10px;
    background: #1e293b;
    border-radius: 6px;
    font-size: 0.78rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-left: 4px solid var(--crack-color);
  }
  .box-row.selected { outline: 2px solid var(--accent); }
  .box-actions { display: flex; gap: 6px; }
  .btn-tiny {
    padding: 2px 6px;
    font-size: 0.7rem;
    border-radius: 4px;
    cursor: pointer;
    background: #334155;
    color: #fff;
    border: none;
  }
  .btn-tiny:hover { background: #475569; }

  .kbd-shortcuts { font-size: 0.75rem; color: var(--text-muted); line-height: 1.5; }
  .kbd-shortcuts kbd {
    background: #334155;
    padding: 2px 5px;
    border-radius: 4px;
    color: #fff;
    font-family: monospace;
  }
</style>
</head>
<body>

<header>
  <div class="title-group">
    <h1>Heritage AI — Monument Decay Annotation Reviewer</h1>
    <p>Human-in-the-Loop Verification for SSD Object Detection &bull; Work II</p>
  </div>
  <div class="stats-bar">
    <div class="stat-badge">Total Images: <strong id="statTotal">307</strong></div>
    <div class="stat-badge">Verified Defects: <strong id="statDefects">0</strong></div>
    <div class="stat-badge">Verified Normal: <strong id="statNormal">0</strong></div>
    <div class="stat-badge">Pending Review: <strong id="statPending">307</strong></div>
  </div>
</header>

<div class="main-container">
  <!-- Left Sidebar -->
  <aside class="sidebar">
    <div class="sidebar-filter">
      <button class="filter-btn active" onclick="setFilter('all')">All (307)</button>
      <button class="filter-btn" onclick="setFilter('pending')">Pending</button>
      <button class="filter-btn" onclick="setFilter('verified')">Verified</button>
      <button class="filter-btn" onclick="setFilter('flagged')">Flagged</button>
    </div>
    <div class="image-list" id="imageList">
      <!-- Image items populated via JS -->
    </div>
  </aside>

  <!-- Workspace -->
  <main class="workspace">
    <div class="toolbar">
      <div class="tool-group">
        <button class="btn btn-secondary" onclick="prevImage()" title="Shortcut: Left Arrow">&larr; Prev</button>
        <span id="currentImageName" style="font-size:0.85rem; font-weight:600;">Loading...</span>
        <button class="btn btn-secondary" onclick="nextImage()" title="Shortcut: Right Arrow">Next &rarr;</button>
      </div>

      <div class="tool-group">
        <label style="font-size:0.8rem; color:var(--text-muted);">Active Label:</label>
        <select id="activeClassSelect" style="background:#0f172a; color:#fff; border:1px solid var(--card-border); padding:5px 8px; border-radius:6px; font-size:0.8rem;">
          <option value="Crack">Crack (ID 1)</option>
          <option value="Moss">Moss (ID 2)</option>
          <option value="Seepage">Seepage (ID 3)</option>
        </select>
        <button class="btn btn-danger" onclick="deleteSelectedBox()" id="btnDeleteBox" disabled>Delete Box</button>
      </div>

      <div class="tool-group">
        <button class="btn btn-warning" onclick="flagForReview()" title="Shortcut: F">Flag Review</button>
        <button class="btn btn-success" onclick="markAsNormal()" title="Shortcut: N">Mark Normal (0 boxes)</button>
        <button class="btn btn-primary" onclick="saveAndNext()" title="Shortcut: S or D">Save & Next</button>
      </div>
    </div>

    <div class="canvas-container" id="canvasContainer">
      <canvas id="annotationCanvas"></canvas>
    </div>
  </main>

  <!-- Right Info Panel -->
  <aside class="control-panel">
    <div class="panel-section">
      <h3>Image Info</h3>
      <div style="font-size:0.78rem; line-height:1.6; color:var(--text-muted);">
        <div><strong>Site:</strong> <span id="infoSite">-</span></div>
        <div><strong>Resolution:</strong> <span id="infoDim">-</span></div>
        <div><strong>Status:</strong> <span id="infoStatus">-</span></div>
      </div>
    </div>

    <div class="panel-section">
      <h3>Bounding Boxes (<span id="boxCount">0</span>)</h3>
      <div class="box-list" id="boxList">
        <p style="font-size:0.75rem; color:var(--text-muted); text-align:center;">No boxes present.</p>
      </div>
    </div>

    <div class="panel-section">
      <h3>Keyboard Shortcuts</h3>
      <div class="kbd-shortcuts">
        <div><kbd>Drag</kbd> Draw new bounding box</div>
        <div><kbd>Click Box</kbd> Select to edit / move</div>
        <div><kbd>Delete</kbd> Delete selected box</div>
        <div><kbd>N</kbd> Mark as Normal (0 boxes)</div>
        <div><kbd>S</kbd> or <kbd>D</kbd> Save verification & Next</div>
        <div><kbd>F</kbd> Flag for manual review</div>
        <div><kbd>&larr;</kbd> / <kbd>&rarr;</kbd> Navigate images</div>
      </div>
    </div>
  </aside>
</div>

<script>
let images = [];
let currentIndex = 0;
let currentAnnotation = null;
let currentImageObj = new Image();
let selectedBoxIndex = -1;
let filterMode = 'all';

// Drawing state
let isDrawing = false;
let startX = 0, startY = 0;
let currentX = 0, currentY = 0;

// Canvas & sizing
const canvas = document.getElementById('annotationCanvas');
const ctx = canvas.getContext('2d');
const container = document.getElementById('canvasContainer');

const CLASS_COLORS = {
  'Crack': '#ef4444',
  'Moss': '#22c55e',
  'Seepage': '#0284c7',
  'Normal': '#10b981'
};

async function init() {
  await fetchImages();
  await updateStats();
  if (images.length > 0) {
    loadImage(0);
  }
}

async function fetchImages() {
  const res = await fetch('/api/images');
  images = await res.json();
  renderImageList();
}

async function updateStats() {
  const res = await fetch('/api/stats');
  const s = await res.json();
  document.getElementById('statTotal').innerText = s.total;
  document.getElementById('statDefects').innerText = s.verified_defects;
  document.getElementById('statNormal').innerText = s.verified_normal;
  document.getElementById('statPending').innerText = s.pending;
}

function setFilter(mode) {
  filterMode = mode;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  renderImageList();
}

function renderImageList() {
  const list = document.getElementById('imageList');
  list.innerHTML = '';

  const filtered = images.filter((img, idx) => {
    img._originalIndex = idx;
    if (filterMode === 'all') return true;
    if (filterMode === 'pending') return img.status === 'UNREVIEWED' || img.status === 'REVIEW_REQUIRED' || img.status === 'PROPOSED_NORMAL';
    if (filterMode === 'verified') return img.status === 'VERIFIED_DEFECT' || img.status === 'VERIFIED_NORMAL';
    if (filterMode === 'flagged') return img.status === 'FLAGGED_AMBIGUOUS';
    return true;
  });

  filtered.forEach(img => {
    const div = document.createElement('div');
    div.className = `image-item ${img._originalIndex === currentIndex ? 'active' : ''}`;
    div.onclick = () => loadImage(img._originalIndex);

    let tagClass = 'tag-review';
    if (img.status === 'VERIFIED_DEFECT') tagClass = 'tag-verified-defect';
    else if (img.status === 'VERIFIED_NORMAL') tagClass = 'tag-verified-normal';
    else if (img.status === 'PROPOSED_NORMAL') tagClass = 'tag-proposed-normal';
    else if (img.status === 'FLAGGED_AMBIGUOUS') tagClass = 'tag-flagged';

    div.innerHTML = `
      <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:180px;">${img.filename}</span>
      <span class="status-tag ${tagClass}">${img.status.replace('_', ' ')}</span>
    `;
    list.appendChild(div);
  });
}

async function loadImage(index) {
  if (index < 0 || index >= images.length) return;
  currentIndex = index;
  renderImageList();

  const imgInfo = images[currentIndex];
  document.getElementById('currentImageName').innerText = `[${currentIndex + 1}/${images.length}] ${imgInfo.filename}`;
  document.getElementById('infoSite').innerText = imgInfo.site;
  document.getElementById('infoDim').innerText = `${imgInfo.width} x ${imgInfo.height}`;
  document.getElementById('infoStatus').innerText = imgInfo.status;

  // Fetch annotation / proposal
  const annRes = await fetch(`/api/annotation/${encodeURIComponent(imgInfo.filename)}`);
  currentAnnotation = await annRes.json();
  selectedBoxIndex = -1;
  document.getElementById('btnDeleteBox').disabled = true;

  // Load image into canvas
  currentImageObj = new Image();
  currentImageObj.src = `/api/image/${encodeURIComponent(imgInfo.rel_path)}`;
  currentImageObj.onload = () => {
    resizeCanvas();
    draw();
    renderBoxList();
  };
}

function resizeCanvas() {
  if (!currentImageObj.width) return;
  const maxW = container.clientWidth - 40;
  const maxH = container.clientHeight - 40;
  const scale = Math.min(maxW / currentImageObj.width, maxH / currentImageObj.height, 1.0);

  canvas.width = currentImageObj.width * scale;
  canvas.height = currentImageObj.height * scale;
  canvas._scale = scale;
}

window.addEventListener('resize', () => {
  resizeCanvas();
  draw();
});

function draw() {
  if (!ctx || !currentImageObj.width) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(currentImageObj, 0, 0, canvas.width, canvas.height);

  const scale = canvas._scale || 1.0;

  // Draw bounding boxes
  if (currentAnnotation && currentAnnotation.boxes) {
    currentAnnotation.boxes.forEach((b, idx) => {
      const x = b.xmin * scale;
      const y = b.ymin * scale;
      const w = (b.xmax - b.xmin) * scale;
      const h = (b.ymax - b.ymin) * scale;

      const color = CLASS_COLORS[b.label] || '#38bdf8';
      ctx.lineWidth = idx === selectedBoxIndex ? 3 : 2;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);

      ctx.fillStyle = color;
      ctx.globalAlpha = 0.15;
      ctx.fillRect(x, y, w, h);
      ctx.globalAlpha = 1.0;

      // Label badge
      ctx.fillStyle = color;
      ctx.font = '12px sans-serif';
      const labelText = `${b.label} (${b.confidence ? Math.round(b.confidence * 100) + '%' : '100%'})`;
      const textW = ctx.measureText(labelText).width;
      ctx.fillRect(x, Math.max(0, y - 18), textW + 8, 18);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x + 4, Math.max(14, y - 4));
    });
  }

  // Draw active drawing box
  if (isDrawing) {
    const rx = Math.min(startX, currentX);
    const ry = Math.min(startY, currentY);
    const rw = Math.abs(currentX - startX);
    const rh = Math.abs(currentY - startY);

    const activeCls = document.getElementById('activeClassSelect').value;
    ctx.strokeStyle = CLASS_COLORS[activeCls] || '#fff';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.setLineDash([]);
  }
}

// Mouse events on canvas
canvas.addEventListener('mousedown', (e) => {
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  const scale = canvas._scale || 1.0;

  // Check if clicked inside existing box to select it
  let clickedBox = -1;
  if (currentAnnotation && currentAnnotation.boxes) {
    for (let i = currentAnnotation.boxes.length - 1; i >= 0; i--) {
      const b = currentAnnotation.boxes[i];
      if (mouseX >= b.xmin * scale && mouseX <= b.xmax * scale &&
          mouseY >= b.ymin * scale && mouseY <= b.ymax * scale) {
        clickedBox = i;
        break;
      }
    }
  }

  if (clickedBox !== -1 && !e.shiftKey) {
    selectedBoxIndex = clickedBox;
    document.getElementById('btnDeleteBox').disabled = false;
    document.getElementById('activeClassSelect').value = currentAnnotation.boxes[selectedBoxIndex].label;
    renderBoxList();
    draw();
  } else {
    // Start drawing new box
    selectedBoxIndex = -1;
    document.getElementById('btnDeleteBox').disabled = true;
    renderBoxList();
    isDrawing = true;
    startX = mouseX;
    startY = mouseY;
    currentX = mouseX;
    currentY = mouseY;
  }
});

canvas.addEventListener('mousemove', (e) => {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  currentX = e.clientX - rect.left;
  currentY = e.clientY - rect.top;
  draw();
});

canvas.addEventListener('mouseup', () => {
  if (!isDrawing) return;
  isDrawing = false;
  const scale = canvas._scale || 1.0;

  const rx = Math.min(startX, currentX) / scale;
  const ry = Math.min(startY, currentY) / scale;
  const rw = Math.abs(currentX - startX) / scale;
  const rh = Math.abs(currentY - startY) / scale;

  if (rw > 15 && rh > 15) {
    const activeLabel = document.getElementById('activeClassSelect').value;
    const newBox = {
      xmin: Math.round(rx),
      ymin: Math.round(ry),
      xmax: Math.round(rx + rw),
      ymax: Math.round(ry + rh),
      label: activeLabel,
      class_id: activeLabel === 'Crack' ? 1 : (activeLabel === 'Moss' ? 2 : 3),
      confidence: 1.0,
      source: 'MANUAL_DRAWN'
    };
    if (!currentAnnotation.boxes) currentAnnotation.boxes = [];
    currentAnnotation.boxes.push(newBox);
    selectedBoxIndex = currentAnnotation.boxes.length - 1;
    document.getElementById('btnDeleteBox').disabled = false;
    renderBoxList();
  }
  draw();
});

function renderBoxList() {
  const boxList = document.getElementById('boxList');
  const countEl = document.getElementById('boxCount');
  if (!currentAnnotation || !currentAnnotation.boxes || currentAnnotation.boxes.length === 0) {
    boxList.innerHTML = '<p style="font-size:0.75rem; color:var(--text-muted); text-align:center;">No boxes (Normal / Defect-free).</p>';
    countEl.innerText = '0';
    return;
  }

  countEl.innerText = currentAnnotation.boxes.length;
  boxList.innerHTML = '';
  currentAnnotation.boxes.forEach((b, idx) => {
    const row = document.createElement('div');
    row.className = `box-row ${idx === selectedBoxIndex ? 'selected' : ''}`;
    row.style.borderLeftColor = CLASS_COLORS[b.label] || '#fff';
    row.onclick = () => {
      selectedBoxIndex = idx;
      document.getElementById('btnDeleteBox').disabled = false;
      document.getElementById('activeClassSelect').value = b.label;
      renderBoxList();
      draw();
    };

    row.innerHTML = `
      <div>
        <strong>${b.label}</strong>
        <span style="color:var(--text-muted); font-size:0.7rem;">[${Math.round(b.xmin)}, ${Math.round(b.ymin)}, ${Math.round(b.xmax)}, ${Math.round(b.ymax)}]</span>
      </div>
      <div class="box-actions">
        <button class="btn-tiny" onclick="event.stopPropagation(); changeBoxClass(${idx}, 'Crack')">C</button>
        <button class="btn-tiny" onclick="event.stopPropagation(); changeBoxClass(${idx}, 'Moss')">M</button>
        <button class="btn-tiny" onclick="event.stopPropagation(); changeBoxClass(${idx}, 'Seepage')">S</button>
        <button class="btn-tiny" style="background:#7f1d1d;" onclick="event.stopPropagation(); deleteBoxAt(${idx})">&times;</button>
      </div>
    `;
    boxList.appendChild(row);
  });
}

function changeBoxClass(idx, newLabel) {
  if (!currentAnnotation.boxes[idx]) return;
  currentAnnotation.boxes[idx].label = newLabel;
  currentAnnotation.boxes[idx].class_id = newLabel === 'Crack' ? 1 : (newLabel === 'Moss' ? 2 : 3);
  renderBoxList();
  draw();
}

function deleteBoxAt(idx) {
  if (!currentAnnotation.boxes[idx]) return;
  currentAnnotation.boxes.splice(idx, 1);
  selectedBoxIndex = -1;
  document.getElementById('btnDeleteBox').disabled = true;
  renderBoxList();
  draw();
}

function deleteSelectedBox() {
  if (selectedBoxIndex >= 0) {
    deleteBoxAt(selectedBoxIndex);
  }
}

async function markAsNormal() {
  if (!currentAnnotation) return;
  currentAnnotation.boxes = [];
  currentAnnotation.status = 'VERIFIED_NORMAL';
  currentAnnotation.notes = 'Verified as clean intact heritage surface (0 defect boxes)';
  await saveCurrent(true);
}

async function flagForReview() {
  const notes = prompt('Enter reason for flagging (ambiguous region/texture):', 'Ambiguous surface texture / lighting shadow');
  if (notes === null) return;
  const res = await fetch(`/api/flag/${encodeURIComponent(images[currentIndex].filename)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ notes: notes })
  });
  if (res.ok) {
    images[currentIndex].status = 'FLAGGED_AMBIGUOUS';
    nextImage();
  }
}

async function saveAndNext() {
  if (!currentAnnotation) return;
  currentAnnotation.status = currentAnnotation.boxes.length > 0 ? 'VERIFIED_DEFECT' : 'VERIFIED_NORMAL';
  await saveCurrent(true);
}

async function saveCurrent(advance = false) {
  const imgInfo = images[currentIndex];
  currentAnnotation.reviewer = 'Human Reviewer';
  currentAnnotation.review_timestamp = new Date().toISOString();

  const res = await fetch(`/api/annotation/${encodeURIComponent(imgInfo.filename)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(currentAnnotation)
  });

  if (res.ok) {
    images[currentIndex].status = currentAnnotation.status;
    await updateStats();
    if (advance) nextImage();
    else renderImageList();
  } else {
    alert('Failed to save annotation.');
  }
}

function nextImage() {
  if (currentIndex < images.length - 1) {
    loadImage(currentIndex + 1);
  }
}

function prevImage() {
  if (currentIndex > 0) {
    loadImage(currentIndex - 1);
  }
}

// Global hotkeys
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowRight') nextImage();
  else if (e.key === 'ArrowLeft') prevImage();
  else if (e.key === 'Delete' || e.key === 'Backspace') deleteSelectedBox();
  else if (e.key === 'n' || e.key === 'N') markAsNormal();
  else if (e.key === 's' || e.key === 'S' || e.key === 'd' || e.key === 'D') saveAndNext();
  else if (e.key === 'f' || e.key === 'F') flagForReview();
});

window.onload = init;
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/api/images")
def api_images():
    catalog = get_image_catalog()
    images_meta = []
    for item in catalog:
        fn = item["filename"]
        base_name = os.path.splitext(fn)[0]

        # Check status: review JSON -> proposal JSON -> unreviewed
        review_json = os.path.join(REVIEW_DIR, f"{base_name}.json")
        prop_json = os.path.join(PROPOSALS_DIR, f"{base_name}.json")

        status = ReviewStatus.UNREVIEWED.value
        box_count = 0
        if os.path.exists(review_json):
            with open(review_json, "r", encoding="utf-8") as fp:
                rdata = json.load(fp)
                status = rdata.get("status", ReviewStatus.VERIFIED_DEFECT.value)
                box_count = len(rdata.get("boxes", []))
        elif os.path.exists(prop_json):
            with open(prop_json, "r", encoding="utf-8") as fp:
                pdata = json.load(fp)
                status = pdata.get("status", ReviewStatus.REVIEW_REQUIRED.value)
                box_count = len(pdata.get("boxes", []))

        images_meta.append({
            "filename": fn,
            "rel_path": item["rel_path"],
            "site": item["site"],
            "width": item["width"],
            "height": item["height"],
            "status": status,
            "box_count": box_count
        })
    return jsonify(images_meta)

@app.route("/api/stats")
def api_stats():
    catalog = get_image_catalog()
    total = len(catalog)
    v_defects = 0
    v_normal = 0
    pending = 0

    for item in catalog:
        base_name = os.path.splitext(item["filename"])[0]
        review_json = os.path.join(REVIEW_DIR, f"{base_name}.json")
        if os.path.exists(review_json):
            with open(review_json, "r", encoding="utf-8") as fp:
                st = json.load(fp).get("status", "")
                if st == ReviewStatus.VERIFIED_NORMAL.value:
                    v_normal += 1
                else:
                    v_defects += 1
        else:
            pending += 1

    return jsonify({
        "total": total,
        "verified_defects": v_defects,
        "verified_normal": v_normal,
        "pending": pending
    })

@app.route("/api/image/<path:rel_path>")
def api_image(rel_path):
    full_path = os.path.join(DATASET_ROOT, rel_path)
    if not os.path.exists(full_path):
        return "Image not found", 404
    return send_file(full_path, mimetype="image/jpeg")

@app.route("/api/annotation/<path:filename>")
def api_annotation(filename):
    base_name = os.path.splitext(filename)[0]
    review_json = os.path.join(REVIEW_DIR, f"{base_name}.json")
    prop_json = os.path.join(PROPOSALS_DIR, f"{base_name}.json")

    if os.path.exists(review_json):
        with open(review_json, "r", encoding="utf-8") as fp:
            return jsonify(json.load(fp))

    if os.path.exists(prop_json):
        with open(prop_json, "r", encoding="utf-8") as fp:
            return jsonify(json.load(fp))

    # Empty fallback
    return jsonify({
        "filename": filename,
        "status": ReviewStatus.UNREVIEWED.value,
        "boxes": []
    })

@app.route("/api/annotation/<path:filename>", methods=["POST"])
def api_save_annotation(filename):
    base_name = os.path.splitext(filename)[0]
    data = request.json
    record = ImageAnnotationRecord.from_dict(data)

    # 1. Save JSON to review_dir
    review_json = os.path.join(REVIEW_DIR, f"{base_name}.json")
    with open(review_json, "w", encoding="utf-8") as fp:
        json.dump(record.to_dict(), fp, indent=2)

    # 2. Immediately export standard Pascal VOC XML to ai/annotations/raw/
    export_record_to_voc_xml(record, RAW_XML_DIR, DATASET_ROOT)

    return jsonify({"success": True, "saved_json": review_json, "status": record.status})

@app.route("/api/flag/<path:filename>", methods=["POST"])
def api_flag(filename):
    notes = request.json.get("notes", "Ambiguous region")
    base_name = os.path.splitext(filename)[0]

    # Save flagged status
    review_json = os.path.join(REVIEW_DIR, f"{base_name}.json")
    data = {
        "filename": filename,
        "status": ReviewStatus.FLAGGED_AMBIGUOUS.value,
        "notes": notes,
        "boxes": []
    }
    with open(review_json, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)

    # Append to flagged CSV
    file_exists = os.path.exists(FLAGGED_CSV)
    with open(FLAGGED_CSV, "a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if not file_exists:
            writer.writerow(["filename", "site", "suspected_class", "region_description", "status", "reviewer_notes"])
        writer.writerow([filename, "TEMPLE", "Ambiguous", notes, "FLAGGED", notes])

    return jsonify({"success": True})

def start_server(host="127.0.0.1", port=5500):
    print("=" * 70)
    print(" HERITAGE AI DECAY MONITORING — ANNOTATION REVIEW SERVER")
    print("=" * 70)
    print(f"Server URL:     http://{host}:{port}")
    print(f"Dataset Root:   {DATASET_ROOT}")
    print(f"Raw Output XML: {RAW_XML_DIR}")
    print("=" * 70)
    print("Open http://127.0.0.1:5500 in your browser to review proposals.\n")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5500)
    args = parser.parse_args()
    start_server(args.host, args.port)
