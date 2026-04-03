function cloneLensState(lensKey) {
  const source = LENS_STATES[lensKey];
  const out = {};
  Object.keys(source).forEach((token) => {
    out[token] = [source[token][0], source[token][1]];
  });
  return out;
}

function easeInOutCubic(t) {
  return t < 0.5
    ? 4 * t * t * t
    : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function syncProjectionButtons(lensKey) {
  const toolbar = document.getElementById('projectionToolbar16');
  if (!toolbar) return;
  toolbar.querySelectorAll('.lens-btn').forEach((btn) => {
    const active = btn.dataset.lens === lensKey;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', active ? 'true' : 'false');
  });
}

function updateProjectionReadout(lensKey) {
  const lensLabel = document.getElementById('projectionLensLabel16');
  const meaningLabel = document.getElementById('projectionMeaning16');
  const neighborsWrap = document.getElementById('projectionNeighbors16');
  const readout = document.getElementById('projectionReadout16');
  if (!lensLabel || !meaningLabel || !neighborsWrap || !readout) return;

  lensLabel.textContent = LENS_LABELS[lensKey] || lensKey;
  meaningLabel.textContent = LENS_MEANINGS[lensKey] || '';
  neighborsWrap.innerHTML = '';
  (LENS_NEIGHBORS[lensKey] || []).forEach((token) => {
    neighborsWrap.appendChild(createEl('span', {
      className: 'neighbor-chip',
      text: token
    }));
  });

  readout.classList.add('updating');
  if (projectionState.readoutTimer) clearTimeout(projectionState.readoutTimer);
  projectionState.readoutTimer = setTimeout(() => {
    readout.classList.remove('updating');
    projectionState.readoutTimer = null;
  }, 200);
}

function resizeProjectionCanvas() {
  if (!projectionState.canvas || !projectionState.ctx) return;
  const canvas = projectionState.canvas;
  const rect = canvas.getBoundingClientRect();
  const cssWidth = Math.max(1, Math.floor(rect.width));
  const cssHeight = Math.max(1, Math.floor(rect.height));
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(cssWidth * dpr);
  canvas.height = Math.floor(cssHeight * dpr);
  projectionState.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawProjection() {
  if (!projectionState.initialized || !projectionState.ctx || !projectionState.canvas || !projectionState.currentPositions) return;
  const canvas = projectionState.canvas;
  const ctx = projectionState.ctx;
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.width / dpr;
  const height = canvas.height / dpr;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const pad = 32;
  const plotW = Math.max(1, width - pad * 2);
  const plotH = Math.max(1, height - pad * 2);
  const centerX = pad + plotW * 0.5;
  const centerY = pad + plotH * 0.5;
  const neighbors = new Set(LENS_NEIGHBORS[projectionState.activeLens] || []);
  const emphasized = new Set(LENS_EMPHASIS[projectionState.activeLens] || []);

  const GROUP_POINT_STYLES = {
    pets: { fill: [123, 154, 255], stroke: [186, 218, 255] },
    bigcats: { fill: [255, 170, 85], stroke: [255, 214, 180] },
    care: { fill: [180, 157, 255], stroke: [220, 205, 255] }
  };
  const toRgba = (rgb, alpha) => `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;

  const toXY = (coords) => [pad + coords[0] * plotW, pad + coords[1] * plotH];

  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(123,154,255,0.12)';
  for (let i = 0; i <= 10; i += 1) {
    const x = pad + (plotW * i) / 10;
    const y = pad + (plotH * i) / 10;
    ctx.beginPath();
    ctx.moveTo(x, pad);
    ctx.lineTo(x, pad + plotH);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(pad + plotW, y);
    ctx.stroke();
  }

  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = 'rgba(180,157,255,0.3)';
  ctx.beginPath();
  ctx.moveTo(centerX, pad);
  ctx.lineTo(centerX, pad + plotH);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(pad, centerY);
  ctx.lineTo(pad + plotW, centerY);
  ctx.stroke();
  ctx.setLineDash([]);

  PROJECTION_TOKENS.forEach((token) => {
    if (token === 'cat') return;
    const coords = projectionState.currentPositions[token];
    if (!coords) return;
    const [x, y] = toXY(coords);
    const group = TOKEN_GROUPS[token];
    const style = GROUP_POINT_STYLES[group] || GROUP_POINT_STYLES.pets;
    const isNeighbor = neighbors.has(token);
    const isEmphasized = emphasized.has(token);
    const alpha = isEmphasized ? 0.95 : 0.32;
    const labelOffset = TOKEN_LABEL_OFFSETS[token] || [9, -10];

    if (isNeighbor) {
      ctx.beginPath();
      ctx.arc(x, y, 16, 0, Math.PI * 2);
      ctx.fillStyle = isEmphasized ? 'rgba(255,214,68,0.2)' : 'rgba(255,214,68,0.1)';
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(x, y, 9, 0, Math.PI * 2);
    ctx.fillStyle = toRgba(style.fill, alpha);
    ctx.fill();
    ctx.lineWidth = 1.8;
    ctx.strokeStyle = toRgba(style.stroke, isEmphasized ? 0.75 : 0.34);
    ctx.stroke();

    ctx.font = '600 19px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = isEmphasized ? 'rgba(0,0,0,1)' : 'rgba(0,0,0,0.55)';
    ctx.fillText(token, x + labelOffset[0], y + labelOffset[1]);
  });

  const catCoords = projectionState.currentPositions.cat || [0.5, 0.5];
  const [catX, catY] = toXY(catCoords);
  ctx.beginPath();
  ctx.arc(catX, catY, 22, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(90,232,142,0.18)';
  ctx.fill();
  ctx.beginPath();
  ctx.arc(catX, catY, 12, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(90,232,142,0.95)';
  ctx.fill();
  ctx.lineWidth = 2.2;
  ctx.strokeStyle = 'rgba(10,90,45,0.85)';
  ctx.stroke();

  ctx.font = '700 19px Inter, sans-serif';
  ctx.fillStyle = '#000000';
  ctx.textAlign = 'left';
  ctx.fillText('cat (anchor)', catX + 12, catY + 1);
}

function animateProjectionFrame(ts) {
  if (!projectionState.animFrom || !projectionState.animTo) {
    projectionState.rafId = null;
    return;
  }
  const duration = 760;
  const raw = Math.min(1, (ts - projectionState.animStart) / duration);
  const eased = easeInOutCubic(raw);

  PROJECTION_TOKENS.forEach((token) => {
    const from = projectionState.animFrom[token];
    const to = projectionState.animTo[token];
    if (!from || !to) return;
    projectionState.currentPositions[token][0] = from[0] + (to[0] - from[0]) * eased;
    projectionState.currentPositions[token][1] = from[1] + (to[1] - from[1]) * eased;
  });

  drawProjection();

  if (raw < 1) {
    projectionState.rafId = requestAnimationFrame(animateProjectionFrame);
    return;
  }

  projectionState.currentPositions = cloneLensState(projectionState.activeLens);
  projectionState.rafId = null;
  projectionState.animFrom = null;
  projectionState.animTo = null;
  drawProjection();
}

function setProjectionLens(lensKey) {
  if (!LENS_STATES[lensKey] || !projectionState.initialized) return;
  if (projectionState.activeLens === lensKey && !projectionState.rafId) return;

  if (!projectionState.currentPositions) {
    projectionState.activeLens = lensKey;
    projectionState.currentPositions = cloneLensState(lensKey);
    syncProjectionButtons(lensKey);
    updateProjectionReadout(lensKey);
    drawProjection();
    return;
  }

  const from = {};
  PROJECTION_TOKENS.forEach((token) => {
    from[token] = [
      projectionState.currentPositions[token][0],
      projectionState.currentPositions[token][1]
    ];
  });

  projectionState.animFrom = from;
  projectionState.animTo = cloneLensState(lensKey);
  projectionState.animStart = performance.now();
  projectionState.activeLens = lensKey;
  syncProjectionButtons(lensKey);
  updateProjectionReadout(lensKey);

  if (projectionState.rafId) cancelAnimationFrame(projectionState.rafId);
  projectionState.rafId = requestAnimationFrame(animateProjectionFrame);
}

function initProjectionSlide() {
  const slide = document.getElementById('slide-16');
  const canvas = document.getElementById('projectionCanvas16');
  const toolbar = document.getElementById('projectionToolbar16');
  if (!slide || !canvas || !toolbar) return;

  if (!projectionState.initialized) {
    projectionState.canvas = canvas;
    projectionState.ctx = canvas.getContext('2d');
    projectionState.currentPositions = cloneLensState(DEFAULT_PROJECTION_LENS);
    projectionState.activeLens = DEFAULT_PROJECTION_LENS;

    toolbar.querySelectorAll('.lens-btn').forEach((btn) => {
      addTrackedListener(btn, 'click', () => {
        if (typeof runMutationWithHistory === 'function') {
          if (runMutationWithHistory(() => setProjectionLens(btn.dataset.lens))) {
            scheduleDeckRefresh({ reason: 'projection-lens', typeset: false });
          }
          return;
        }
        setProjectionLens(btn.dataset.lens);
        scheduleDeckRefresh({ reason: 'projection-lens', typeset: false });
      });
    });

    if (!projectionState.resizeBound) {
      addTrackedListener(window, 'resize', () => {
        if (!projectionState.initialized) return;
        resizeProjectionCanvas();
        drawProjection();
      });
      projectionState.resizeBound = true;
    }

    projectionState.initialized = true;
  }

  syncProjectionButtons(projectionState.activeLens);
  updateProjectionReadout(projectionState.activeLens);
  resizeProjectionCanvas();
  drawProjection();
}

function resetProjectionSlide() {
  if (!projectionState.initialized) return;
  if (projectionState.rafId) {
    cancelAnimationFrame(projectionState.rafId);
    projectionState.rafId = null;
  }
  if (projectionState.readoutTimer) {
    clearTimeout(projectionState.readoutTimer);
    projectionState.readoutTimer = null;
  }
  projectionState.activeLens = DEFAULT_PROJECTION_LENS;
  projectionState.currentPositions = cloneLensState(DEFAULT_PROJECTION_LENS);
  projectionState.animFrom = null;
  projectionState.animTo = null;
  syncProjectionButtons(DEFAULT_PROJECTION_LENS);
  updateProjectionReadout(DEFAULT_PROJECTION_LENS);
  resizeProjectionCanvas();
  drawProjection();
}
