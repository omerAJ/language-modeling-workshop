function initAttentionP1Slide() {
  const slide = document.getElementById('slide-19');
  const tokenCols = document.getElementById('attn19p1-cols');
  const projRow = document.getElementById('attn19p1-proj-row');
  if (!slide || !tokenCols || !projRow) return;

  if (!state.attentionP1.initialized) {
    // Build in-stage token-lane chips
    tokenCols.innerHTML = '';
    ATTN_P1_TOKENS.forEach((token) => {
      const cell = createEl('div', {
        className: 'attn19p1-chip-cell',
        dataset: { token }
      });
      const chipWrap = createEl('div', { className: 'attn19-chip-wrap attn19p1-chip-wrap' });
      chipWrap.appendChild(createEl('div', {
        className: 'attn19-token-chip',
        id: 'attn19p1-chip-' + token,
        text: token
      }));
      cell.appendChild(chipWrap);
      tokenCols.appendChild(cell);
    });

    // Build projection columns
    projRow.innerHTML = '';
    ATTN_P1_PROJS.forEach((proj) => {
      const col = createEl('div', { className: 'attn19p1-proj-col', dataset: { proj } });
      const matrixLabel = createEl('div', { className: 'attn19p1-matrix-label' }, [
        'W',
        createEl('sub', { text: proj.toUpperCase() })
      ]);

      const mulRow = createEl('div', { className: 'attn19p1-mul-row' });
      mulRow.appendChild(createEl('div', { className: 'attn19p1-mini-x-wrap' }, [
        createEl('span', { className: 'attn19p1-mini-x-label' }, [
          'x',
          createEl('sub', { text: ATTN_P1_FOCUS })
        ]),
        createVectorRect({
          id: 'attn19p1-mini-x-' + proj,
          baseClass: 'attn19p1-mini-x-vector',
          dividerClass: 'attn19p1-vec-divider'
        })
      ]));
      mulRow.appendChild(createEl('span', { className: 'attn19p1-mul-symbol', text: '\u00d7' }));

      const matrix = createEl('div', {
        className: 'attn19p1-matrix',
        id: 'attn19p1-matrix-' + proj
      });
      for (let i = 1; i <= 3; i += 1) {
        matrix.appendChild(createEl('span', {
          className: 'attn19p1-matrix-grid-v',
          style: { left: (i * 25) + '%' }
        }));
        matrix.appendChild(createEl('span', {
          className: 'attn19p1-matrix-grid-h',
          style: { top: (i * 25) + '%' }
        }));
      }
      const matrixCluster = createEl('div', { className: 'attn19p1-matrix-cluster' });
      matrixCluster.appendChild(matrix);
      matrixCluster.appendChild(matrixLabel);
      mulRow.appendChild(matrixCluster);
      col.appendChild(mulRow);

      const vWrap = createEl('div', { className: 'attn19p1-vector-wrap attn19p1-output-wrap' });
      const label = createEl('span', { className: 'attn19p1-vec-label' }, [
        proj,
        createEl('sub', { text: ATTN_P1_FOCUS })
      ]);
      const vec = createEl('div', {
        className: 'attn19p1-proj-vector',
        id: 'attn19p1-vec-' + proj
      });
      for (let i = 1; i <= 3; i += 1) {
        vec.appendChild(createEl('span', {
          className: 'attn19p1-vec-divider',
          style: { left: (i * 25) + '%' }
        }));
      }
      vWrap.appendChild(label);
      vWrap.appendChild(vec);
      col.appendChild(vWrap);

      col.appendChild(createEl('div', {
        className: 'attn19p1-proj-header',
        text: ATTN_P1_PROJ_NAMES[proj]
      }));

      col.appendChild(createEl('div', {
        className: 'attn19p1-meaning',
        text: ATTN_P1_MEANINGS[proj]
      }));

      projRow.appendChild(col);
    });

    if (!state.attentionP1.resizeBound) {
      addTrackedListener(window, 'resize', () => {
        if (!state.attentionP1.initialized) return;
        updateAttentionP1Overlay();
      });
      state.attentionP1.resizeBound = true;
    }

    state.attentionP1.initialized = true;
  }

  const takeaway = document.getElementById('attn19p1-takeaway');
  if (takeaway) takeaway.innerHTML = ATTN_P1_TAKEAWAYS[state.attentionP1.step] || ATTN_P1_TAKEAWAYS[0];
  updateAttentionP1Overlay();
}

function updateAttentionP1Overlay() {
  const stage = document.getElementById('attn19p1-stage');
  const overlay = document.getElementById('attn19p1-overlay');
  const contentWrap = stage ? stage.querySelector('.attn19p1-content') : null;
  const satChip = document.getElementById('attn19p1-chip-' + ATTN_P1_FOCUS);
  const xSat = document.getElementById('attn19p1-x-sat');
  const sourceLine = document.getElementById('attn19p1-line-source');
  const busTrunkLine = document.getElementById('attn19p1-line-bus-trunk');
  const busMainLine = document.getElementById('attn19p1-line-bus-main');
  const copyNode = document.getElementById('attn19p1-copy-node');
  const copyLabel = document.getElementById('attn19p1-copy-label');
  if (!stage || !overlay || !xSat || !sourceLine || !busTrunkLine || !busMainLine || !copyNode || !copyLabel) return;

  const stageRect = stage.getBoundingClientRect();
  if (stageRect.width < 1 || stageRect.height < 1) return;
  overlay.setAttribute('viewBox', '0 0 ' + stageRect.width + ' ' + stageRect.height);

  const anchor = (el, fx, fy) => {
    const r = el.getBoundingClientRect();
    return {
      x: r.left - stageRect.left + r.width * fx,
      y: r.top - stageRect.top + r.height * fy
    };
  };

  if (contentWrap) {
    contentWrap.style.transform = 'translateX(0px)';
    if (satChip) {
      const satCenterX = anchor(satChip, 0.5, 0.5).x;
      const xCenterX = anchor(xSat, 0.5, 0.5).x;
      const deltaX = satCenterX - xCenterX;
      contentWrap.style.transform = 'translateX(' + deltaX.toFixed(2) + 'px)';
    }
  }

  const xBottom = anchor(xSat, 0.5, 1);
  const linePad = Math.max(stageRect.height * 0.009, 3.5);
  const branchTargets = [];

  ATTN_P1_PROJS.forEach((proj) => {
    const miniX = document.getElementById('attn19p1-mini-x-' + proj);
    const matrix = document.getElementById('attn19p1-matrix-' + proj);
    const line = document.getElementById('attn19p1-line-' + proj);
    if (!miniX || !matrix || !line) return;
    branchTargets.push({
      proj,
      line,
      target: anchor(miniX, 0.5, 0),
      matrixTopY: anchor(matrix, 0.5, 0).y
    });
  });

  if (!branchTargets.length) return;

  const nearestCopyY = Math.min.apply(null, branchTargets.map((entry) => entry.target.y));
  const nearestMatrixY = Math.min.apply(null, branchTargets.map((entry) => entry.matrixTopY));
  const minCopyX = Math.min.apply(null, branchTargets.map((entry) => entry.target.x));
  const maxCopyX = Math.max.apply(null, branchTargets.map((entry) => entry.target.x));
  const nodeX = xBottom.x;
  const nodeRadius = Math.max(stageRect.height * 0.018, 6.4);
  const minNodeGap = Math.max(stageRect.height * 0.3, 84);
  const maxNodeY = nearestMatrixY - nodeRadius - Math.max(stageRect.height * 0.045, 9);
  const nodeY = Math.max(
    xBottom.y + nodeRadius + 2,
    Math.min(xBottom.y + minNodeGap, maxNodeY)
  );
  const busStartX = minCopyX;
  const busEndX = maxCopyX;
  const busClearBelowNode = Math.max(stageRect.height * 0.02, 6);
  const busMinY = nodeY + nodeRadius + busClearBelowNode;
  const busMaxY = nearestMatrixY - Math.max(stageRect.height * 0.026, 9);
  const busPreferredY = Math.min(
    nearestCopyY - Math.max(stageRect.height * 0.028, 11),
    nodeY + Math.max(stageRect.height * 0.072, 18)
  );
  let busY = Math.max(busMinY, Math.min(busPreferredY, busMaxY));
  if (!Number.isFinite(busY)) busY = busMinY;
  if (busY < busMinY) busY = busMinY;
  if (busY > busMaxY) busY = busMaxY;

  copyNode.setAttribute('cx', nodeX.toFixed(2));
  copyNode.setAttribute('cy', nodeY.toFixed(2));
  copyNode.setAttribute('r', nodeRadius.toFixed(2));
  copyLabel.setAttribute('x', nodeX.toFixed(2));
  copyLabel.setAttribute('y', nodeY.toFixed(2));

  const sourceY1 = xBottom.y + linePad;
  const sourceY2 = nodeY - nodeRadius - Math.max(stageRect.height * 0.006, 2);
  sourceLine.setAttribute('x1', nodeX.toFixed(2));
  sourceLine.setAttribute('y1', sourceY1.toFixed(2));
  sourceLine.setAttribute('x2', nodeX.toFixed(2));
  sourceLine.setAttribute('y2', sourceY2.toFixed(2));
  busTrunkLine.setAttribute('x1', nodeX.toFixed(2));
  busTrunkLine.setAttribute('y1', (nodeY + nodeRadius + Math.max(stageRect.height * 0.004, 1.2)).toFixed(2));
  busTrunkLine.setAttribute('x2', nodeX.toFixed(2));
  busTrunkLine.setAttribute('y2', busY.toFixed(2));
  busMainLine.setAttribute('x1', busStartX.toFixed(2));
  busMainLine.setAttribute('y1', busY.toFixed(2));
  busMainLine.setAttribute('x2', busEndX.toFixed(2));
  busMainLine.setAttribute('y2', busY.toFixed(2));

  branchTargets.forEach((entry) => {
    const toX = entry.target.x;
    const toY = entry.target.y - Math.max(stageRect.height * 0.004, 1.2);
    entry.line.setAttribute('x1', toX.toFixed(2));
    entry.line.setAttribute('y1', busY.toFixed(2));
    entry.line.setAttribute('x2', toX.toFixed(2));
    entry.line.setAttribute('y2', toY.toFixed(2));
  });
}

function setAttentionP1Step(step) {
  const slide = document.getElementById('slide-19');
  const takeaway = document.getElementById('attn19p1-takeaway');
  if (!slide || !takeaway) return;

  const clamped = Math.max(0, Math.min(ATTN_P1_MAX_STEP, step));
  state.attentionP1.step = clamped;

  slide.classList.toggle('attn19p1-show-source', clamped >= 1);
  slide.classList.toggle('attn19p1-show-copies', clamped >= 2);
  slide.classList.toggle('attn19p1-show-q-matrix', clamped >= 3);
  slide.classList.toggle('attn19p1-show-q-output', clamped >= 4);
  slide.classList.toggle('attn19p1-show-k-matrix', clamped >= 5);
  slide.classList.toggle('attn19p1-show-k-output', clamped >= 6);
  slide.classList.toggle('attn19p1-show-v-matrix', clamped >= 7);
  slide.classList.toggle('attn19p1-show-v-output', clamped >= 8);
  takeaway.innerHTML = ATTN_P1_TAKEAWAYS[clamped] || ATTN_P1_TAKEAWAYS[0];

  updateAttentionP1Overlay();
  requestAnimationFrame(updateAttentionP1Overlay);
  if (state.attentionP1.overlayTimer) clearTimeout(state.attentionP1.overlayTimer);
  state.attentionP1.overlayTimer = setTimeout(() => {
    updateAttentionP1Overlay();
    state.attentionP1.overlayTimer = null;
  }, 240);
}

function runAttentionP1Step() {
  if (!state.attentionP1.initialized) initAttentionP1Slide();
  if (state.attentionP1.step >= ATTN_P1_MAX_STEP) return false;
  setAttentionP1Step(state.attentionP1.step + 1);
  return true;
}

function resetAttentionP1Slide() {
  const slide = document.getElementById('slide-19');
  if (!slide) return;
  if (state.attentionP1.overlayTimer) {
    clearTimeout(state.attentionP1.overlayTimer);
    state.attentionP1.overlayTimer = null;
  }
  setAttentionP1Step(0);
}

/* ===================================================== */
