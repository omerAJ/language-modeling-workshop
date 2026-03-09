function createAttentionMultiHeadMatrixRow(prefix, token, values = null, dims = 4, rowClass = '', vectorClass = '') {
  const row = createEl('div', {
    className: 'attn24-matrix-row ' + rowClass,
    id: prefix + '-row-' + token,
    dataset: { token }
  });
  const wrap = createEl('div', {
    className: 'attn19-vector-wrap attn19-vector-wrap-x'
  });
  const vector = Array.isArray(values)
    ? createAttentionQkvVectorRect(prefix + '-vector-' + token, vectorClass, values)
    : createAttentionSkeletonVectorRect(prefix + '-vector-' + token, vectorClass, dims);
  wrap.appendChild(vector);
  row.appendChild(wrap);
  return row;
}

function createAttentionMultiHeadSourceMatrix() {
  const panel = createEl('div', {
    className: 'attn24-source-panel',
    id: 'attn24-source-panel'
  });
  panel.appendChild(createEl('div', {
    className: 'attn24-source-label',
    id: 'attn24-source-label',
    html: 'Embedding Matrix \\(X\\)'
  }));
  panel.appendChild(createEl('div', {
    className: 'attn24-source-meta',
    id: 'attn24-source-meta',
    html: inlineMath('X \\in \\mathbb{R}^{S \\times d}')
  }));
  const shell = createEl('div', {
    className: 'attn24-source-shell',
    id: 'attn24-source-shell'
  });
  const slots = createEl('div', {
    className: 'attn24-matrix-slots',
    id: 'attn24-source-slots'
  });
  ATTN_MHA_TOKENS.forEach((token) => {
    slots.appendChild(createAttentionMultiHeadMatrixRow(
      'attn24-source',
      token,
      ATTN_QKV_X_VECTORS[token] || [],
      4,
      'attn24-source-row'
    ));
  });
  shell.appendChild(slots);
  panel.appendChild(shell);
  return panel;
}

function createAttentionMultiHeadMiniX(head) {
  const wrap = createEl('div', {
    className: 'attn24-head-mini-x-wrap',
    id: 'attn24-head-mini-x-wrap-' + head
  });
  wrap.appendChild(createEl('div', {
    className: 'attn24-head-mini-x-label',
    html: inlineMath('X')
  }));
  const shell = createEl('div', {
    className: 'attn24-head-mini-x-shell',
    id: 'attn24-head-mini-x-shell-' + head
  });
  const slots = createEl('div', {
    className: 'attn24-matrix-slots',
    id: 'attn24-head-mini-x-slots-' + head
  });
  ATTN_MHA_TOKENS.forEach((token) => {
    slots.appendChild(createAttentionMultiHeadMatrixRow(
      'attn24-head-mini-x-' + head,
      token,
      null,
      4,
      'attn24-head-mini-x-row'
    ));
  });
  shell.appendChild(slots);
  wrap.appendChild(shell);
  return wrap;
}

function createAttentionMultiHeadQkvMatrix(head, proj) {
  const wrap = createEl('div', {
    className: 'attn24-head-qkv-wrap',
    id: 'attn24-head-qkv-wrap-' + proj + '-' + head,
    dataset: { proj, head }
  });
  wrap.appendChild(createEl('div', {
    className: 'attn24-head-qkv-label',
    html: inlineMath(proj.toUpperCase())
  }));
  const shell = createEl('div', {
    className: 'attn24-head-qkv-shell',
    id: 'attn24-head-qkv-shell-' + proj + '-' + head
  });
  const slots = createEl('div', {
    className: 'attn24-matrix-slots',
    id: 'attn24-head-qkv-slots-' + proj + '-' + head
  });
  ATTN_MHA_TOKENS.forEach((token) => {
    slots.appendChild(createAttentionMultiHeadMatrixRow(
      'attn24-head-qkv-' + proj + '-' + head,
      token,
      null,
      ATTN_MHA_HEAD_DIM,
      'attn24-head-qkv-row'
    ));
  });
  shell.appendChild(slots);
  wrap.appendChild(shell);
  return wrap;
}

function createAttentionMultiHeadWMatrix(head, proj) {
  const wrap = createEl('div', {
    className: 'attn24-head-w-wrap',
    id: 'attn24-head-w-wrap-' + proj + '-' + head,
    dataset: { proj, head }
  });
  const shell = createEl('div', {
    className: 'attn24-head-w-shell',
    id: 'attn24-head-w-shell-' + proj + '-' + head
  });
  const grid = createEl('div', {
    className: 'attn24-head-w-grid',
    id: 'attn24-head-w-grid-' + proj + '-' + head
  });
  grid.appendChild(createEl('span', {
    className: 'attn24-head-w-grid-v'
  }));
  [25, 50, 75].forEach((pct) => {
    grid.appendChild(createEl('span', {
      className: 'attn24-head-w-grid-h',
      style: { top: pct + '%' }
    }));
  });
  shell.appendChild(grid);
  wrap.appendChild(shell);
  wrap.appendChild(createEl('div', {
    className: 'attn24-head-w-label',
    id: 'attn24-head-w-label-' + proj + '-' + head
  }, createEl('span', {
    html: inlineMath('W_' + proj.toUpperCase())
  })));
  wrap.appendChild(createEl('div', {
    className: 'attn24-head-w-dims',
    id: 'attn24-head-w-dims-' + proj + '-' + head,
    html: inlineMath('4 \\times 2')
  }));
  return wrap;
}

function createAttentionMultiHeadProjectionRow(head, proj) {
  const row = createEl('div', {
    className: 'attn24-head-proj-row',
    id: 'attn24-head-proj-row-' + proj + '-' + head,
    dataset: { head, proj }
  });
  row.appendChild(createAttentionMultiHeadWMatrix(head, proj));
  row.appendChild(createEl('div', {
    className: 'attn24-head-proj-arrow',
    id: 'attn24-head-proj-arrow-' + proj + '-' + head,
    text: '→'
  }));
  row.appendChild(createAttentionMultiHeadQkvMatrix(head, proj));
  return row;
}

function createAttentionMultiHeadHeadOverlay(head) {
  const overlay = createEl('div', {
    className: 'attn24-head-overlay',
    id: 'attn24-head-overlay-' + head,
    ariaHidden: 'true'
  });
  overlay.appendChild(createEl('div', {
    className: 'attn24-head-overlay-line',
    id: 'attn24-head-line-source-' + head
  }));
  overlay.appendChild(createEl('div', {
    className: 'attn24-head-overlay-bus',
    id: 'attn24-head-line-bus-' + head
  }));
  overlay.appendChild(createEl('div', {
    className: 'attn24-head-copy-node',
    id: 'attn24-head-copy-node-' + head
  }));
  overlay.appendChild(createEl('div', {
    className: 'attn24-head-copy-label',
    id: 'attn24-head-copy-label-' + head,
    text: '×3'
  }));
  ATTN_MHA_PROJS.forEach((proj) => {
    overlay.appendChild(createEl('div', {
      className: 'attn24-head-overlay-line',
      id: 'attn24-head-line-' + proj + '-' + head
    }));
  });
  return overlay;
}

function createAttentionMultiHeadAttentionBlock(head) {
  return createEl('div', {
    className: 'attn24-head-attn-block',
    id: 'attn24-head-attn-block-' + head
  }, [
    createEl('div', {
      className: 'attn24-head-attn-block-title',
      text: 'Masked Attention'
    }),
    createEl('div', {
      className: 'attn24-head-attn-block-formula'
    }, createEl('span', {
      html: inlineMath('\\operatorname{softmax}\\!\\left(\\frac{QK^{\\mathsf{T}}}{\\sqrt{d_h}} + M\\right)V')
    }))
  ]);
}

function createAttentionMultiHeadOutputMatrix(head) {
  const wrap = createEl('div', {
    className: 'attn24-head-output-wrap',
    id: 'attn24-head-output-wrap-' + head
  });
  const headIndex = head === 'h1' ? '1' : '2';
  wrap.appendChild(createEl('div', {
    className: 'attn24-head-output-label',
    html: inlineMath('O^{(' + headIndex + ')}')
  }));
  const shell = createEl('div', {
    className: 'attn24-head-output-shell',
    id: 'attn24-head-output-shell-' + head
  });
  const slots = createEl('div', {
    className: 'attn24-matrix-slots',
    id: 'attn24-head-output-slots-' + head
  });
  ATTN_MHA_TOKENS.forEach((token) => {
    const row = createAttentionMultiHeadMatrixRow(
      'attn24-head-output-' + head,
      token,
      (ATTN_MHA_OUTPUT_ROWS[head] || {})[token] || [],
      ATTN_MHA_HEAD_DIM,
      'attn24-head-output-row'
    );
    row.classList.add('is-hidden');
    slots.appendChild(row);
  });
  shell.appendChild(slots);
  wrap.appendChild(shell);
  return wrap;
}

function createAttentionMultiHeadConcatRow(token) {
  return createAttentionMultiHeadMatrixRow(
    'attn24-concat',
    token,
    null,
    ATTN_MHA_COMBINED_DIM,
    'attn24-concat-row',
    'attn24-concat-vector'
  );
}

function createAttentionMultiHeadConcatMatrix() {
  const wrap = createEl('div', {
    className: 'attn24-concat-wrap',
    id: 'attn24-concat-wrap'
  });
  wrap.appendChild(createEl('div', {
    className: 'attn24-concat-label',
    id: 'attn24-concat-label',
    text: 'Concatenated Heads'
  }));
  wrap.appendChild(createEl('div', {
    className: 'attn24-concat-dims',
    id: 'attn24-concat-dims',
    html: inlineMath('\\operatorname{Concat}(O^{(1)}, O^{(2)}) \\in \\mathbb{R}^{S \\times d}')
  }));
  const shell = createEl('div', {
    className: 'attn24-concat-shell',
    id: 'attn24-concat-shell'
  });
  const slots = createEl('div', {
    className: 'attn24-matrix-slots',
    id: 'attn24-concat-slots'
  });
  ATTN_MHA_TOKENS.forEach((token) => {
    slots.appendChild(createAttentionMultiHeadConcatRow(token));
  });
  shell.appendChild(slots);
  shell.appendChild(createEl('span', {
    className: 'attn24-concat-target is-left',
    id: 'attn24-concat-target-left'
  }));
  shell.appendChild(createEl('span', {
    className: 'attn24-concat-target is-right',
    id: 'attn24-concat-target-right'
  }));
  wrap.appendChild(shell);
  return wrap;
}

function createAttentionMultiHeadWoMatrix() {
  const wrap = createEl('div', {
    className: 'attn24-wo-wrap',
    id: 'attn24-wo-wrap'
  });
  wrap.appendChild(createEl('div', {
    className: 'attn24-wo-label',
    id: 'attn24-wo-label'
  }, createEl('span', {
    html: inlineMath('W_O')
  })));
  wrap.appendChild(createEl('div', {
    className: 'attn24-wo-dims',
    id: 'attn24-wo-dims',
    html: inlineMath('4 \\times 4')
  }));
  const shell = createEl('div', {
    className: 'attn24-wo-shell',
    id: 'attn24-wo-shell'
  });
  const grid = createEl('div', {
    className: 'attn24-wo-grid',
    id: 'attn24-wo-grid'
  });
  [25, 50, 75].forEach((pct) => {
    grid.appendChild(createEl('span', {
      className: 'attn24-wo-grid-v',
      style: { left: pct + '%' }
    }));
    grid.appendChild(createEl('span', {
      className: 'attn24-wo-grid-h',
      style: { top: pct + '%' }
    }));
  });
  shell.appendChild(grid);
  wrap.appendChild(shell);
  return wrap;
}

function createAttentionMultiHeadFinalOutputMatrix() {
  const wrap = createEl('div', {
    className: 'attn24-mha-output-wrap',
    id: 'attn24-mha-output-wrap'
  });
  wrap.appendChild(createEl('div', {
    className: 'attn24-mha-output-label',
    id: 'attn24-mha-output-label',
    text: 'Multi-Head Output'
  }));
  wrap.appendChild(createEl('div', {
    className: 'attn24-mha-output-dims',
    id: 'attn24-mha-output-dims',
    html: inlineMath('O \\in \\mathbb{R}^{S \\times d}')
  }));
  const shell = createEl('div', {
    className: 'attn24-mha-output-shell',
    id: 'attn24-mha-output-shell'
  });
  const slots = createEl('div', {
    className: 'attn24-matrix-slots',
    id: 'attn24-mha-output-slots'
  });
  ATTN_MHA_TOKENS.forEach((token) => {
    slots.appendChild(createAttentionMultiHeadMatrixRow(
      'attn24-mha-output',
      token,
      null,
      ATTN_MHA_MODEL_DIM,
      'attn24-mha-output-row',
      'attn24-final-output-vector'
    ));
  });
  shell.appendChild(slots);
  wrap.appendChild(shell);
  return wrap;
}

function createAttentionMultiHeadCombineStage() {
  const stage = createEl('div', {
    className: 'attn24-combine-stage-inner',
    id: 'attn24-combine-stage-inner'
  });
  stage.appendChild(createAttentionMultiHeadConcatMatrix());
  stage.appendChild(createEl('div', {
    className: 'attn24-combine-mul',
    id: 'attn24-combine-mul',
    text: '×'
  }));
  stage.appendChild(createAttentionMultiHeadWoMatrix());
  stage.appendChild(createEl('div', {
    className: 'attn24-combine-equals',
    id: 'attn24-combine-equals',
    text: '='
  }));
  stage.appendChild(createAttentionMultiHeadFinalOutputMatrix());
  return stage;
}

function createAttentionMultiHeadHeadCard(head) {
  const card = createEl('div', {
    className: 'attn24-head-card',
    id: 'attn24-head-card-' + head,
    dataset: { head }
  });
  card.appendChild(createEl('div', {
    className: 'attn24-head-header',
    id: 'attn24-head-header-' + head
  }, [
    createEl('div', {
      className: 'attn24-head-title',
      text: ATTN_MHA_HEAD_LABELS[head]
    }),
    createEl('div', {
      className: 'attn24-head-meta'
    }, [
      'd',
      createEl('sub', { text: 'h' }),
      ' = ',
      String(ATTN_MHA_HEAD_DIM)
    ])
  ]));
  const body = createEl('div', {
    className: 'attn24-head-body',
    id: 'attn24-head-body-' + head
  });
  body.appendChild(createAttentionMultiHeadHeadOverlay(head));
  const sourceCol = createEl('div', {
    className: 'attn24-head-source-col',
    id: 'attn24-head-source-col-' + head
  });
  sourceCol.appendChild(createAttentionMultiHeadMiniX(head));
  body.appendChild(sourceCol);

  const projPane = createEl('div', {
    className: 'attn24-head-proj-pane',
    id: 'attn24-head-proj-pane-' + head
  });
  ATTN_MHA_PROJS.forEach((proj) => {
    projPane.appendChild(createAttentionMultiHeadProjectionRow(head, proj));
  });
  body.appendChild(projPane);
  card.appendChild(body);
  card.appendChild(createAttentionMultiHeadAttentionBlock(head));
  card.appendChild(createAttentionMultiHeadOutputMatrix(head));
  return card;
}

function clearAttentionMultiHeadTimers() {
  state.attentionMultiHead.timers.forEach((timerId) => clearTimeout(timerId));
  state.attentionMultiHead.timers = [];
  state.attentionMultiHead.rafIds.forEach((rafId) => cancelAnimationFrame(rafId));
  state.attentionMultiHead.rafIds = [];
}

function clearAttentionMultiHeadGhosts() {
  const layer = document.getElementById('attn24-ghost-layer');
  if (!layer) return;
  layer.querySelectorAll('.attn24-ghost').forEach((ghost) => ghost.remove());
}

function cloneAttentionMultiHeadFragment(node, extraClass = '') {
  if (!node) return null;
  const ghost = node.cloneNode(true);
  const stripIds = (el) => {
    if (!el || el.nodeType !== 1) return;
    el.removeAttribute('id');
    Array.from(el.children).forEach(stripIds);
  };
  stripIds(ghost);
  ghost.classList.add('attn24-ghost');
  if (extraClass) ghost.classList.add(extraClass);
  return ghost;
}

function placeAttentionMultiHeadGhost(ghost, rect, stageRect) {
  if (!ghost || !rect || !stageRect) return;
  ghost.style.left = (rect.left - stageRect.left).toFixed(2) + 'px';
  ghost.style.top = (rect.top - stageRect.top).toFixed(2) + 'px';
  ghost.style.width = rect.width.toFixed(2) + 'px';
  ghost.style.height = rect.height.toFixed(2) + 'px';
}

function animateAttentionMultiHeadGhost(ghost, sourceEl, targetEl, durationMs, options = {}) {
  const stage = document.getElementById('attn24-stage');
  if (!ghost || !stage || !sourceEl || !targetEl) return false;

  const stageRect = stage.getBoundingClientRect();
  const sourceRect = options.useGhostPosition ? ghost.getBoundingClientRect() : sourceEl.getBoundingClientRect();
  const targetRect = targetEl.getBoundingClientRect();
  if (
    stageRect.width < 1 || stageRect.height < 1
    || sourceRect.width < 1 || sourceRect.height < 1
    || targetRect.width < 1 || targetRect.height < 1
  ) return false;

  const fromX = sourceRect.left - stageRect.left + sourceRect.width * 0.5;
  const fromY = sourceRect.top - stageRect.top + sourceRect.height * 0.5;
  const toX = targetRect.left - stageRect.left + targetRect.width * 0.5;
  const toY = targetRect.top - stageRect.top + targetRect.height * 0.5;
  const dx = toX - fromX;
  const dy = toY - fromY;
  const scaleX = sourceRect.width > 0 ? targetRect.width / sourceRect.width : 1;
  const scaleY = sourceRect.height > 0 ? targetRect.height / sourceRect.height : 1;

  ghost.style.transition = 'transform ' + durationMs + 'ms cubic-bezier(0.2, 0.75, 0.3, 1), opacity ' + ATTN_MATRIX_FADE_MS + 'ms ease';
  const rafId = requestAnimationFrame(() => {
    ghost.style.transform = 'translate3d(' + dx.toFixed(2) + 'px, ' + dy.toFixed(2) + 'px, 0) scale(' + scaleX.toFixed(3) + ', ' + scaleY.toFixed(3) + ')';
  });
  state.attentionMultiHead.rafIds.push(rafId);
  return true;
}

function createAttentionMultiHeadSplitGhost() {
  const stage = document.getElementById('attn24-stage');
  const layer = document.getElementById('attn24-ghost-layer');
  const sourceShell = document.getElementById('attn24-source-shell');
  if (!stage || !layer || !sourceShell) return null;
  const stageRect = stage.getBoundingClientRect();
  const sourceRect = sourceShell.getBoundingClientRect();
  if (stageRect.width < 1 || stageRect.height < 1 || sourceRect.width < 1 || sourceRect.height < 1) return null;
  const ghost = cloneAttentionMultiHeadFragment(sourceShell);
  if (!ghost) return null;
  placeAttentionMultiHeadGhost(ghost, sourceRect, stageRect);
  layer.appendChild(ghost);
  return ghost;
}

function createAttentionMultiHeadOutputGhost(head) {
  const stage = document.getElementById('attn24-stage');
  const layer = document.getElementById('attn24-ghost-layer');
  const sourceShell = document.getElementById('attn24-head-output-shell-' + head);
  if (!stage || !layer || !sourceShell) return null;
  const stageRect = stage.getBoundingClientRect();
  const sourceRect = sourceShell.getBoundingClientRect();
  if (stageRect.width < 1 || stageRect.height < 1 || sourceRect.width < 1 || sourceRect.height < 1) return null;
  const ghost = cloneAttentionMultiHeadFragment(sourceShell, 'attn24-output-ghost');
  if (!ghost) return null;
  placeAttentionMultiHeadGhost(ghost, sourceRect, stageRect);
  layer.appendChild(ghost);
  return ghost;
}

function setAttentionMultiHeadSourceOutputVisible(head, visible) {
  const wrap = document.getElementById('attn24-head-output-wrap-' + head);
  if (!wrap) return;
  wrap.style.opacity = visible ? '' : '0';
  wrap.style.visibility = visible ? '' : 'hidden';
}

function setAttentionMultiHeadOutputRowVisible(head, token, visible) {
  const row = document.getElementById('attn24-head-output-' + head + '-row-' + token);
  if (!row) return;
  row.classList.toggle('is-hidden', !visible);
}

function setAttentionMultiHeadOutputRowActive(head, token, active) {
  const row = document.getElementById('attn24-head-output-' + head + '-row-' + token);
  if (!row) return;
  row.classList.toggle('is-active', !!active);
}

function setAttentionMultiHeadAttentionBlockActive(head, active) {
  const block = document.getElementById('attn24-head-attn-block-' + head);
  if (!block) return;
  block.classList.toggle('is-active', !!active);
}

function clearAttentionMultiHeadActiveRows() {
  ATTN_MHA_HEADS.forEach((head) => {
    ATTN_MHA_TOKENS.forEach((token) => {
      setAttentionMultiHeadOutputRowActive(head, token, false);
    });
    setAttentionMultiHeadAttentionBlockActive(head, false);
  });
}

function syncAttentionMultiHeadOutputRows(visibleCount = state.attentionMultiHead.outputVisibleCount) {
  const clamped = Math.max(0, Math.min(ATTN_MHA_TOKENS.length, visibleCount));
  ATTN_MHA_HEADS.forEach((head) => {
    ATTN_MHA_TOKENS.forEach((token, idx) => {
      setAttentionMultiHeadOutputRowVisible(head, token, idx < clamped);
    });
  });
}

function updateAttentionMultiHeadHeadOverlay(head) {
  const body = document.getElementById('attn24-head-body-' + head);
  const sourceShell = document.getElementById('attn24-head-mini-x-shell-' + head);
  const overlay = document.getElementById('attn24-head-overlay-' + head);
  const sourceLine = document.getElementById('attn24-head-line-source-' + head);
  const busLine = document.getElementById('attn24-head-line-bus-' + head);
  const copyNode = document.getElementById('attn24-head-copy-node-' + head);
  const copyLabel = document.getElementById('attn24-head-copy-label-' + head);
  if (!body || !sourceShell || !overlay || !sourceLine || !busLine || !copyNode || !copyLabel) return;

  const bodyRect = body.getBoundingClientRect();
  if (bodyRect.width < 1 || bodyRect.height < 1) return;

  const point = (el, fx, fy) => {
    const rect = el.getBoundingClientRect();
    return {
      x: rect.left - bodyRect.left + rect.width * fx,
      y: rect.top - bodyRect.top + rect.height * fy
    };
  };
  const setLine = (el, x, y, w, h) => {
    if (!el) return;
    el.style.left = x.toFixed(2) + 'px';
    el.style.top = y.toFixed(2) + 'px';
    el.style.width = Math.max(1, w).toFixed(2) + 'px';
    el.style.height = Math.max(1, h).toFixed(2) + 'px';
  };

  const sourcePoint = point(sourceShell, 1, 0.5);
  const targets = ATTN_MHA_PROJS.map((proj) => {
    const shell = document.getElementById('attn24-head-w-shell-' + proj + '-' + head);
    const line = document.getElementById('attn24-head-line-' + proj + '-' + head);
    if (!shell || !line) return null;
    return {
      line,
      point: point(shell, 0, 0.5)
    };
  }).filter(Boolean);
  if (!targets.length) return;

  const minTargetY = Math.min.apply(null, targets.map((entry) => entry.point.y));
  const maxTargetY = Math.max.apply(null, targets.map((entry) => entry.point.y));
  const nearestTargetX = Math.min.apply(null, targets.map((entry) => entry.point.x));
  const gapWidth = Math.max(1, nearestTargetX - sourcePoint.x);
  const rightBuffer = Math.max(18, Math.min(30, gapWidth * 0.48));
  const busX = Math.max(sourcePoint.x + 12, nearestTargetX - rightBuffer);
  const busTop = Math.min(sourcePoint.y, minTargetY);
  const busBottom = Math.max(sourcePoint.y, maxTargetY);

  setLine(sourceLine, sourcePoint.x, sourcePoint.y - 0.5, Math.max(1, busX - sourcePoint.x), 1.4);
  setLine(busLine, busX - 0.5, busTop, 1.4, Math.max(1, busBottom - busTop));

  copyNode.style.left = busX.toFixed(2) + 'px';
  copyNode.style.top = sourcePoint.y.toFixed(2) + 'px';
  copyLabel.style.left = busX.toFixed(2) + 'px';
  copyLabel.style.top = sourcePoint.y.toFixed(2) + 'px';

  targets.forEach((entry) => {
    setLine(
      entry.line,
      Math.min(busX, entry.point.x),
      entry.point.y - 0.5,
      Math.abs(entry.point.x - busX),
      1.4
    );
  });
}

function updateAttentionMultiHeadHeadOverlays() {
  ATTN_MHA_HEADS.forEach((head) => updateAttentionMultiHeadHeadOverlay(head));
}

function updateAttentionMultiHeadOverlay() {
  const stage = document.getElementById('attn24-stage');
  const overlay = document.getElementById('attn24-overlay');
  const sourceShell = document.getElementById('attn24-source-shell');
  const sourceLine = document.getElementById('attn24-line-source');
  const busTrunkLine = document.getElementById('attn24-line-bus-trunk');
  const busMainLine = document.getElementById('attn24-line-bus-main');
  const copyNode = document.getElementById('attn24-copy-node');
  const copyLabel = document.getElementById('attn24-copy-label');
  if (!stage || !overlay || !sourceShell || !sourceLine || !busTrunkLine || !busMainLine || !copyNode || !copyLabel) return;

  const stageRect = stage.getBoundingClientRect();
  if (stageRect.width < 1 || stageRect.height < 1) return;
  overlay.setAttribute('viewBox', '0 0 ' + stageRect.width + ' ' + stageRect.height);

  const anchor = (el, fx, fy) => {
    const rect = el.getBoundingClientRect();
    return {
      x: rect.left - stageRect.left + rect.width * fx,
      y: rect.top - stageRect.top + rect.height * fy
    };
  };

  const sourceBottom = anchor(sourceShell, 0.5, 1);
  const branchTargets = [];
  ATTN_MHA_HEADS.forEach((head) => {
    const shell = document.getElementById('attn24-head-mini-x-shell-' + head);
    const line = document.getElementById('attn24-line-' + head);
    if (!shell || !line) return;
    branchTargets.push({
      line,
      target: anchor(shell, 0.5, 0)
    });
  });
  if (!branchTargets.length) return;

  const minTargetX = Math.min.apply(null, branchTargets.map((entry) => entry.target.x));
  const maxTargetX = Math.max.apply(null, branchTargets.map((entry) => entry.target.x));
  const minTargetY = Math.min.apply(null, branchTargets.map((entry) => entry.target.y));
  const nodeRadius = Math.max(stageRect.height * 0.015, 6.2);
  const nodeX = sourceBottom.x;
  const nodeY = Math.min(
    minTargetY - Math.max(stageRect.height * 0.08, 28),
    sourceBottom.y + Math.max(stageRect.height * 0.12, 42)
  );
  const busY = Math.max(
    nodeY + nodeRadius + Math.max(stageRect.height * 0.016, 7),
    minTargetY - Math.max(stageRect.height * 0.028, 12)
  );

  copyNode.setAttribute('cx', nodeX.toFixed(2));
  copyNode.setAttribute('cy', nodeY.toFixed(2));
  copyNode.setAttribute('r', nodeRadius.toFixed(2));
  copyLabel.setAttribute('x', nodeX.toFixed(2));
  copyLabel.setAttribute('y', nodeY.toFixed(2));

  sourceLine.setAttribute('x1', nodeX.toFixed(2));
  sourceLine.setAttribute('y1', (sourceBottom.y + 3).toFixed(2));
  sourceLine.setAttribute('x2', nodeX.toFixed(2));
  sourceLine.setAttribute('y2', (nodeY - nodeRadius - 2).toFixed(2));

  busTrunkLine.setAttribute('x1', nodeX.toFixed(2));
  busTrunkLine.setAttribute('y1', (nodeY + nodeRadius + 1).toFixed(2));
  busTrunkLine.setAttribute('x2', nodeX.toFixed(2));
  busTrunkLine.setAttribute('y2', busY.toFixed(2));

  busMainLine.setAttribute('x1', minTargetX.toFixed(2));
  busMainLine.setAttribute('y1', busY.toFixed(2));
  busMainLine.setAttribute('x2', maxTargetX.toFixed(2));
  busMainLine.setAttribute('y2', busY.toFixed(2));

  branchTargets.forEach((entry) => {
    entry.line.setAttribute('x1', entry.target.x.toFixed(2));
    entry.line.setAttribute('y1', busY.toFixed(2));
    entry.line.setAttribute('x2', entry.target.x.toFixed(2));
    entry.line.setAttribute('y2', (entry.target.y - 2).toFixed(2));
  });

  updateAttentionMultiHeadHeadOverlays();
}

function resetAttentionMultiHeadVisuals() {
  const slide = document.getElementById('slide-24');
  clearAttentionMultiHeadTimers();
  clearAttentionMultiHeadGhosts();
  state.attentionMultiHead.splitDone = false;
  state.attentionMultiHead.projDone = false;
  state.attentionMultiHead.attnDone = false;
  state.attentionMultiHead.outputDone = false;
  state.attentionMultiHead.concatDone = false;
  state.attentionMultiHead.outputProjectionDone = false;
  state.attentionMultiHead.combineVisible = false;
  state.attentionMultiHead.outputVisibleCount = 0;
  if (!slide) return;
  slide.classList.remove('attn24-show-split', 'attn24-show-proj', 'attn24-show-attn', 'attn24-show-output', 'attn24-show-concat', 'attn24-show-concat-ready', 'attn24-show-wo');
  slide.querySelectorAll('.attn24-head-card').forEach((card) => {
    card.classList.remove('is-split-visible', 'is-proj-visible', 'is-qkv-visible', 'is-attn-visible', 'is-output-visible');
  });
  syncAttentionMultiHeadOutputRows(0);
  clearAttentionMultiHeadActiveRows();
  ATTN_MHA_HEADS.forEach((head) => setAttentionMultiHeadSourceOutputVisible(head, true));
  updateAttentionMultiHeadOverlay();
}

function settleAttentionMultiHeadSplitState() {
  const slide = document.getElementById('slide-24');
  resetAttentionMultiHeadVisuals();
  if (!slide) return;
  slide.classList.add('attn24-show-split');
  ATTN_MHA_HEADS.forEach((head) => {
    const card = document.getElementById('attn24-head-card-' + head);
    if (card) card.classList.add('is-split-visible');
  });
  state.attentionMultiHead.splitDone = true;
  updateAttentionMultiHeadOverlay();
}

function settleAttentionMultiHeadProjectionState() {
  const slide = document.getElementById('slide-24');
  settleAttentionMultiHeadSplitState();
  if (!slide) return;
  slide.classList.add('attn24-show-proj');
  ATTN_MHA_HEADS.forEach((head) => {
    const card = document.getElementById('attn24-head-card-' + head);
    if (!card) return;
    card.classList.add('is-proj-visible', 'is-qkv-visible');
  });
  state.attentionMultiHead.splitDone = true;
  state.attentionMultiHead.projDone = true;
  updateAttentionMultiHeadOverlay();
}

function settleAttentionMultiHeadAttentionState() {
  const slide = document.getElementById('slide-24');
  settleAttentionMultiHeadProjectionState();
  if (!slide) return;
  slide.classList.add('attn24-show-attn');
  ATTN_MHA_HEADS.forEach((head) => {
    const card = document.getElementById('attn24-head-card-' + head);
    if (card) card.classList.add('is-attn-visible');
  });
  state.attentionMultiHead.splitDone = true;
  state.attentionMultiHead.projDone = true;
  state.attentionMultiHead.attnDone = true;
  updateAttentionMultiHeadOverlay();
}

function settleAttentionMultiHeadOutputState() {
  const slide = document.getElementById('slide-24');
  settleAttentionMultiHeadAttentionState();
  if (!slide) return;
  slide.classList.add('attn24-show-output');
  ATTN_MHA_HEADS.forEach((head) => {
    const card = document.getElementById('attn24-head-card-' + head);
    if (card) card.classList.add('is-output-visible');
  });
  state.attentionMultiHead.splitDone = true;
  state.attentionMultiHead.projDone = true;
  state.attentionMultiHead.attnDone = true;
  state.attentionMultiHead.outputDone = true;
  state.attentionMultiHead.outputVisibleCount = ATTN_MHA_TOKENS.length;
  syncAttentionMultiHeadOutputRows();
  clearAttentionMultiHeadActiveRows();
  ATTN_MHA_HEADS.forEach((head) => setAttentionMultiHeadSourceOutputVisible(head, true));
  updateAttentionMultiHeadOverlay();
}

function settleAttentionMultiHeadConcatState() {
  const slide = document.getElementById('slide-24');
  settleAttentionMultiHeadOutputState();
  if (!slide) return;
  slide.classList.add('attn24-show-concat', 'attn24-show-concat-ready');
  slide.classList.remove('attn24-show-wo');
  state.attentionMultiHead.splitDone = true;
  state.attentionMultiHead.projDone = true;
  state.attentionMultiHead.attnDone = true;
  state.attentionMultiHead.outputDone = true;
  state.attentionMultiHead.concatDone = true;
  state.attentionMultiHead.combineVisible = true;
  state.attentionMultiHead.outputProjectionDone = false;
  ATTN_MHA_HEADS.forEach((head) => setAttentionMultiHeadSourceOutputVisible(head, true));
  updateAttentionMultiHeadOverlay();
}

function settleAttentionMultiHeadOutputProjectionState() {
  const slide = document.getElementById('slide-24');
  settleAttentionMultiHeadConcatState();
  if (!slide) return;
  slide.classList.add('attn24-show-wo');
  state.attentionMultiHead.concatDone = true;
  state.attentionMultiHead.combineVisible = true;
  state.attentionMultiHead.outputProjectionDone = true;
  ATTN_MHA_HEADS.forEach((head) => setAttentionMultiHeadSourceOutputVisible(head, true));
  updateAttentionMultiHeadOverlay();
}

function runAttentionMultiHeadSplitSequence() {
  const slide = document.getElementById('slide-24');
  if (!slide) return;
  resetAttentionMultiHeadVisuals();
  slide.classList.add('attn24-show-split');
  updateAttentionMultiHeadOverlay();

  ATTN_MHA_HEADS.forEach((head, idx) => {
    const card = document.getElementById('attn24-head-card-' + head);
    const source = document.getElementById('attn24-source-shell');
    const target = document.getElementById('attn24-head-mini-x-shell-' + head);
    const ghost = createAttentionMultiHeadSplitGhost();
    if (ghost && source && target) {
      animateAttentionMultiHeadGhost(ghost, source, target, ATTN_MHA_SPLIT_MS);
      state.attentionMultiHead.timers.push(setTimeout(() => {
        if (card) card.classList.add('is-split-visible');
      }, Math.max(ATTN_MHA_SPLIT_MS - 110, 140) + (idx * 30)));
      state.attentionMultiHead.timers.push(setTimeout(() => {
        ghost.remove();
      }, ATTN_MHA_SPLIT_MS + ATTN_MATRIX_FADE_MS + 40));
    } else if (card) {
      card.classList.add('is-split-visible');
    }
  });

  state.attentionMultiHead.timers.push(setTimeout(() => {
    settleAttentionMultiHeadSplitState();
  }, ATTN_MHA_SPLIT_MS + ATTN_MATRIX_FADE_MS + 80));
}

function runAttentionMultiHeadProjectionSequence() {
  const slide = document.getElementById('slide-24');
  if (!slide) return;
  if (!state.attentionMultiHead.splitDone) {
    settleAttentionMultiHeadSplitState();
  }
  clearAttentionMultiHeadTimers();
  clearAttentionMultiHeadGhosts();
  slide.classList.add('attn24-show-split');
  slide.classList.add('attn24-show-proj');

  ATTN_MHA_HEADS.forEach((head, idx) => {
    const card = document.getElementById('attn24-head-card-' + head);
    if (!card) return;
    state.attentionMultiHead.timers.push(setTimeout(() => {
      card.classList.add('is-split-visible', 'is-proj-visible');
    }, idx * ATTN_MHA_PROJ_STAGGER_MS));
    state.attentionMultiHead.timers.push(setTimeout(() => {
      card.classList.add('is-qkv-visible');
    }, (idx * ATTN_MHA_PROJ_STAGGER_MS) + ATTN_MHA_PROJ_REVEAL_MS));
  });

  state.attentionMultiHead.timers.push(setTimeout(() => {
    settleAttentionMultiHeadProjectionState();
  }, ((ATTN_MHA_HEADS.length - 1) * ATTN_MHA_PROJ_STAGGER_MS) + ATTN_MHA_PROJ_REVEAL_MS + ATTN_MATRIX_FADE_MS + 40));
}

function runAttentionMultiHeadAttentionSequence() {
  const slide = document.getElementById('slide-24');
  if (!slide) return;
  if (!state.attentionMultiHead.projDone) {
    settleAttentionMultiHeadProjectionState();
  }
  clearAttentionMultiHeadTimers();
  clearAttentionMultiHeadGhosts();
  slide.classList.add('attn24-show-split', 'attn24-show-proj', 'attn24-show-attn');

  ATTN_MHA_HEADS.forEach((head, idx) => {
    const card = document.getElementById('attn24-head-card-' + head);
    if (!card) return;
    state.attentionMultiHead.timers.push(setTimeout(() => {
      card.classList.add('is-attn-visible');
    }, idx * ATTN_MHA_ATTN_STAGGER_MS));
  });

  state.attentionMultiHead.timers.push(setTimeout(() => {
    settleAttentionMultiHeadAttentionState();
  }, ((ATTN_MHA_HEADS.length - 1) * ATTN_MHA_ATTN_STAGGER_MS) + ATTN_MHA_ATTN_REVEAL_MS + 40));
}

function runAttentionMultiHeadOutputSequence() {
  const slide = document.getElementById('slide-24');
  if (!slide) return;
  if (!state.attentionMultiHead.attnDone) {
    settleAttentionMultiHeadAttentionState();
  }
  clearAttentionMultiHeadTimers();
  clearAttentionMultiHeadGhosts();
  slide.classList.add('attn24-show-split', 'attn24-show-proj', 'attn24-show-attn', 'attn24-show-output');
  ATTN_MHA_HEADS.forEach((head) => {
    const card = document.getElementById('attn24-head-card-' + head);
    if (card) card.classList.add('is-output-visible');
  });

  state.attentionMultiHead.outputVisibleCount = 0;
  syncAttentionMultiHeadOutputRows(0);
  clearAttentionMultiHeadActiveRows();

  ATTN_MHA_TOKENS.forEach((token, idx) => {
    state.attentionMultiHead.timers.push(setTimeout(() => {
      state.attentionMultiHead.outputVisibleCount = idx + 1;
      syncAttentionMultiHeadOutputRows();
      ATTN_MHA_HEADS.forEach((head) => {
        setAttentionMultiHeadAttentionBlockActive(head, true);
        setAttentionMultiHeadOutputRowActive(head, token, true);
      });
    }, 40 + (idx * ATTN_MHA_OUTPUT_ROW_STAGGER_MS)));

    state.attentionMultiHead.timers.push(setTimeout(() => {
      ATTN_MHA_HEADS.forEach((head) => {
        setAttentionMultiHeadAttentionBlockActive(head, false);
        setAttentionMultiHeadOutputRowActive(head, token, false);
      });
    }, 40 + (idx * ATTN_MHA_OUTPUT_ROW_STAGGER_MS) + ATTN_MHA_OUTPUT_ROW_MS));
  });

  state.attentionMultiHead.timers.push(setTimeout(() => {
    settleAttentionMultiHeadOutputState();
  }, 40 + ((ATTN_MHA_TOKENS.length - 1) * ATTN_MHA_OUTPUT_ROW_STAGGER_MS) + ATTN_MHA_OUTPUT_ROW_MS + 60));
}

function runAttentionMultiHeadConcatSequence() {
  const slide = document.getElementById('slide-24');
  if (!slide) return;
  if (!state.attentionMultiHead.outputDone) {
    settleAttentionMultiHeadOutputState();
  }
  clearAttentionMultiHeadTimers();
  clearAttentionMultiHeadGhosts();
  slide.classList.add('attn24-show-concat');
  slide.classList.remove('attn24-show-wo');

  ATTN_MHA_HEADS.forEach((head) => {
    const sourceShell = document.getElementById('attn24-head-output-shell-' + head);
    const target = document.getElementById(head === 'h1' ? 'attn24-concat-target-left' : 'attn24-concat-target-right');
    const ghost = createAttentionMultiHeadOutputGhost(head);
    setAttentionMultiHeadSourceOutputVisible(head, false);
    if (ghost && sourceShell && target) {
      animateAttentionMultiHeadGhost(ghost, sourceShell, target, ATTN_MHA_CONCAT_MS);
      state.attentionMultiHead.timers.push(setTimeout(() => {
        ghost.remove();
      }, ATTN_MHA_CONCAT_MS + ATTN_MATRIX_FADE_MS + 40));
    }
  });

  state.attentionMultiHead.timers.push(setTimeout(() => {
    slide.classList.add('attn24-show-concat-ready');
  }, Math.max(ATTN_MHA_CONCAT_MS - 120, 180)));

  state.attentionMultiHead.timers.push(setTimeout(() => {
    settleAttentionMultiHeadConcatState();
  }, ATTN_MHA_CONCAT_MS + ATTN_MATRIX_FADE_MS + 80));
}

function runAttentionMultiHeadOutputProjectionSequence() {
  const slide = document.getElementById('slide-24');
  if (!slide) return;
  if (!state.attentionMultiHead.concatDone) {
    settleAttentionMultiHeadConcatState();
  }
  clearAttentionMultiHeadTimers();
  clearAttentionMultiHeadGhosts();
  slide.classList.add('attn24-show-concat', 'attn24-show-concat-ready');
  slide.classList.remove('attn24-show-wo');

  state.attentionMultiHead.timers.push(setTimeout(() => {
    slide.classList.add('attn24-show-wo');
  }, 30));

  state.attentionMultiHead.timers.push(setTimeout(() => {
    settleAttentionMultiHeadOutputProjectionState();
  }, ATTN_MHA_WO_REVEAL_MS + 60));
}

function initAttentionMultiHeadSlide() {
  const slide = document.getElementById('slide-24');
  const sourceWrap = document.getElementById('attn24-source-wrap');
  const headGrid = document.getElementById('attn24-head-grid');
  const combineStage = document.getElementById('attn24-combine-stage');
  if (!slide || !sourceWrap || !headGrid || !combineStage) return;

  if (!state.attentionMultiHead.initialized) {
    sourceWrap.innerHTML = '';
    headGrid.innerHTML = '';
    combineStage.innerHTML = '';
    sourceWrap.appendChild(createAttentionMultiHeadSourceMatrix());
    ATTN_MHA_HEADS.forEach((head) => {
      headGrid.appendChild(createAttentionMultiHeadHeadCard(head));
    });
    combineStage.appendChild(createAttentionMultiHeadCombineStage());

    if (!state.attentionMultiHead.resizeBound) {
      addTrackedListener(window, 'resize', () => {
        if (!state.attentionMultiHead.initialized) return;
        if (state.attentionMultiHead.step >= 6) {
          settleAttentionMultiHeadOutputProjectionState();
        } else if (state.attentionMultiHead.step >= 5) {
          settleAttentionMultiHeadConcatState();
        } else if (state.attentionMultiHead.step >= 4) {
          settleAttentionMultiHeadOutputState();
        } else if (state.attentionMultiHead.step >= 3) {
          settleAttentionMultiHeadAttentionState();
        } else if (state.attentionMultiHead.step >= 2) {
          settleAttentionMultiHeadProjectionState();
        } else if (state.attentionMultiHead.step >= 1) {
          settleAttentionMultiHeadSplitState();
        } else {
          resetAttentionMultiHeadVisuals();
        }
      });
      state.attentionMultiHead.resizeBound = true;
    }

    state.attentionMultiHead.initialized = true;
  }

  const takeaway = document.getElementById('attn24-takeaway');
  if (takeaway) setMathHTML(takeaway, ATTN_MHA_TAKEAWAYS[state.attentionMultiHead.step] || ATTN_MHA_TAKEAWAYS[0]);
  typesetMath(slide);

  if (state.attentionMultiHead.step >= 6) {
    settleAttentionMultiHeadOutputProjectionState();
  } else if (state.attentionMultiHead.step >= 5) {
    settleAttentionMultiHeadConcatState();
  } else if (state.attentionMultiHead.step >= 4) {
    settleAttentionMultiHeadOutputState();
  } else if (state.attentionMultiHead.step >= 3) {
    settleAttentionMultiHeadAttentionState();
  } else if (state.attentionMultiHead.step >= 2) {
    settleAttentionMultiHeadProjectionState();
  } else if (state.attentionMultiHead.step >= 1) {
    settleAttentionMultiHeadSplitState();
  } else {
    resetAttentionMultiHeadVisuals();
  }
}

function setAttentionMultiHeadStep(step) {
  const slide = document.getElementById('slide-24');
  const takeaway = document.getElementById('attn24-takeaway');
  if (!slide || !takeaway) return;

  const prevStep = state.attentionMultiHead.step;
  const clamped = Math.max(0, Math.min(ATTN_MHA_MAX_STEP, step));
  state.attentionMultiHead.step = clamped;
  setMathHTML(takeaway, ATTN_MHA_TAKEAWAYS[clamped] || ATTN_MHA_TAKEAWAYS[0]);
  const animateStep = clamped === prevStep + 1;

  if (clamped === 0) {
    resetAttentionMultiHeadVisuals();
    return;
  }

  if (clamped === 1) {
    if (!state.attentionMultiHead.splitDone && animateStep) {
      runAttentionMultiHeadSplitSequence();
    } else {
      settleAttentionMultiHeadSplitState();
    }
    return;
  }

  if (!state.attentionMultiHead.splitDone) {
    settleAttentionMultiHeadSplitState();
  }
  if (clamped === 2) {
    if (!state.attentionMultiHead.projDone && animateStep) {
      runAttentionMultiHeadProjectionSequence();
    } else {
      settleAttentionMultiHeadProjectionState();
    }
    return;
  }

  if (!state.attentionMultiHead.projDone) {
    settleAttentionMultiHeadProjectionState();
  }
  if (clamped === 3) {
    if (!state.attentionMultiHead.attnDone && animateStep) {
      runAttentionMultiHeadAttentionSequence();
    } else {
      settleAttentionMultiHeadAttentionState();
    }
    return;
  }

  if (!state.attentionMultiHead.attnDone) {
    settleAttentionMultiHeadAttentionState();
  }
  if (clamped === 4) {
    if (!state.attentionMultiHead.outputDone && animateStep) {
      runAttentionMultiHeadOutputSequence();
    } else {
      settleAttentionMultiHeadOutputState();
    }
    return;
  }

  if (!state.attentionMultiHead.outputDone) {
    settleAttentionMultiHeadOutputState();
  }
  if (clamped === 5) {
    if (!state.attentionMultiHead.concatDone && animateStep) {
      runAttentionMultiHeadConcatSequence();
    } else {
      settleAttentionMultiHeadConcatState();
    }
    return;
  }

  if (!state.attentionMultiHead.concatDone) {
    settleAttentionMultiHeadConcatState();
  }
  if (clamped === 6) {
    if (!state.attentionMultiHead.outputProjectionDone && animateStep) {
      runAttentionMultiHeadOutputProjectionSequence();
    } else {
      settleAttentionMultiHeadOutputProjectionState();
    }
  }
}

function runAttentionMultiHeadStep() {
  if (!state.attentionMultiHead.initialized) initAttentionMultiHeadSlide();
  if (state.attentionMultiHead.step >= ATTN_MHA_MAX_STEP) return false;
  setAttentionMultiHeadStep(state.attentionMultiHead.step + 1);
  return true;
}

function resetAttentionMultiHeadSlide() {
  const slide = document.getElementById('slide-24');
  if (!slide) return;
  setAttentionMultiHeadStep(0);
}

/* =====================================================
   Slide-25 — Positional Encoding
   ===================================================== */
