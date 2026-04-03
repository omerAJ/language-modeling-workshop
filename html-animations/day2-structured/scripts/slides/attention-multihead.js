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

function createAttentionMultiHeadDummyX(head, proj) {
  const wrap = createEl('div', {
    className: 'attn24-head-dummy-x-wrap',
    id: 'attn24-head-dummy-x-wrap-' + proj + '-' + head,
    dataset: { proj, head }
  });
  const shell = createEl('div', {
    className: 'attn24-head-dummy-x-shell',
    id: 'attn24-head-dummy-x-shell-' + proj + '-' + head
  });
  const grid = createEl('div', {
    className: 'attn24-head-dummy-x-grid'
  });
  [25, 50, 75].forEach((pct) => {
    grid.appendChild(createEl('span', {
      className: 'attn24-head-dummy-x-grid-v',
      style: { left: pct + '%' }
    }));
  });
  [20, 40, 60, 80].forEach((pct) => {
    grid.appendChild(createEl('span', {
      className: 'attn24-head-dummy-x-grid-h',
      style: { top: pct + '%' }
    }));
  });
  shell.appendChild(grid);
  wrap.appendChild(shell);
  wrap.appendChild(createEl('div', {
    className: 'attn24-head-dummy-x-label',
    html: inlineMath('X')
  }));
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
  [20, 40, 60, 80].forEach((pct) => {
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
  return wrap;
}

function createAttentionMultiHeadProjectionRow(head, proj) {
  const row = createEl('div', {
    className: 'attn24-head-proj-row',
    id: 'attn24-head-proj-row-' + proj + '-' + head,
    dataset: { head, proj }
  });
  row.appendChild(createAttentionMultiHeadDummyX(head, proj));
  row.appendChild(createEl('div', {
    className: 'attn24-head-proj-op',
    text: '×'
  }));
  row.appendChild(createAttentionMultiHeadWMatrix(head, proj));
  row.appendChild(createEl('div', {
    className: 'attn24-head-proj-op',
    text: '='
  }));
  row.appendChild(createAttentionMultiHeadQkvMatrix(head, proj));
  return row;
}

function createAttentionMultiHeadHeadOverlay(head) {
  const overlay = svgEl('svg', {
    class: 'attn24-head-overlay',
    id: 'attn24-head-overlay-' + head,
    ariaHidden: 'true'
  });
  overlay.appendChild(svgEl('line', {
    class: 'attn24-head-overlay-line',
    id: 'attn24-head-line-source-' + head
  }));
  overlay.appendChild(svgEl('path', {
    class: 'attn24-head-overlay-bus',
    id: 'attn24-head-line-bus-' + head
  }));
  ATTN_MHA_PROJS.forEach((proj) => {
    overlay.appendChild(svgEl('path', {
      class: 'attn24-head-overlay-line',
      id: 'attn24-head-line-' + proj + '-' + head
    }));
  });
  const badge = svgEl('g', {
    class: 'attn24-head-copy-badge',
    id: 'attn24-head-copy-badge-' + head
  });
  badge.appendChild(svgEl('rect', {
    class: 'attn24-head-copy-pill',
    id: 'attn24-head-copy-pill-' + head
  }));
  badge.appendChild(svgEl('text', {
    class: 'attn24-head-copy-label',
    id: 'attn24-head-copy-label-' + head,
    text: '\u00d7' + String(ATTN_MHA_PROJS.length)
  }));
  overlay.appendChild(badge);
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

function syncAttentionMultiHeadCaptions() {
  ATTN_MHA_HEADS.forEach((head) => {
    const cap = document.getElementById('attn24-head-caption-' + head);
    if (!cap) return;
    const titleEl = cap.querySelector('.attn24-head-caption-title');
    const metaEl = cap.querySelector('.attn24-head-caption-meta');
    if (titleEl) titleEl.textContent = ATTN_MHA_HEAD_LABELS[head] || head;
    if (metaEl) {
      metaEl.replaceChildren(
        'd',
        createEl('sub', { text: 'h' }),
        ' = ',
        String(ATTN_MHA_HEAD_DIM)
      );
    }
  });
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
  const sc = layoutState.currentScale || 1;
  ghost.style.left = ((rect.left - stageRect.left) / sc).toFixed(2) + 'px';
  ghost.style.top = ((rect.top - stageRect.top) / sc).toFixed(2) + 'px';
  ghost.style.width = (rect.width / sc).toFixed(2) + 'px';
  ghost.style.height = (rect.height / sc).toFixed(2) + 'px';
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

  const sc = layoutState.currentScale || 1;
  const fromX = sourceRect.left - stageRect.left + sourceRect.width * 0.5;
  const fromY = sourceRect.top - stageRect.top + sourceRect.height * 0.5;
  const toX = targetRect.left - stageRect.left + targetRect.width * 0.5;
  const toY = targetRect.top - stageRect.top + targetRect.height * 0.5;
  const dx = (toX - fromX) / sc;
  const dy = (toY - fromY) / sc;
  const scaleX = sourceRect.width > 0 ? targetRect.width / sourceRect.width : 1;
  const scaleY = sourceRect.height > 0 ? targetRect.height / sourceRect.height : 1;

  const transformVal = 'translate3d(' + dx.toFixed(2) + 'px, ' + dy.toFixed(2) + 'px, 0) scale(' + scaleX.toFixed(3) + ', ' + scaleY.toFixed(3) + ')';
  const onDone = options.onDone;

  ghost.style.transition = 'none';
  ghost.style.transform = 'translate3d(0,0,0) scale(1,1)';
  void ghost.getBoundingClientRect();

  /* Double-RAF: first frame commits the no-transform + transition property;
     second frame applies the target transform so the browser sees an actual
     property change and starts the CSS transition reliably. */
  const raf1 = requestAnimationFrame(() => {
    ghost.style.transition = 'transform ' + durationMs + 'ms cubic-bezier(0.2, 0.75, 0.3, 1), opacity ' + ATTN_MATRIX_FADE_MS + 'ms ease';
    const raf2 = requestAnimationFrame(() => {
      ghost.style.transform = transformVal;

      if (typeof onDone === 'function') {
        let fired = false;
        const finish = () => {
          if (fired) return;
          fired = true;
          ghost.removeEventListener('transitionend', onTransitionEnd);
          onDone();
        };
        const onTransitionEnd = (e) => {
          if (e.propertyName === 'transform') finish();
        };
        ghost.addEventListener('transitionend', onTransitionEnd);
        const safetyId = setTimeout(finish, durationMs + 200);
        state.attentionMultiHead.timers.push(safetyId);
      }
    });
    state.attentionMultiHead.rafIds.push(raf2);
  });
  state.attentionMultiHead.rafIds.push(raf1);
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

function setAttentionMultiHeadHeadSourceVisible(head, visible) {
  const wrap = document.getElementById('attn24-head-mini-x-wrap-' + head);
  if (!wrap) return;
  wrap.style.opacity = visible ? '1' : '0';
  wrap.style.visibility = visible ? 'visible' : 'hidden';
  wrap.style.transform = visible ? 'translateY(0)' : 'translateY(0.18rem)';
}

function setAttentionMultiHeadDummyCopiesVisible(head, visible) {
  ATTN_MHA_PROJS.forEach((proj) => {
    const wrap = document.getElementById('attn24-head-dummy-x-wrap-' + proj + '-' + head);
    if (!wrap) return;
    wrap.style.opacity = visible ? '1' : '0';
    wrap.style.visibility = visible ? 'visible' : 'hidden';
    wrap.style.transform = visible ? 'translateY(0)' : 'translateY(0.18rem)';
  });
}

function setAttentionMultiHeadHeadOverlayVisible(head, visible) {
  const overlay = document.getElementById('attn24-head-overlay-' + head);
  if (!overlay) return;
  overlay.style.opacity = visible ? '1' : '0';
  overlay.querySelectorAll('.attn24-head-overlay-line, .attn24-head-overlay-bus, .attn24-head-copy-badge').forEach((el) => {
    el.style.opacity = visible ? '1' : '0';
    el.style.visibility = visible ? 'visible' : 'hidden';
  });
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
  const copyPill = document.getElementById('attn24-head-copy-pill-' + head);
  const copyLabel = document.getElementById('attn24-head-copy-label-' + head);
  if (!body || !sourceShell || !overlay || !sourceLine || !busLine || !copyPill || !copyLabel) return;

  const bodyRect = body.getBoundingClientRect();
  if (bodyRect.width < 1 || bodyRect.height < 1) return;
  overlay.setAttribute('viewBox', '0 0 ' + bodyRect.width + ' ' + bodyRect.height);

  const point = (el, fx, fy) => {
    const rect = el.getBoundingClientRect();
    return {
      x: rect.left - bodyRect.left + rect.width * fx,
      y: rect.top - bodyRect.top + rect.height * fy
    };
  };
  const setLine = (el, from, to) => {
    if (!el) return;
    el.setAttribute('x1', from.x.toFixed(2));
    el.setAttribute('y1', from.y.toFixed(2));
    el.setAttribute('x2', to.x.toFixed(2));
    el.setAttribute('y2', to.y.toFixed(2));
  };
  const setPath = (el, points) => {
    if (!el) return;
    el.setAttribute(
      'd',
      points.map((pt, idx) => (
        (idx === 0 ? 'M' : ' L') + pt.x.toFixed(2) + ',' + pt.y.toFixed(2)
      )).join('')
    );
  };

  const sourcePoint = point(sourceShell, 1, 0.5);
  const targets = ATTN_MHA_PROJS.map((proj) => {
    const dummyShell = document.getElementById('attn24-head-dummy-x-shell-' + proj + '-' + head);
    const branchLine = document.getElementById('attn24-head-line-' + proj + '-' + head);
    if (!dummyShell || !branchLine) return null;
    return {
      branchLine,
      dummyLeft: point(dummyShell, 0, 0.5)
    };
  }).filter(Boolean);
  if (!targets.length) return;

  const ys = targets.map((entry) => entry.dummyLeft.y);
  const maxTargetY = Math.max.apply(null, ys);
  const minTargetY = Math.min.apply(null, ys);
  const nearestTargetX = Math.min.apply(null, targets.map((entry) => entry.dummyLeft.x));
  const gapWidth = Math.max(1, nearestTargetX - sourcePoint.x);
  const remPx = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
  const pillH = Math.max(remPx * 1.1, 18);
  const pillW = Math.max(remPx * 1.1, 38);
  const rx = pillH * 0.5;
  const busX = sourcePoint.x + Math.max(14, Math.min(32, gapWidth * 0.4));
  const pillLeft = busX;
  const pillRight = busX + pillW;
  const pillTop = sourcePoint.y - pillH * 0.5;
  const stubLen = Math.max(remPx * 0.55, 9);
  const busColX = pillRight + stubLen;
  const yTrunkTop = Math.min(sourcePoint.y, minTargetY);
  const yTrunkBot = Math.max(sourcePoint.y, maxTargetY);

  setLine(sourceLine, sourcePoint, { x: pillLeft, y: sourcePoint.y });
  /* L-shaped bus: short horizontal stub from pill right, then vertical trunk */
  {
    const sy = sourcePoint.y.toFixed(2);
    const ex = busColX.toFixed(2);
    let d = `M ${pillRight.toFixed(2)},${sy} L ${ex},${sy} L ${ex},${yTrunkBot.toFixed(2)}`;
    if (yTrunkTop < sourcePoint.y) {
      d += ` M ${ex},${sourcePoint.y.toFixed(2)} L ${ex},${yTrunkTop.toFixed(2)}`;
    }
    busLine.setAttribute('d', d);
  }

  copyPill.setAttribute('x', pillLeft.toFixed(2));
  copyPill.setAttribute('y', pillTop.toFixed(2));
  copyPill.setAttribute('width', pillW.toFixed(2));
  copyPill.setAttribute('height', pillH.toFixed(2));
  copyPill.setAttribute('rx', rx.toFixed(2));
  copyPill.setAttribute('ry', rx.toFixed(2));
  copyLabel.setAttribute('x', (pillLeft + pillW * 0.5).toFixed(2));
  copyLabel.setAttribute('y', sourcePoint.y.toFixed(2));
  copyLabel.style.fontSize = Math.round(pillH * 0.56) + 'px';

  targets.forEach((entry) => {
    const y = entry.dummyLeft.y;
    const xEnd = entry.dummyLeft.x;
    setPath(entry.branchLine, [
      { x: busColX, y },
      { x: xEnd, y }
    ]);
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

  const xBottom = anchor(sourceShell, 0.5, 1);
  const linePad = Math.max(stageRect.height * 0.009, 3.5);
  const branchTargets = [];
  ATTN_MHA_HEADS.forEach((head) => {
    const shell = document.getElementById('attn24-head-mini-x-shell-' + head);
    const line = document.getElementById('attn24-line-' + head);
    const card = document.getElementById('attn24-head-card-' + head);
    if (!shell || !line) return;
    branchTargets.push({
      line,
      target: anchor(shell, 0.5, 0),
      capTopY: card ? anchor(card, 0.5, 0).y : anchor(shell, 0.5, 0).y
    });
  });
  if (!branchTargets.length) return;

  const nearestCopyY = Math.min.apply(null, branchTargets.map((entry) => entry.target.y));
  const nearestCapY = Math.min.apply(null, branchTargets.map((entry) => entry.capTopY));
  const minCopyX = Math.min.apply(null, branchTargets.map((entry) => entry.target.x));
  const maxCopyX = Math.max.apply(null, branchTargets.map((entry) => entry.target.x));
  const nodeX = xBottom.x;
  const nodeRadius = Math.max(stageRect.height * 0.018, 6.4);
  const minNodeGap = Math.max(stageRect.height * 0.3, 84);
  const maxNodeY = nearestCapY - nodeRadius - Math.max(stageRect.height * 0.045, 9);
  const nodeY = Math.max(
    xBottom.y + nodeRadius + 2,
    Math.min(xBottom.y + minNodeGap, maxNodeY)
  );
  const busStartX = minCopyX;
  const busEndX = maxCopyX;
  const busClearBelowNode = Math.max(stageRect.height * 0.02, 6);
  const busMinY = nodeY + nodeRadius + busClearBelowNode;
  const busMaxY = nearestCapY - Math.max(stageRect.height * 0.026, 9);
  const busPreferredY = Math.min(
    nearestCopyY - Math.max(stageRect.height * 0.028, 11),
    nodeY + Math.max(stageRect.height * 0.072, 18)
  );
  let busY = Math.max(busMinY, Math.min(busPreferredY, busMaxY));
  if (!Number.isFinite(busY)) busY = busMinY;
  if (busY < busMinY) busY = busMinY;
  if (busY > busMaxY) busY = busMaxY;

  const remPx = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
  busY = Math.min(busY + remPx * 1, nearestCopyY - 4);

  const nodeYShift = remPx * 0.5;
  copyNode.setAttribute('cx', nodeX.toFixed(2));
  copyNode.setAttribute('cy', (nodeY + nodeYShift).toFixed(2));
  copyNode.setAttribute('r', nodeRadius.toFixed(2));
  copyLabel.setAttribute('x', nodeX.toFixed(2));
  copyLabel.setAttribute('y', (nodeY + nodeYShift).toFixed(2));

  const sourceY1 = xBottom.y + linePad;
  const sourceY2 = nodeY - nodeRadius - Math.max(stageRect.height * 0.006, 2);
  sourceLine.setAttribute('x1', nodeX.toFixed(2));
  sourceLine.setAttribute('y1', sourceY1.toFixed(2));
  sourceLine.setAttribute('x2', nodeX.toFixed(2));
  sourceLine.setAttribute('y2', sourceY2.toFixed(2));

  busTrunkLine.setAttribute('x1', nodeX.toFixed(2));
  busTrunkLine.setAttribute('y1', (nodeY + nodeYShift + nodeRadius + Math.max(stageRect.height * 0.004, 1.2)).toFixed(2));
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

  updateAttentionMultiHeadHeadOverlays();
}

const ATTN24_BUS_LINE_IDS = [
  'attn24-line-source',
  'attn24-line-bus-trunk',
  'attn24-line-bus-main',
  'attn24-line-h1',
  'attn24-line-h2'
];

const ATTN24_BUS_DRAW_EASE = 'cubic-bezier(0.34, 0.08, 0.22, 1)';

function attentionMultiHeadSvgLineTotalLength(el) {
  if (!el) return 0;
  if (typeof el.getTotalLength === 'function') {
    try {
      const L = el.getTotalLength();
      if (L > 0.5) return L;
    } catch (err) {
      /* fall through */
    }
  }
  const x1 = parseFloat(el.getAttribute('x1'));
  const y1 = parseFloat(el.getAttribute('y1'));
  const x2 = parseFloat(el.getAttribute('x2'));
  const y2 = parseFloat(el.getAttribute('y2'));
  if ([x1, y1, x2, y2].some((n) => Number.isNaN(n))) return 0;
  const len = Math.hypot(x2 - x1, y2 - y1);
  return len > 0.5 ? len : 1;
}

function clearAttentionMultiHeadStageBusDrawStyles() {
  ATTN24_BUS_LINE_IDS.forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.transition = '';
    el.style.strokeDasharray = '';
    el.style.strokeDashoffset = '';
  });
  ATTN_MHA_HEADS.forEach((head) => {
    const el = document.getElementById('attn24-line-' + head);
    if (el) el.setAttribute('marker-end', 'url(#attn24-arr-marker)');
  });
  const node = document.getElementById('attn24-copy-node');
  const label = document.getElementById('attn24-copy-label');
  if (node) {
    node.style.transition = '';
    node.style.opacity = '';
  }
  if (label) {
    label.style.transition = '';
    label.style.opacity = '';
  }
}

function prepAttentionMultiHeadStageBusStrokeHidden() {
  /* Use a large safe value — actual length is measured per-segment at animation time */
  ATTN24_BUS_LINE_IDS.forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.transition = 'none';
    el.style.strokeDasharray = '9999';
    el.style.strokeDashoffset = '9999';
    if (id === 'attn24-line-h1' || id === 'attn24-line-h2') {
      el.setAttribute('marker-end', 'none');
    }
  });
  const node = document.getElementById('attn24-copy-node');
  const label = document.getElementById('attn24-copy-label');
  if (node) {
    node.style.transition = 'none';
    node.style.opacity = '0';
  }
  if (label) {
    label.style.transition = 'none';
    label.style.opacity = '0';
  }
}

/**
 * Hide the element in the CURRENT frame, then start the draw animation in the NEXT frame.
 * The RAF boundary is a guaranteed style flush for SVG stroke properties.
 * getBoundingClientRect() is NOT sufficient for stroke-dashoffset — it only flushes layout.
 */
function runAttentionMultiHeadStageLineDraw(el, drawMs) {
  if (!el) return;
  const L = attentionMultiHeadSvgLineTotalLength(el);
  if (L < 0.5) return;
  const dash = L.toFixed(2);
  /* 1 — commit hidden state THIS frame (transition: none so no CSS cascade interference) */
  el.style.transition = 'none';
  el.style.strokeDasharray = dash;
  el.style.strokeDashoffset = dash;
  /* 2 — start the animation in the NEXT frame; the frame boundary IS the flush */
  const rafId = requestAnimationFrame(() => {
    el.style.transition = 'stroke-dashoffset ' + drawMs + 'ms ' + ATTN24_BUS_DRAW_EASE;
    el.style.strokeDashoffset = '0';
  });
  state.attentionMultiHead.rafIds.push(rafId);
}

function scheduleAttentionMultiHeadStageBusDrawChain(slide) {
  const drawMs = ATTN_QKV_COMPARE_DRAW_MS;
  const gap = ATTN_MHA_BUS_SEGMENT_GAP_MS;

  const push = (fn, delay) => {
    state.attentionMultiHead.timers.push(
      setTimeout(() => {
        if (!slide.classList.contains('attn24-show-split') || slide.classList.contains('attn24-source-collapsed')) return;
        fn();
      }, delay)
    );
  };

  /* First segment: one RAF to ensure coordinates from updateAttentionMultiHeadOverlay are committed */
  const busRaf = requestAnimationFrame(() => {
    if (!slide.classList.contains('attn24-show-split') || slide.classList.contains('attn24-source-collapsed')) return;
    const src = document.getElementById('attn24-line-source');
    runAttentionMultiHeadStageLineDraw(src, drawMs);
  });
  state.attentionMultiHead.rafIds.push(busRaf);

  push(() => {
    const node = document.getElementById('attn24-copy-node');
    const label = document.getElementById('attn24-copy-label');
    if (node) {
      node.style.transition = 'opacity ' + ATTN_QKV_COMPARE_HEAD_FADE_MS + 'ms ease';
      node.style.opacity = '1';
    }
    if (label) {
      label.style.transition = 'opacity ' + ATTN_QKV_COMPARE_HEAD_FADE_MS + 'ms ease';
      label.style.opacity = '1';
    }
  }, Math.max(drawMs - 70, 120));

  const trunkStart = drawMs + gap;
  push(() => {
    runAttentionMultiHeadStageLineDraw(document.getElementById('attn24-line-bus-trunk'), drawMs);
  }, trunkStart);

  const mainStart = trunkStart + drawMs + gap;
  push(() => {
    runAttentionMultiHeadStageLineDraw(document.getElementById('attn24-line-bus-main'), drawMs);
  }, mainStart);

  const branchStart = mainStart + drawMs + gap;
  push(() => {
    runAttentionMultiHeadStageLineDraw(document.getElementById('attn24-line-h1'), drawMs);
    runAttentionMultiHeadStageLineDraw(document.getElementById('attn24-line-h2'), drawMs);
  }, branchStart);

  const branchEnd = branchStart + drawMs;
  push(() => {
    const h1 = document.getElementById('attn24-line-h1');
    const h2 = document.getElementById('attn24-line-h2');
    if (h1) h1.setAttribute('marker-end', 'url(#attn24-arr-marker)');
    if (h2) h2.setAttribute('marker-end', 'url(#attn24-arr-marker)');
  }, branchEnd + 16);

  return branchEnd + 100;
}

function resetAttentionMultiHeadVisuals() {
  const slide = document.getElementById('slide-24');
  clearAttentionMultiHeadTimers();
  clearAttentionMultiHeadGhosts();
  state.attentionMultiHead.splitDone = false;
  state.attentionMultiHead.sourceCollapsedDone = false;
  state.attentionMultiHead.projDone = false;
  state.attentionMultiHead.attnDone = false;
  state.attentionMultiHead.outputDone = false;
  state.attentionMultiHead.concatDone = false;
  state.attentionMultiHead.outputProjectionDone = false;
  state.attentionMultiHead.combineVisible = false;
  state.attentionMultiHead.outputVisibleCount = 0;
  if (!slide) return;
  slide.classList.remove(
    'attn24-show-split',
    'attn24-source-collapsed',
    'attn24-show-proj',
    'attn24-show-attn',
    'attn24-show-output',
    'attn24-show-concat-flight',
    'attn24-show-concat',
    'attn24-show-concat-ready',
    'attn24-show-wo'
  );
  clearAttentionMultiHeadStageBusDrawStyles();
  slide.querySelectorAll('.attn24-head-card').forEach((card) => {
    card.classList.remove(
      'is-split-visible',
      'is-proj-visible',
      'is-proj-dummies',
      'is-proj-weights',
      'is-qkv-visible',
      'is-attn-visible',
      'is-output-visible'
    );
  });
  syncAttentionMultiHeadOutputRows(0);
  clearAttentionMultiHeadActiveRows();
  ATTN_MHA_HEADS.forEach((head) => {
    setAttentionMultiHeadHeadSourceVisible(head, false);
    setAttentionMultiHeadDummyCopiesVisible(head, false);
    setAttentionMultiHeadHeadOverlayVisible(head, false);
    setAttentionMultiHeadSourceOutputVisible(head, true);
  });
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
    setAttentionMultiHeadHeadSourceVisible(head, true);
    setAttentionMultiHeadDummyCopiesVisible(head, false);
    setAttentionMultiHeadHeadOverlayVisible(head, false);
  });
  state.attentionMultiHead.splitDone = true;
  updateAttentionMultiHeadOverlay();
}

function settleAttentionMultiHeadSourceCollapseState() {
  const slide = document.getElementById('slide-24');
  settleAttentionMultiHeadSplitState();
  if (!slide) return;
  slide.classList.add('attn24-source-collapsed');
  state.attentionMultiHead.splitDone = true;
  state.attentionMultiHead.sourceCollapsedDone = true;
  ATTN_MHA_HEADS.forEach((head) => {
    setAttentionMultiHeadHeadSourceVisible(head, true);
    setAttentionMultiHeadDummyCopiesVisible(head, false);
    setAttentionMultiHeadHeadOverlayVisible(head, false);
  });
  updateAttentionMultiHeadOverlay();
}

function settleAttentionMultiHeadProjectionState() {
  const slide = document.getElementById('slide-24');
  settleAttentionMultiHeadSourceCollapseState();
  if (!slide) return;
  slide.classList.add('attn24-show-proj');
  ATTN_MHA_HEADS.forEach((head) => {
    const card = document.getElementById('attn24-head-card-' + head);
    if (!card) return;
    card.classList.add('is-proj-visible', 'is-proj-dummies', 'is-proj-weights', 'is-qkv-visible');
    setAttentionMultiHeadHeadSourceVisible(head, true);
    setAttentionMultiHeadDummyCopiesVisible(head, true);
    setAttentionMultiHeadHeadOverlayVisible(head, true);
  });
  state.attentionMultiHead.splitDone = true;
  state.attentionMultiHead.sourceCollapsedDone = true;
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
    setAttentionMultiHeadHeadSourceVisible(head, true);
    setAttentionMultiHeadDummyCopiesVisible(head, true);
    setAttentionMultiHeadHeadOverlayVisible(head, true);
  });
  state.attentionMultiHead.splitDone = true;
  state.attentionMultiHead.sourceCollapsedDone = true;
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
    setAttentionMultiHeadHeadSourceVisible(head, true);
    setAttentionMultiHeadDummyCopiesVisible(head, true);
    setAttentionMultiHeadHeadOverlayVisible(head, true);
  });
  state.attentionMultiHead.splitDone = true;
  state.attentionMultiHead.sourceCollapsedDone = true;
  state.attentionMultiHead.projDone = true;
  state.attentionMultiHead.attnDone = true;
  state.attentionMultiHead.outputDone = true;
  state.attentionMultiHead.outputVisibleCount = ATTN_MHA_TOKENS.length;
  syncAttentionMultiHeadOutputRows();
  clearAttentionMultiHeadActiveRows();
  ATTN_MHA_HEADS.forEach((head) => setAttentionMultiHeadSourceOutputVisible(head, true));
  ATTN_MHA_HEADS.forEach((head) => {
    setAttentionMultiHeadHeadSourceVisible(head, true);
    setAttentionMultiHeadDummyCopiesVisible(head, true);
    setAttentionMultiHeadHeadOverlayVisible(head, true);
  });
  updateAttentionMultiHeadOverlay();
}

function settleAttentionMultiHeadConcatState() {
  const slide = document.getElementById('slide-24');
  settleAttentionMultiHeadOutputState();
  if (!slide) return;
  slide.classList.remove('attn24-show-concat-flight');
  slide.classList.add('attn24-show-concat', 'attn24-show-concat-ready');
  slide.classList.remove('attn24-show-wo');
  state.attentionMultiHead.splitDone = true;
  state.attentionMultiHead.sourceCollapsedDone = true;
  state.attentionMultiHead.projDone = true;
  state.attentionMultiHead.attnDone = true;
  state.attentionMultiHead.outputDone = true;
  state.attentionMultiHead.concatDone = true;
  state.attentionMultiHead.combineVisible = true;
  state.attentionMultiHead.outputProjectionDone = false;
  ATTN_MHA_HEADS.forEach((head) => setAttentionMultiHeadSourceOutputVisible(head, true));
  ATTN_MHA_HEADS.forEach((head) => {
    setAttentionMultiHeadHeadSourceVisible(head, true);
    setAttentionMultiHeadDummyCopiesVisible(head, true);
    setAttentionMultiHeadHeadOverlayVisible(head, true);
  });
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
  state.attentionMultiHead.sourceCollapsedDone = true;
  ATTN_MHA_HEADS.forEach((head) => setAttentionMultiHeadSourceOutputVisible(head, true));
  ATTN_MHA_HEADS.forEach((head) => {
    setAttentionMultiHeadHeadSourceVisible(head, true);
    setAttentionMultiHeadDummyCopiesVisible(head, true);
    setAttentionMultiHeadHeadOverlayVisible(head, true);
  });
  updateAttentionMultiHeadOverlay();
}

function runAttentionMultiHeadSplitSequence() {
  const slide = document.getElementById('slide-24');
  if (!slide) return;
  resetAttentionMultiHeadVisuals();
  const rafId = requestAnimationFrame(() => {
    slide.classList.add('attn24-show-split');
    updateAttentionMultiHeadOverlay();
    prepAttentionMultiHeadStageBusStrokeHidden();
    const busBuildMs = scheduleAttentionMultiHeadStageBusDrawChain(slide);

    ATTN_MHA_HEADS.forEach((head, idx) => {
      const card = document.getElementById('attn24-head-card-' + head);
      const source = document.getElementById('attn24-source-shell');
      const target = document.getElementById('attn24-head-mini-x-shell-' + head);
      const ghost = createAttentionMultiHeadSplitGhost();
      if (ghost && source && target) {
        animateAttentionMultiHeadGhost(ghost, source, target, ATTN_MHA_SPLIT_MS);
        state.attentionMultiHead.timers.push(setTimeout(() => {
          if (card) card.classList.add('is-split-visible');
          setAttentionMultiHeadHeadSourceVisible(head, true);
          setAttentionMultiHeadDummyCopiesVisible(head, false);
          setAttentionMultiHeadHeadOverlayVisible(head, false);
        }, Math.max(ATTN_MHA_SPLIT_MS - 110, 140) + (idx * 30)));
        state.attentionMultiHead.timers.push(setTimeout(() => {
          ghost.remove();
        }, ATTN_MHA_SPLIT_MS + ATTN_MATRIX_FADE_MS + 40));
      } else if (card) {
        card.classList.add('is-split-visible');
        setAttentionMultiHeadHeadSourceVisible(head, true);
        setAttentionMultiHeadDummyCopiesVisible(head, false);
        setAttentionMultiHeadHeadOverlayVisible(head, false);
      }
    });

    const ghostDoneMs = ATTN_MHA_SPLIT_MS + ATTN_MATRIX_FADE_MS + 80;
    state.attentionMultiHead.timers.push(setTimeout(() => {
      settleAttentionMultiHeadSplitState();
    }, Math.max(ghostDoneMs, busBuildMs)));
  });
  state.attentionMultiHead.rafIds.push(rafId);
}

function runAttentionMultiHeadSourceCollapseSequence() {
  const slide = document.getElementById('slide-24');
  if (!slide) return;
  if (!state.attentionMultiHead.splitDone) {
    settleAttentionMultiHeadSplitState();
  }
  clearAttentionMultiHeadTimers();
  clearAttentionMultiHeadGhosts();
  slide.classList.add('attn24-show-split');

  state.attentionMultiHead.timers.push(setTimeout(() => {
    slide.classList.add('attn24-source-collapsed');
  }, 30));

  state.attentionMultiHead.timers.push(setTimeout(() => {
    settleAttentionMultiHeadSourceCollapseState();
  }, ATTN_MATRIX_FADE_MS + 120));
}

function runAttentionMultiHeadProjectionSequence() {
  const slide = document.getElementById('slide-24');
  if (!slide) return;
  if (!state.attentionMultiHead.sourceCollapsedDone) {
    settleAttentionMultiHeadSourceCollapseState();
  }
  clearAttentionMultiHeadTimers();
  clearAttentionMultiHeadGhosts();
  slide.classList.add('attn24-show-split', 'attn24-source-collapsed');
  slide.classList.add('attn24-show-proj');

  ATTN_MHA_HEADS.forEach((head) => {
    const card = document.getElementById('attn24-head-card-' + head);
    if (!card) {
      return;
    }
    card.classList.remove('is-proj-visible', 'is-proj-dummies', 'is-proj-weights', 'is-qkv-visible');
    setAttentionMultiHeadDummyCopiesVisible(head, false);
    setAttentionMultiHeadHeadOverlayVisible(head, false);
  });

  ATTN_MHA_HEADS.forEach((head, idx) => {
    const card = document.getElementById('attn24-head-card-' + head);
    if (!card) return;
    const t0 = idx * ATTN_MHA_PROJ_STAGGER_MS;
    state.attentionMultiHead.timers.push(setTimeout(() => {
      card.classList.add('is-split-visible', 'is-proj-visible', 'is-proj-dummies');
      setAttentionMultiHeadHeadSourceVisible(head, true);
      setAttentionMultiHeadDummyCopiesVisible(head, true);
      setAttentionMultiHeadHeadOverlayVisible(head, true);
      updateAttentionMultiHeadHeadOverlay(head);
    }, t0));
    state.attentionMultiHead.timers.push(setTimeout(() => {
      card.classList.add('is-proj-weights');
      updateAttentionMultiHeadHeadOverlay(head);
    }, t0 + ATTN_MHA_PROJ_DUMMY_MS));
    state.attentionMultiHead.timers.push(setTimeout(() => {
      card.classList.add('is-qkv-visible');
      updateAttentionMultiHeadHeadOverlay(head);
    }, t0 + ATTN_MHA_PROJ_DUMMY_MS + ATTN_MHA_PROJ_WEIGHTS_MS));
  });

  const lastStagger = (ATTN_MHA_HEADS.length - 1) * ATTN_MHA_PROJ_STAGGER_MS;
  const projAnimEnd = lastStagger + ATTN_MHA_PROJ_DUMMY_MS + ATTN_MHA_PROJ_WEIGHTS_MS + ATTN_MHA_PROJ_REVEAL_MS;
  state.attentionMultiHead.timers.push(setTimeout(() => {
    settleAttentionMultiHeadProjectionState();
  }, projAnimEnd + ATTN_MATRIX_FADE_MS + 50));
}

function runAttentionMultiHeadAttentionSequence() {
  const slide = document.getElementById('slide-24');
  if (!slide) return;
  if (!state.attentionMultiHead.projDone) {
    settleAttentionMultiHeadProjectionState();
  }
  clearAttentionMultiHeadTimers();
  clearAttentionMultiHeadGhosts();
  slide.classList.add('attn24-show-split', 'attn24-source-collapsed', 'attn24-show-proj', 'attn24-show-attn');

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
  slide.classList.add('attn24-show-split', 'attn24-source-collapsed', 'attn24-show-proj', 'attn24-show-attn', 'attn24-show-output');
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
  slide.classList.add('attn24-show-concat-flight');
  slide.classList.remove('attn24-show-concat', 'attn24-show-concat-ready', 'attn24-show-wo');

  const ghosts = [];
  let landed = 0;

  function onGhostLanded() {
    landed++;
    if (landed < ghosts.length) return;

    /* All ghosts have finished their CSS transform flight. */

    /* 1 — Fade ghosts out */
    ghosts.forEach((g) => { g.style.opacity = '0'; });

    /* 2 — After the opacity fade, swap to concat layout */
    state.attentionMultiHead.timers.push(setTimeout(() => {
      slide.classList.add('attn24-show-concat');
      slide.classList.remove('attn24-show-concat-flight');
      ghosts.forEach((g) => g.remove());
    }, ATTN_MATRIX_FADE_MS + 40));

    /* 3 — Reveal the concat matrix */
    state.attentionMultiHead.timers.push(setTimeout(() => {
      slide.classList.add('attn24-show-concat-ready');
    }, ATTN_MATRIX_FADE_MS + 160));

    /* 4 — Settle into final state */
    state.attentionMultiHead.timers.push(setTimeout(() => {
      settleAttentionMultiHeadConcatState();
    }, ATTN_MATRIX_FADE_MS + 320));
  }

  ATTN_MHA_HEADS.forEach((head) => {
    const sourceShell = document.getElementById('attn24-head-output-shell-' + head);
    const target = document.getElementById(head === 'h1' ? 'attn24-concat-target-left' : 'attn24-concat-target-right');
    const ghost = createAttentionMultiHeadOutputGhost(head);
    setAttentionMultiHeadSourceOutputVisible(head, false);
    if (ghost && sourceShell && target) {
      const didAnimate = animateAttentionMultiHeadGhost(ghost, sourceShell, target, ATTN_MHA_CONCAT_MS, {
        onDone: onGhostLanded
      });
      if (didAnimate) {
        ghosts.push(ghost);
      } else {
        ghost.remove();
      }
    }
  });

  if (ghosts.length === 0) {
    slide.classList.add('attn24-show-concat', 'attn24-show-concat-ready');
    slide.classList.remove('attn24-show-concat-flight');
    state.attentionMultiHead.timers.push(setTimeout(() => {
      settleAttentionMultiHeadConcatState();
    }, 60));
  }
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
  slide.classList.remove('attn24-show-concat-flight', 'attn24-show-wo');

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
        const s = state.attentionMultiHead;
        const doneForStep = [
          true, s.splitDone, s.sourceCollapsedDone, s.projDone,
          s.attnDone, s.outputDone, s.concatDone, s.outputProjectionDone
        ];
        if (!doneForStep[s.step]) return;
        if (s.step >= 7)      settleAttentionMultiHeadOutputProjectionState();
        else if (s.step >= 6) settleAttentionMultiHeadConcatState();
        else if (s.step >= 5) settleAttentionMultiHeadOutputState();
        else if (s.step >= 4) settleAttentionMultiHeadAttentionState();
        else if (s.step >= 3) settleAttentionMultiHeadProjectionState();
        else if (s.step >= 2) settleAttentionMultiHeadSourceCollapseState();
        else if (s.step >= 1) settleAttentionMultiHeadSplitState();
        else                  resetAttentionMultiHeadVisuals();
      });
      state.attentionMultiHead.resizeBound = true;
    }

    state.attentionMultiHead.initialized = true;
  }

  const stageCopyLabel = document.getElementById('attn24-copy-label');
  if (stageCopyLabel) {
    stageCopyLabel.textContent = '\u00d7' + String(ATTN_MHA_HEADS.length);
  }

  const takeaway = document.getElementById('attn24-takeaway');
  if (takeaway) setMathHTML(takeaway, ATTN_MHA_TAKEAWAYS[state.attentionMultiHead.step] || ATTN_MHA_TAKEAWAYS[0]);
  syncAttentionMultiHeadCaptions();
  typesetMath(slide);

  if (state.attentionMultiHead.step >= 7) {
    settleAttentionMultiHeadOutputProjectionState();
  } else if (state.attentionMultiHead.step >= 6) {
    settleAttentionMultiHeadConcatState();
  } else if (state.attentionMultiHead.step >= 5) {
    settleAttentionMultiHeadOutputState();
  } else if (state.attentionMultiHead.step >= 4) {
    settleAttentionMultiHeadAttentionState();
  } else if (state.attentionMultiHead.step >= 3) {
    settleAttentionMultiHeadProjectionState();
  } else if (state.attentionMultiHead.step >= 2) {
    settleAttentionMultiHeadSourceCollapseState();
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
    if (!state.attentionMultiHead.sourceCollapsedDone && animateStep) {
      runAttentionMultiHeadSourceCollapseSequence();
    } else {
      settleAttentionMultiHeadSourceCollapseState();
    }
    return;
  }

  if (!state.attentionMultiHead.sourceCollapsedDone) {
    settleAttentionMultiHeadSourceCollapseState();
  }
  if (clamped === 3) {
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
  if (clamped === 4) {
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
  if (clamped === 5) {
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
  if (clamped === 6) {
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
  if (clamped === 7) {
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
