function createAttentionQkvVectorRect(vectorId, extraClass = '') {
  return createVectorRect({
    id: vectorId,
    baseClass: 'attn19-vector' + (extraClass ? ' ' + extraClass : ''),
    dividerClass: 'attn19-vector-divider'
  });
}

function createAttentionQkvColumn(token) {
  const col = createEl('div', {
    className: 'attn19-col',
    dataset: { token }
  });

  const scoreSlot = createEl('div', { className: 'attn19-score-slot' });
  const scoreBar = createEl('div', {
    className: 'attn19-score-bar',
    id: 'attn19-score-' + token
  });
  scoreBar.style.setProperty('--score', String(ATTN_QKV_SCORE_LEVELS[token] || 0.2));
  scoreSlot.appendChild(scoreBar);
  col.appendChild(scoreSlot);

  const chipWrap = createEl('div', { className: 'attn19-chip-wrap' });
  if (token === ATTN_QKV_FOCUS) {
    const focusLabel = createEl('span', {
      className: 'attn19-focus-label',
      text: 'FOCUS'
    });
    chipWrap.appendChild(focusLabel);
  }
  const chip = createEl('div', {
    className: 'attn19-token-chip',
    id: 'attn19-chip-' + token,
    text: token
  });
  chipWrap.appendChild(chip);
  col.appendChild(chipWrap);

  const xWrap = createEl('div', {
    className: 'attn19-vector-wrap attn19-vector-wrap-x',
    id: 'attn19-vector-wrap-x-' + token
  });
  const xLabel = createEl('span', {
    className: 'attn19-vector-label',
    text: 'x_' + token
  });
  xWrap.appendChild(xLabel);
  xWrap.appendChild(createAttentionQkvVectorRect('attn19-vector-x-' + token));
  col.appendChild(xWrap);

  const wkTick = createEl('span', {
    className: 'attn19-proj-tick wk',
    id: 'attn19-proj-wk-' + token
  });
  col.appendChild(wkTick);

  const kWrap = createEl('div', {
    className: 'attn19-vector-wrap attn19-vector-wrap-k',
    id: 'attn19-vector-wrap-k-' + token
  });
  const kLabel = createEl('span', {
    className: 'attn19-vector-label',
    text: 'k_' + token
  });
  kWrap.appendChild(kLabel);
  kWrap.appendChild(createAttentionQkvVectorRect('attn19-vector-k-' + token, 'attn19-vector-k'));
  col.appendChild(kWrap);

  const wvTick = createEl('span', {
    className: 'attn19-proj-tick wv',
    id: 'attn19-proj-wv-' + token
  });
  col.appendChild(wvTick);

  const vWrap = createEl('div', {
    className: 'attn19-vector-wrap attn19-vector-wrap-v',
    id: 'attn19-vector-wrap-v-' + token
  });
  const vLabel = createEl('span', {
    className: 'attn19-vector-label',
    text: 'v_' + token
  });
  vWrap.appendChild(vLabel);
  vWrap.appendChild(createAttentionQkvVectorRect('attn19-vector-v-' + token));
  col.appendChild(vWrap);

  return col;
}

function clearAttentionQkvCompareTimers() {
  attentionQkvState.compareTimers.forEach((timerId) => clearTimeout(timerId));
  attentionQkvState.compareTimers = [];
}

function hideAttentionQkvCompareArrow(token) {
  const path = document.getElementById('attn19-compare-path-' + token);
  const head = document.getElementById('attn19-compare-head-' + token);
  const dot = document.getElementById('attn19-compare-dot-' + token);
  hideArrowElements({ path, head, dot });
}

function showAttentionQkvCompareArrow(token) {
  const path = document.getElementById('attn19-compare-path-' + token);
  const head = document.getElementById('attn19-compare-head-' + token);
  const dot = document.getElementById('attn19-compare-dot-' + token);
  showArrowElements({ path, head, dot });
}

function syncAttentionQkvCompareVisuals(visibleCount = attentionQkvState.compareVisibleCount) {
  const clamped = Math.max(0, Math.min(ATTN_QKV_COMPARE_TOKENS.length, visibleCount));
  ATTN_QKV_COMPARE_TOKENS.forEach((token, idx) => {
    if (idx < clamped) showAttentionQkvCompareArrow(token);
    else hideAttentionQkvCompareArrow(token);
  });
}

function animateAttentionQkvCompareArrow(token) {
  const path = document.getElementById('attn19-compare-path-' + token);
  const head = document.getElementById('attn19-compare-head-' + token);
  const dot = document.getElementById('attn19-compare-dot-' + token);
  if (!path || !head || !dot) return;

  hideAttentionQkvCompareArrow(token);
  void path.getBoundingClientRect();
  path.style.transition = 'stroke-dashoffset ' + ATTN_QKV_COMPARE_DRAW_MS + 'ms cubic-bezier(0.34, 0.08, 0.22, 1)';
  path.style.strokeDashoffset = '0';

  const headTimer = setTimeout(() => {
    head.style.transition = 'opacity ' + ATTN_QKV_COMPARE_HEAD_FADE_MS + 'ms ease';
    head.style.opacity = '1';
    dot.style.transition = 'opacity ' + ATTN_QKV_COMPARE_HEAD_FADE_MS + 'ms ease';
    dot.style.opacity = '1';
  }, Math.max(ATTN_QKV_COMPARE_DRAW_MS - 70, 120));
  attentionQkvState.compareTimers.push(headTimer);
}

function runAttentionQkvCompareSequence() {
  const slide = document.getElementById('slide-19');
  if (!slide) return;
  clearAttentionQkvCompareTimers();
  attentionQkvState.compareDone = false;
  attentionQkvState.compareVisibleCount = 0;
  slide.classList.remove('attn19-compare-done');
  syncAttentionQkvCompareVisuals(0);

  ATTN_QKV_COMPARE_TOKENS.forEach((token, idx) => {
    const timer = setTimeout(() => {
      if (attentionQkvState.step < 4) return;
      attentionQkvState.compareVisibleCount = idx + 1;
      animateAttentionQkvCompareArrow(token);

      if (idx === ATTN_QKV_COMPARE_TOKENS.length - 1) {
        const doneTimer = setTimeout(() => {
          if (attentionQkvState.step < 4) return;
          attentionQkvState.compareDone = true;
          attentionQkvState.compareVisibleCount = ATTN_QKV_COMPARE_TOKENS.length;
          slide.classList.add('attn19-compare-done');
          syncAttentionQkvCompareVisuals(attentionQkvState.compareVisibleCount);
        }, ATTN_QKV_COMPARE_DRAW_MS + ATTN_QKV_SCORE_REVEAL_DELAY_MS);
        attentionQkvState.compareTimers.push(doneTimer);
      }
    }, idx * ATTN_QKV_COMPARE_STAGGER_MS);
    attentionQkvState.compareTimers.push(timer);
  });
}

function updateAttentionQkvOverlay() {
  const stage = document.getElementById('attn19-stage');
  const overlay = document.getElementById('attn19-overlay');
  const qCallout = document.getElementById('attn19-q-callout');
  const qVector = document.getElementById('attn19-q-sat');
  const satChip = document.getElementById('attn19-chip-' + ATTN_QKV_FOCUS);
  const satXVector = document.getElementById('attn19-vector-x-' + ATTN_QKV_FOCUS);
  const satKVector = document.getElementById('attn19-vector-k-' + ATTN_QKV_FOCUS);
  if (!stage || !overlay || !qCallout || !qVector || !satChip || !satXVector || !satKVector) return;

  const stageRect = stage.getBoundingClientRect();
  if (stageRect.width < 1 || stageRect.height < 1) return;
  const stageH = stageRect.height;
  overlay.setAttribute('viewBox', '0 0 ' + stageRect.width + ' ' + stageRect.height);

  const anchor = (el, x, y) => {
    const rect = el.getBoundingClientRect();
    return {
      x: rect.left - stageRect.left + rect.width * x,
      y: rect.top - stageRect.top + rect.height * y
    };
  };

  const satChipRect = satChip.getBoundingClientRect();
  const satCenterX = satChipRect.left - stageRect.left + satChipRect.width * 0.5;
  const xSatBottom = anchor(satXVector, 0.5, 1);
  const kSatTop = anchor(satKVector, 0.5, 0);
  const qCalloutRect = qCallout.getBoundingClientRect();
  const gapFromX = Math.max(stageH * 0.03, 10);
  const gapToK = Math.max(stageH * 0.022, 8);
  const qTopDesired = xSatBottom.y + gapFromX;
  const qTopMax = kSatTop.y - qCalloutRect.height - gapToK;
  const qTop = Math.max(stageH * 0.02, Math.min(qTopDesired, qTopMax));
  qCallout.style.left = satCenterX.toFixed(2) + 'px';
  qCallout.style.top = qTop.toFixed(2) + 'px';

  const qVectorTop = anchor(qVector, 0.5, 0);
  const qBottom = anchor(qVector, 0.5, 1);
  const qLink = document.getElementById('attn19-q-link');
  if (qLink) {
    const pad = Math.max(stageH * 0.007, 3.4);
    qLink.setAttribute('x1', xSatBottom.x.toFixed(2));
    qLink.setAttribute('y1', (xSatBottom.y + pad).toFixed(2));
    qLink.setAttribute('x2', qVectorTop.x.toFixed(2));
    qLink.setAttribute('y2', (qVectorTop.y - pad).toFixed(2));
  }

  const labelWk = document.getElementById('attn19-label-wk');
  const labelWv = document.getElementById('attn19-label-wv');
  const wkTick = document.getElementById('attn19-proj-wk-cat');
  const wvTick = document.getElementById('attn19-proj-wv-cat');
  if (labelWk && wkTick) {
    const tickRect = wkTick.getBoundingClientRect();
    const x = tickRect.left - stageRect.left - Math.max(stageRect.width * 0.028, 14);
    const y = tickRect.top - stageRect.top + tickRect.height * 0.5;
    labelWk.style.left = x.toFixed(2) + 'px';
    labelWk.style.top = y.toFixed(2) + 'px';
  }
  if (labelWv && wvTick) {
    const tickRect = wvTick.getBoundingClientRect();
    const x = tickRect.left - stageRect.left - Math.max(stageRect.width * 0.028, 14);
    const y = tickRect.top - stageRect.top + tickRect.height * 0.5;
    labelWv.style.left = x.toFixed(2) + 'px';
    labelWv.style.top = y.toFixed(2) + 'px';
  }

  let compareMinX = qBottom.x;
  let compareMaxX = qBottom.x;
  let compareMinControlY = qBottom.y;
  const compareSource = {
    x: qBottom.x,
    y: qBottom.y
  };

  ATTN_QKV_COMPARE_TOKENS.forEach((token) => {
    const kVector = document.getElementById('attn19-vector-k-' + token);
    const path = document.getElementById('attn19-compare-path-' + token);
    const head = document.getElementById('attn19-compare-head-' + token);
    const dot = document.getElementById('attn19-compare-dot-' + token);
    if (!kVector || !path || !head || !dot) return;

    const kTop = anchor(kVector, 0.5, 0);
    const target = {
      x: kTop.x,
      y: kTop.y + Math.max(stageH * 0.003, 1)
    };
    const distanceX = Math.abs(compareSource.x - target.x);
    const distanceY = target.y - compareSource.y;
    const curveFactor = ATTN_QKV_COMPARE_CURVE_FACTORS[token] || 0.5;
    let controlY = compareSource.y + distanceY * curveFactor + distanceX * 0.01;
    const controlYMax = target.y - Math.max(stageH * 0.006, 2.2);
    if (controlY > controlYMax) controlY = controlYMax;
    const cp1x = compareSource.x + (target.x - compareSource.x) * 0.3;
    const cp2x = compareSource.x + (target.x - compareSource.x) * 0.72;

    compareMinX = Math.min(compareMinX, target.x);
    compareMaxX = Math.max(compareMaxX, target.x);
    compareMinControlY = Math.min(compareMinControlY, controlY);

    const d = 'M ' + compareSource.x.toFixed(2) + ' ' + compareSource.y.toFixed(2)
      + ' C ' + cp1x.toFixed(2) + ' ' + controlY.toFixed(2)
      + ' ' + cp2x.toFixed(2) + ' ' + controlY.toFixed(2)
      + ' ' + target.x.toFixed(2) + ' ' + target.y.toFixed(2);
    path.setAttribute('d', d);

    const dirX = target.x - cp2x;
    const dirY = target.y - controlY;
    const dirLen = Math.hypot(dirX, dirY) || 1;
    const unitX = dirX / dirLen;
    const unitY = dirY / dirLen;
    const headLen = Math.max(stageH * 0.022, 5.1);
    const wing = headLen * 0.54;
    const baseX = target.x - unitX * headLen;
    const baseY = target.y - unitY * headLen;
    const wingX = -unitY * wing;
    const wingY = unitX * wing;
    const points = target.x.toFixed(2) + ',' + target.y.toFixed(2)
      + ' ' + (baseX + wingX).toFixed(2) + ',' + (baseY + wingY).toFixed(2)
      + ' ' + (baseX - wingX).toFixed(2) + ',' + (baseY - wingY).toFixed(2);
    head.setAttribute('points', points);

    dot.setAttribute('cx', target.x.toFixed(2));
    dot.setAttribute('cy', target.y.toFixed(2));
    dot.setAttribute('r', Math.max(stageH * 0.0052, 1.7).toFixed(2));
  });

  const scoreCaption = document.getElementById('attn19-score-caption');
  const scoreFootnote = document.getElementById('attn19-score-footnote');
  if (scoreCaption) {
    const captionX = (compareMinX + compareMaxX) * 0.5;
    const captionFloor = qTop - Math.max(stageH * 0.01, 3);
    const captionY = Math.max(captionFloor, compareMinControlY - Math.max(stageH * 0.018, 6));
    scoreCaption.setAttribute('x', captionX.toFixed(2));
    scoreCaption.setAttribute('y', captionY.toFixed(2));
    scoreCaption.setAttribute('text-anchor', 'middle');
    if (scoreFootnote) {
      scoreFootnote.setAttribute('x', captionX.toFixed(2));
      scoreFootnote.setAttribute('y', (captionY + Math.max(stageH * 0.028, 8)).toFixed(2));
      scoreFootnote.setAttribute('text-anchor', 'middle');
    }
  }

  if (attentionQkvState.step < 4) {
    syncAttentionQkvCompareVisuals(0);
  } else if (attentionQkvState.compareDone) {
    syncAttentionQkvCompareVisuals(ATTN_QKV_COMPARE_TOKENS.length);
  }
}

function setAttentionQkvStep(step) {
  const slide = document.getElementById('slide-19');
  const takeaway = document.getElementById('attn19-takeaway');
  if (!slide || !takeaway) return;

  const prevStep = attentionQkvState.step;
  const clamped = Math.max(0, Math.min(ATTN_QKV_MAX_STEP, step));
  attentionQkvState.step = clamped;

  slide.classList.toggle('attn19-show-k', clamped >= 1);
  slide.classList.toggle('attn19-show-v', clamped >= 2);
  slide.classList.toggle('attn19-show-q', clamped >= 3);
  slide.classList.toggle('attn19-show-compare', clamped >= 4);
  takeaway.textContent = ATTN_QKV_TAKEAWAYS[clamped] || ATTN_QKV_TAKEAWAYS[0];

  if (clamped < 4) {
    clearAttentionQkvCompareTimers();
    attentionQkvState.compareDone = false;
    attentionQkvState.compareVisibleCount = 0;
    slide.classList.remove('attn19-compare-done');
  }

  updateAttentionQkvOverlay();

  if (clamped === 4) {
    if (prevStep < 4 || !attentionQkvState.compareDone) {
      runAttentionQkvCompareSequence();
    } else {
      attentionQkvState.compareVisibleCount = ATTN_QKV_COMPARE_TOKENS.length;
      attentionQkvState.compareDone = true;
      slide.classList.add('attn19-compare-done');
      syncAttentionQkvCompareVisuals(attentionQkvState.compareVisibleCount);
    }
  } else {
    syncAttentionQkvCompareVisuals(0);
  }

  requestAnimationFrame(updateAttentionQkvOverlay);
  if (attentionQkvState.overlayTimer) clearTimeout(attentionQkvState.overlayTimer);
  attentionQkvState.overlayTimer = setTimeout(() => {
    updateAttentionQkvOverlay();
    attentionQkvState.overlayTimer = null;
  }, 240);
}

function initAttentionQkvSlide() {
  const slide = document.getElementById('slide-19');
  const cols = document.getElementById('attn19-cols');
  if (!slide || !cols) return;

  if (!attentionQkvState.initialized) {
    cols.innerHTML = '';
    ATTN_QKV_TOKENS.forEach((token) => {
      cols.appendChild(createAttentionQkvColumn(token));
    });

    if (!attentionQkvState.resizeBound) {
      addTrackedListener(window, 'resize', () => {
        if (!attentionQkvState.initialized) return;
        clearAttentionQkvCompareTimers();
        if (attentionQkvState.step >= 4) {
          attentionQkvState.compareDone = true;
          attentionQkvState.compareVisibleCount = ATTN_QKV_COMPARE_TOKENS.length;
          const activeSlide = document.getElementById('slide-19');
          if (activeSlide) activeSlide.classList.add('attn19-compare-done');
        } else {
          attentionQkvState.compareDone = false;
          attentionQkvState.compareVisibleCount = 0;
        }
        updateAttentionQkvOverlay();
        if (attentionQkvState.step >= 4 && attentionQkvState.compareDone) {
          syncAttentionQkvCompareVisuals(attentionQkvState.compareVisibleCount);
        } else {
          syncAttentionQkvCompareVisuals(0);
        }
      });
      attentionQkvState.resizeBound = true;
    }

    attentionQkvState.initialized = true;
  }

  const takeaway = document.getElementById('attn19-takeaway');
  if (takeaway) takeaway.textContent = ATTN_QKV_TAKEAWAYS[attentionQkvState.step] || ATTN_QKV_TAKEAWAYS[0];
  updateAttentionQkvOverlay();
  if (attentionQkvState.step >= 4 && attentionQkvState.compareDone) {
    syncAttentionQkvCompareVisuals(attentionQkvState.compareVisibleCount);
  } else {
    syncAttentionQkvCompareVisuals(0);
  }
}

function runAttentionQkvStep(slideEl) {
  if (!slideEl || slideEl.id !== 'slide-19') return false;
  if (!attentionQkvState.initialized) initAttentionQkvSlide();
  if (attentionQkvState.step >= ATTN_QKV_MAX_STEP) return false;
  setAttentionQkvStep(attentionQkvState.step + 1);
  return true;
}

function resetAttentionQkvSlide() {
  const slide = document.getElementById('slide-19');
  if (!slide) return;
  if (attentionQkvState.overlayTimer) {
    clearTimeout(attentionQkvState.overlayTimer);
    attentionQkvState.overlayTimer = null;
  }
  clearAttentionQkvCompareTimers();
  attentionQkvState.compareDone = false;
  attentionQkvState.compareVisibleCount = 0;
  slide.classList.remove('attn19-compare-done');
  setAttentionQkvStep(0);
}

