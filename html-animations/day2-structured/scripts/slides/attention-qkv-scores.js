function createAttentionQkvColumn(token) {
  const col = createEl('div', {
    className: 'attn19-col',
    dataset: { token }
  });

  const chipWrap = createEl('div', { className: 'attn19-chip-wrap' });
  chipWrap.appendChild(createEl('div', {
    className: 'attn19-token-chip',
    id: 'attn19-chip-' + token,
    text: token
  }));
  col.appendChild(chipWrap);

  const xWrap = createEl('div', {
    className: 'attn19-vector-wrap attn19-vector-wrap-x',
    id: 'attn19-vector-wrap-x-' + token
  }, [
    createMathSubLabel('x', token, 'attn19-vector-label'),
    createAttentionQkvVectorRect('attn19-vector-x-' + token, '', ATTN_QKV_X_VECTORS[token] || ATTN_QKV_QUERY_VECTOR)
  ]);
  col.appendChild(xWrap);

  const scoreOpWrap = createEl('div', {
    className: 'attn19-score-op-wrap',
    id: 'attn19-score-op-wrap-' + token
  });
  const dotNode = createEl('div', {
    className: 'attn19-dot-node',
    id: 'attn19-dot-node-' + token
  }, createMathQTransposeK());
  const scoreQual = ATTN_QKV_SCORE_QUAL[token] || { tier: 'medium' };
  const scorePill = createEl('div', {
    className: 'attn19-score-pill',
    id: 'attn19-score-pill-' + token,
    dataset: { qual: scoreQual.tier }
  }, formatScoreValue(ATTN_WGT_RAW_SCORE_BY_TOKEN[token]));
  scoreOpWrap.appendChild(dotNode);
  scoreOpWrap.appendChild(scorePill);
  col.appendChild(scoreOpWrap);

  const kWrap = createEl('div', {
    className: 'attn19-vector-wrap attn19-vector-wrap-k',
    id: 'attn19-vector-wrap-k-' + token
  }, [
    createMathSubLabel('k', token, 'attn19-vector-label'),
    createAttentionQkvVectorRect('attn19-vector-k-' + token, 'attn19-vector-k', ATTN_QKV_KEY_VECTORS[token] || ATTN_QKV_QUERY_VECTOR)
  ]);
  col.appendChild(kWrap);

  const vWrap = createEl('div', {
    className: 'attn19-vector-wrap attn19-vector-wrap-v',
    id: 'attn19-vector-wrap-v-' + token
  }, [
    createMathSubLabel('v', token, 'attn19-vector-label'),
    createAttentionQkvVectorRect('attn19-vector-v-' + token, '', ATTN_QKV_VALUE_VECTORS[token] || ATTN_QKV_QUERY_VECTOR)
  ]);
  col.appendChild(vWrap);

  return col;
}

function clearAttentionQkvCompareTimers() {
  state.attentionQkv.compareTimers.forEach((timerId) => clearTimeout(timerId));
  state.attentionQkv.compareTimers = [];
}

function getAttentionQkvCompareTriplet(token) {
  return [
    {
      path: document.getElementById('attn19-op-q-path-' + token),
      head: document.getElementById('attn19-op-q-head-' + token)
    },
    {
      path: document.getElementById('attn19-op-k-path-' + token),
      head: document.getElementById('attn19-op-k-head-' + token)
    },
    {
      path: document.getElementById('attn19-op-out-path-' + token),
      head: document.getElementById('attn19-op-out-head-' + token)
    }
  ];
}

function hideAttentionQkvCompareArrow(token) {
  getAttentionQkvCompareTriplet(token).forEach((elements) => {
    hideArrowElements({ path: elements.path, head: elements.head });
  });
}

function showAttentionQkvCompareArrow(token) {
  getAttentionQkvCompareTriplet(token).forEach((elements) => {
    showArrowElements({ path: elements.path, head: elements.head });
  });
}

function syncAttentionQkvCompareVisuals(visibleCount = state.attentionQkv.compareVisibleCount) {
  const clamped = Math.max(0, Math.min(ATTN_QKV_COMPARE_TOKENS.length, visibleCount));
  ATTN_QKV_COMPARE_TOKENS.forEach((token, idx) => {
    if (idx < clamped) showAttentionQkvCompareArrow(token);
    else hideAttentionQkvCompareArrow(token);
  });
}

function animateAttentionQkvCompareArrow(token) {
  hideAttentionQkvCompareArrow(token);
  const triplet = getAttentionQkvCompareTriplet(token).filter((elements) => elements.path && elements.head);
  if (!triplet.length) return;

  triplet.forEach((elements) => {
    const path = elements.path;
    path.style.transition = 'none';
    void path.getBoundingClientRect();
    path.style.transition = 'stroke-dashoffset ' + ATTN_QKV_COMPARE_DRAW_MS + 'ms cubic-bezier(0.34, 0.08, 0.22, 1)';
    path.style.strokeDashoffset = '0';
  });

  const headTimer = setTimeout(() => {
    triplet.forEach((elements) => {
      const head = elements.head;
      head.style.transition = 'opacity ' + ATTN_QKV_COMPARE_HEAD_FADE_MS + 'ms ease';
      head.style.opacity = '1';
    });
  }, Math.max(ATTN_QKV_COMPARE_DRAW_MS - 70, 120));
  state.attentionQkv.compareTimers.push(headTimer);
}

function runAttentionQkvCompareSequence() {
  const slide = document.getElementById('slide-20');
  if (!slide) return;
  clearAttentionQkvCompareTimers();
  state.attentionQkv.compareDone = false;
  state.attentionQkv.compareVisibleCount = 0;
  slide.classList.remove('attn19-compare-done');
  syncAttentionQkvCompareVisuals(0);

  ATTN_QKV_COMPARE_TOKENS.forEach((token, idx) => {
    const timer = setTimeout(() => {
      if (state.attentionQkv.step < 4) return;
      state.attentionQkv.compareVisibleCount = idx + 1;
      animateAttentionQkvCompareArrow(token);

      if (idx === ATTN_QKV_COMPARE_TOKENS.length - 1) {
        const doneTimer = setTimeout(() => {
          if (state.attentionQkv.step < 4) return;
          state.attentionQkv.compareDone = true;
          state.attentionQkv.compareVisibleCount = ATTN_QKV_COMPARE_TOKENS.length;
          slide.classList.add('attn19-compare-done');
          syncAttentionQkvCompareVisuals(state.attentionQkv.compareVisibleCount);
        }, ATTN_QKV_COMPARE_DRAW_MS + ATTN_QKV_SCORE_REVEAL_DELAY_MS);
        state.attentionQkv.compareTimers.push(doneTimer);
      }
    }, idx * ATTN_QKV_COMPARE_STAGGER_MS);
    state.attentionQkv.compareTimers.push(timer);
  });
}

function updateAttentionQkvOverlay() {
  const stage = document.getElementById('attn19-stage');
  const overlay = document.getElementById('attn19-overlay');
  const qCallout = document.getElementById('attn19-q-callout');
  const qVector = document.getElementById('attn19-q-sat');
  const satDotNode = document.getElementById('attn19-dot-node-' + ATTN_QKV_FOCUS);
  const satScorePill = document.getElementById('attn19-score-pill-' + ATTN_QKV_FOCUS);
  const vSatVector = document.getElementById('attn19-vector-v-' + ATTN_QKV_FOCUS);
  const satChip = document.getElementById('attn19-chip-' + ATTN_QKV_FOCUS);
  const satXVector = document.getElementById('attn19-vector-x-' + ATTN_QKV_FOCUS);
  const satKVector = document.getElementById('attn19-vector-k-' + ATTN_QKV_FOCUS);
  if (!stage || !overlay || !qCallout || !qVector || !satDotNode || !satScorePill || !vSatVector || !satChip || !satXVector || !satKVector) return;

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
  const stageY = (rect, frac = 0.5) => (rect.top - stageRect.top) + rect.height * frac;
  const setArrowHeadPoints = (head, from, to, headLen) => {
    if (!head) return;
    const dirX = to.x - from.x;
    const dirY = to.y - from.y;
    const dirLen = Math.hypot(dirX, dirY) || 1;
    const unitX = dirX / dirLen;
    const unitY = dirY / dirLen;
    const wing = headLen * 0.54;
    const baseX = to.x - unitX * headLen;
    const baseY = to.y - unitY * headLen;
    const wingX = -unitY * wing;
    const wingY = unitX * wing;
    const points = to.x.toFixed(2) + ',' + to.y.toFixed(2)
      + ' ' + (baseX + wingX).toFixed(2) + ',' + (baseY + wingY).toFixed(2)
      + ' ' + (baseX - wingX).toFixed(2) + ',' + (baseY - wingY).toFixed(2);
    head.setAttribute('points', points);
  };

  const rowLabels = document.getElementById('attn19-row-labels');
  const rowXLabel = document.getElementById('attn19-row-x-label');
  const rowQLabel = document.getElementById('attn19-row-q-label');
  const rowSLabel = document.getElementById('attn19-row-s-label');
  const rowKLabel = document.getElementById('attn19-row-k-label');
  const rowVLabel = document.getElementById('attn19-row-v-label');
  const syncRowLabelY = (label, targetRect) => {
    if (!rowLabels || !label || !targetRect) return;
    const rowLabelsRect = rowLabels.getBoundingClientRect();
    const y = stageY(targetRect, 0.5) - (rowLabelsRect.top - stageRect.top);
    label.style.top = y.toFixed(2) + 'px';
  };

  const satChipRect = satChip.getBoundingClientRect();
  const satXRect = satXVector.getBoundingClientRect();
  const satDotRect = satDotNode.getBoundingClientRect();
  const satScoreRect = satScorePill.getBoundingClientRect();
  const satKRect = satKVector.getBoundingClientRect();
  const satVRect = vSatVector.getBoundingClientRect();
  const satCenterX = satChipRect.left - stageRect.left + satChipRect.width * 0.5;
  const xSatBottom = anchor(satXVector, 0.5, 1);
  const dotSatTop = satDotRect.top - stageRect.top;
  const qCalloutRect = qCallout.getBoundingClientRect();
  const gapFromX = Math.max(stageH * 0.03, 10);
  const gapToDotNode = Math.max(stageH * 0.018, 7);
  const qDownNudge = Math.max(stageH * 0.012, 4);
  const qTopDesired = xSatBottom.y + gapFromX + qDownNudge;
  const qTopMax = dotSatTop - qCalloutRect.height - gapToDotNode;
  const qTop = Math.max(stageH * 0.02, Math.min(qTopDesired, qTopMax));
  qCallout.style.left = satCenterX.toFixed(2) + 'px';
  qCallout.style.top = qTop.toFixed(2) + 'px';
  const qRect = qVector.getBoundingClientRect();
  syncRowLabelY(rowXLabel, satXRect);
  syncRowLabelY(rowQLabel, qRect);
  syncRowLabelY(rowSLabel, satDotRect || satScoreRect);
  syncRowLabelY(rowKLabel, satKRect);
  syncRowLabelY(rowVLabel, satVRect);

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

  const compareSource = { x: qBottom.x, y: qBottom.y };
  const headLenMain = Math.max(stageH * 0.02, 4.8);
  const headLenMinor = Math.max(stageH * 0.017, 4.2);

  ATTN_QKV_COMPARE_TOKENS.forEach((token) => {
    const dotNode = document.getElementById('attn19-dot-node-' + token);
    const scorePill = document.getElementById('attn19-score-pill-' + token);
    const kVector = document.getElementById('attn19-vector-k-' + token);
    const qPath = document.getElementById('attn19-op-q-path-' + token);
    const qHead = document.getElementById('attn19-op-q-head-' + token);
    const kPath = document.getElementById('attn19-op-k-path-' + token);
    const kHead = document.getElementById('attn19-op-k-head-' + token);
    const outPath = document.getElementById('attn19-op-out-path-' + token);
    const outHead = document.getElementById('attn19-op-out-head-' + token);
    if (!dotNode || !scorePill || !kVector || !qPath || !qHead || !kPath || !kHead || !outPath || !outHead) return;

    const dotTop = anchor(dotNode, 0.5, 0);
    const dotBottom = anchor(dotNode, 0.5, 1);
    const dotRight = anchor(dotNode, 1, 0.5);
    const scoreLeft = anchor(scorePill, 0, 0.5);
    const kTop = anchor(kVector, 0.5, 0);
    const qTarget = {
      x: dotTop.x,
      y: dotTop.y + Math.max(stageH * 0.0024, 1)
    };
    const distanceX = Math.abs(compareSource.x - qTarget.x);
    const distanceY = qTarget.y - compareSource.y;
    const curveFactor = ATTN_QKV_COMPARE_CURVE_FACTORS[token] || 0.5;
    let controlY = compareSource.y + distanceY * curveFactor + distanceX * 0.01;
    const controlYMax = qTarget.y - Math.max(stageH * 0.004, 1.2);
    if (controlY > controlYMax) controlY = controlYMax;
    const cp1x = compareSource.x + (qTarget.x - compareSource.x) * 0.3;
    const cp2x = compareSource.x + (qTarget.x - compareSource.x) * 0.72;
    const qPathD = 'M ' + compareSource.x.toFixed(2) + ' ' + compareSource.y.toFixed(2)
      + ' C ' + cp1x.toFixed(2) + ' ' + controlY.toFixed(2)
      + ' ' + cp2x.toFixed(2) + ' ' + controlY.toFixed(2)
      + ' ' + qTarget.x.toFixed(2) + ' ' + qTarget.y.toFixed(2);
    qPath.setAttribute('d', qPathD);
    setArrowHeadPoints(qHead, { x: cp2x, y: controlY }, qTarget, headLenMain);

    const kPad = Math.max(stageH * 0.0035, 1.2);
    const kSource = { x: kTop.x, y: kTop.y - kPad };
    const kTarget = { x: dotBottom.x, y: dotBottom.y + kPad };
    const kControlY = kSource.y + (kTarget.y - kSource.y) * 0.54;
    const kCp1x = kSource.x + (kTarget.x - kSource.x) * 0.3;
    const kCp2x = kSource.x + (kTarget.x - kSource.x) * 0.72;
    const kPathD = 'M ' + kSource.x.toFixed(2) + ' ' + kSource.y.toFixed(2)
      + ' C ' + kCp1x.toFixed(2) + ' ' + kControlY.toFixed(2)
      + ' ' + kCp2x.toFixed(2) + ' ' + kControlY.toFixed(2)
      + ' ' + kTarget.x.toFixed(2) + ' ' + kTarget.y.toFixed(2);
    kPath.setAttribute('d', kPathD);
    setArrowHeadPoints(kHead, { x: kCp2x, y: kControlY }, kTarget, headLenMain);

    const outPad = Math.max(stageH * 0.0035, 1.2);
    const outSource = { x: dotRight.x + outPad, y: dotRight.y };
    const outTarget = {
      x: Math.max(outSource.x + Math.max(stageH * 0.01, 3), scoreLeft.x - outPad),
      y: scoreLeft.y
    };
    const outPathD = 'M ' + outSource.x.toFixed(2) + ' ' + outSource.y.toFixed(2)
      + ' L ' + outTarget.x.toFixed(2) + ' ' + outTarget.y.toFixed(2);
    outPath.setAttribute('d', outPathD);
    setArrowHeadPoints(outHead, outSource, outTarget, headLenMinor);
  });

  if (state.attentionQkv.step < 4) {
    syncAttentionQkvCompareVisuals(0);
  } else if (state.attentionQkv.compareDone) {
    syncAttentionQkvCompareVisuals(ATTN_QKV_COMPARE_TOKENS.length);
  }
}

function setAttentionQkvStep(step) {
  const slide = document.getElementById('slide-20');
  const takeaway = document.getElementById('attn19-takeaway');
  if (!slide || !takeaway) return;

  const prevStep = state.attentionQkv.step;
  const clamped = Math.max(0, Math.min(ATTN_QKV_MAX_STEP, step));
  state.attentionQkv.step = clamped;
  const cleanupQk = clamped >= 5;

  slide.classList.toggle('attn19-show-k', clamped >= 1 && !cleanupQk);
  slide.classList.toggle('attn19-show-v', clamped >= 2);
  slide.classList.toggle('attn19-show-q', clamped >= 3 && !cleanupQk);
  slide.classList.toggle('attn19-show-scores', clamped >= 4);
  slide.classList.toggle('attn19-show-compare', clamped === 4);
  slide.classList.toggle('attn19-cleanup-qk', cleanupQk);
  takeaway.innerHTML = ATTN_QKV_TAKEAWAYS[clamped] || ATTN_QKV_TAKEAWAYS[0];

  if (clamped !== 4) {
    clearAttentionQkvCompareTimers();
    state.attentionQkv.compareDone = false;
    state.attentionQkv.compareVisibleCount = 0;
    slide.classList.remove('attn19-compare-done');
  }

  updateAttentionQkvOverlay();

  if (clamped === 4) {
    if (prevStep < 4 || !state.attentionQkv.compareDone) {
      runAttentionQkvCompareSequence();
    } else {
      state.attentionQkv.compareVisibleCount = ATTN_QKV_COMPARE_TOKENS.length;
      state.attentionQkv.compareDone = true;
      slide.classList.add('attn19-compare-done');
      syncAttentionQkvCompareVisuals(state.attentionQkv.compareVisibleCount);
    }
  } else {
    syncAttentionQkvCompareVisuals(0);
  }

  requestAnimationFrame(updateAttentionQkvOverlay);
  if (state.attentionQkv.overlayTimer) clearTimeout(state.attentionQkv.overlayTimer);
  state.attentionQkv.overlayTimer = setTimeout(() => {
    updateAttentionQkvOverlay();
    state.attentionQkv.overlayTimer = null;
  }, 240);
}

function initAttentionQkvSlide() {
  const slide = document.getElementById('slide-20');
  const cols = document.getElementById('attn19-cols');
  const qVector = document.getElementById('attn19-q-sat');
  if (!slide || !cols) return;

  if (!state.attentionQkv.initialized) {
    cols.innerHTML = '';
    ATTN_QKV_TOKENS.forEach((token) => {
      cols.appendChild(createAttentionQkvColumn(token));
    });

    if (!state.attentionQkv.resizeBound) {
      addTrackedListener(window, 'resize', () => {
        if (!state.attentionQkv.initialized) return;
        clearAttentionQkvCompareTimers();
        if (state.attentionQkv.step === 4) {
          state.attentionQkv.compareDone = true;
          state.attentionQkv.compareVisibleCount = ATTN_QKV_COMPARE_TOKENS.length;
          const activeSlide = document.getElementById('slide-20');
          if (activeSlide) activeSlide.classList.add('attn19-compare-done');
        } else {
          state.attentionQkv.compareDone = false;
          state.attentionQkv.compareVisibleCount = 0;
        }
        updateAttentionQkvOverlay();
        if (state.attentionQkv.step >= 4 && state.attentionQkv.compareDone) {
          syncAttentionQkvCompareVisuals(state.attentionQkv.compareVisibleCount);
        } else {
          syncAttentionQkvCompareVisuals(0);
        }
      });
      state.attentionQkv.resizeBound = true;
    }

    state.attentionQkv.initialized = true;
  }

  if (qVector) {
    qVector.innerHTML = '';
    populateVectorRect(qVector, ATTN_QKV_QUERY_VECTOR, 'attn19-vector-divider');
  }

  const takeaway = document.getElementById('attn19-takeaway');
  if (takeaway) takeaway.innerHTML = ATTN_QKV_TAKEAWAYS[state.attentionQkv.step] || ATTN_QKV_TAKEAWAYS[0];
  updateAttentionQkvOverlay();
  if (state.attentionQkv.step >= 4 && state.attentionQkv.compareDone) {
    syncAttentionQkvCompareVisuals(state.attentionQkv.compareVisibleCount);
  } else {
    syncAttentionQkvCompareVisuals(0);
  }
}

function runAttentionQkvStep() {
  if (!state.attentionQkv.initialized) initAttentionQkvSlide();
  if (state.attentionQkv.step >= ATTN_QKV_MAX_STEP) return false;
  setAttentionQkvStep(state.attentionQkv.step + 1);
  return true;
}

function resetAttentionQkvSlide() {
  const slide = document.getElementById('slide-20');
  if (!slide) return;
  if (state.attentionQkv.overlayTimer) {
    clearTimeout(state.attentionQkv.overlayTimer);
    state.attentionQkv.overlayTimer = null;
  }
  clearAttentionQkvCompareTimers();
  state.attentionQkv.compareDone = false;
  state.attentionQkv.compareVisibleCount = 0;
  slide.classList.remove('attn19-compare-done');
  setAttentionQkvStep(0);
}

/* =====================================================
   Slide-21 (Step 3) — scale + softmax -> attention weights
   ===================================================== */
