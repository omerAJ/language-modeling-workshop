function createAttentionStep4Column(token) {
  const col = createEl('div', {
    className: 'attn19-col attn22-col',
    id: 'attn22-col-' + token,
    dataset: { token }
  });

  const chipWrap = createEl('div', { className: 'attn19-chip-wrap' });
  chipWrap.appendChild(createEl('div', {
    className: 'attn19-token-chip',
    id: 'attn22-chip-' + token,
    text: token
  }));
  col.appendChild(chipWrap);

  const xLabel = createMathSubLabel('x', token, 'attn19-vector-label');
  xLabel.id = 'attn22-x-label-' + token;
  const xWrap = createEl('div', {
    className: 'attn19-vector-wrap attn19-vector-wrap-x',
    id: 'attn22-vector-wrap-x-' + token
  }, [
    xLabel,
    createAttentionStep4VectorRect('attn22-vector-x-' + token, '', ATTN_QKV_X_VECTORS[token] || ATTN_QKV_QUERY_VECTOR)
  ]);
  col.appendChild(xWrap);

  const scoreOpWrap = createEl('div', {
    className: 'attn19-score-op-wrap',
    id: 'attn22-score-op-wrap-' + token
  });
  const scoreQual = ATTN_STEP4_SCORE_QUAL[token] || { tier: 'medium' };
  const scorePill = createEl('div', {
    className: 'attn19-score-pill',
    id: 'attn22-score-pill-' + token,
    dataset: { qual: scoreQual.tier }
  }, formatScoreValue(ATTN_STEP4_SCORE_BY_TOKEN[token]));
  scoreOpWrap.appendChild(scorePill);
  col.appendChild(scoreOpWrap);
  col.appendChild(createEl('div', {
    className: 'attn22-fly-pill',
    id: 'attn22-fly-pill-' + token,
    'aria-hidden': 'true'
  }));

  const kWrap = createEl('div', {
    className: 'attn19-vector-wrap attn19-vector-wrap-k',
    id: 'attn22-vector-wrap-k-' + token
  }, [
    createMathSubLabel('k', token, 'attn19-vector-label'),
    createAttentionStep4VectorRect('attn22-vector-k-' + token, 'attn19-vector-k', ATTN_QKV_KEY_VECTORS[token] || ATTN_QKV_QUERY_VECTOR)
  ]);
  col.appendChild(kWrap);

  const vLabel = createMathSubLabel('v', token, 'attn19-vector-label');
  vLabel.id = 'attn22-v-label-' + token;
  const vWrap = createEl('div', {
    className: 'attn19-vector-wrap attn19-vector-wrap-v',
    id: 'attn22-vector-wrap-v-' + token
  }, [
    vLabel,
    createAttentionStep4VectorRect('attn22-vector-v-' + token, '', ATTN_QKV_VALUE_VECTORS[token] || ATTN_QKV_QUERY_VECTOR)
  ]);
  col.appendChild(vWrap);

  return col;
}

function setAttentionStep4ValueVectorForToken(token, values, formatter = formatVectorValue) {
  const vector = document.getElementById('attn22-vector-v-' + token);
  if (!vector) return;
  const valueEls = vector.querySelectorAll('.attn19-vector-value');
  valueEls.forEach((el, idx) => {
    const value = Array.isArray(values) && typeof values[idx] !== 'undefined' ? values[idx] : 0;
    el.textContent = formatter(value);
  });
}

function setAttentionStep4ValueLabelForToken(token, mode = 'base') {
  const label = document.getElementById('attn22-v-label-' + token);
  if (!label) return;
  if (mode === 'aggregate' && token === ATTN_STEP4_MERGE_TARGET) {
    setMathHTML(label, formatTokenMath('o', ATTN_STEP4_MERGE_TARGET));
  } else if (mode === 'weighted') {
    const weight = ATTN_STEP4_WEIGHT_BY_TOKEN[token] || 0;
    setMathHTML(label, inlineMath(formatWeightValue(weight) + ' \\cdot ' + formatMathSubscript('v', token)));
  } else {
    setMathHTML(label, formatTokenMath('v', token));
  }
}

function setAttentionStep4AllValueVectorsBase() {
  ATTN_STEP4_TOKENS.forEach((token) => {
    setAttentionStep4ValueVectorForToken(token, ATTN_QKV_VALUE_VECTORS[token] || [], formatVectorValue);
    setAttentionStep4ValueLabelForToken(token, 'base');
  });
}

function setAttentionStep4AllValueVectorsWeighted() {
  ATTN_STEP4_TOKENS.forEach((token) => {
    setAttentionStep4ValueVectorForToken(token, ATTN_STEP4_WEIGHTED_VALUE_VECTORS[token] || [], formatWeightValue);
    setAttentionStep4ValueLabelForToken(token, 'weighted');
  });
}

function setAttentionStep4XVectorForToken(token, values, formatter = formatVectorValue) {
  const vector = document.getElementById('attn22-vector-x-' + token);
  if (!vector) return;
  const valueEls = vector.querySelectorAll('.attn19-vector-value');
  valueEls.forEach((el, idx) => {
    const value = Array.isArray(values) && typeof values[idx] !== 'undefined' ? values[idx] : 0;
    el.textContent = formatter(value);
  });
}

function setAttentionStep4AllXVectorsBase() {
  ATTN_STEP4_TOKENS.forEach((token) => {
    setAttentionStep4XVectorForToken(token, ATTN_QKV_X_VECTORS[token] || [], formatVectorValue);
    const xLabel = document.getElementById('attn22-x-label-' + token);
    if (!xLabel) return;
    setMathHTML(xLabel, formatTokenMath('x', token));
  });
}

function setAttentionStep4SatXValues(values, formatter = formatWeightValue) {
  setAttentionStep4XVectorForToken(ATTN_STEP4_FOCUS, values, formatter);
}

function setAttentionStep4SatXLabel(mode = 'base') {
  const label = document.getElementById('attn22-x-label-' + ATTN_STEP4_FOCUS);
  if (!label) return;
  if (mode === 'residual') {
    setMathHTML(label, formatTokenMath('y', ATTN_STEP4_FOCUS));
    return;
  }
  setMathHTML(label, formatTokenMath('x', ATTN_STEP4_FOCUS));
}

function setAttentionStep4SatAggregateValues(values) {
  setAttentionStep4ValueVectorForToken(ATTN_STEP4_MERGE_TARGET, values, formatWeightValue);
}

function interpolateVectors(fromValues, toValues, t) {
  const from = Array.isArray(fromValues) ? fromValues : [];
  const to = Array.isArray(toValues) ? toValues : [];
  const len = Math.max(from.length, to.length);
  const clamped = Math.max(0, Math.min(1, Number(t) || 0));
  const out = [];
  for (let i = 0; i < len; i += 1) {
    const fromVal = Number(from[i]) || 0;
    const toVal = Number(to[i]) || 0;
    out.push(fromVal + (toVal - fromVal) * clamped);
  }
  return out;
}

function resetAttentionStep4FlyPill(token) {
  const flyPill = document.getElementById('attn22-fly-pill-' + token);
  if (!flyPill) return;
  flyPill.style.transition = 'none';
  flyPill.style.left = '-9999px';
  flyPill.style.top = '-9999px';
  flyPill.style.opacity = '0';
}

function animateAttentionStep4FlyPill(token) {
  const col = document.getElementById('attn22-col-' + token);
  const scorePill = document.getElementById('attn22-score-pill-' + token);
  const vVector = document.getElementById('attn22-vector-v-' + token);
  const flyPill = document.getElementById('attn22-fly-pill-' + token);
  if (!col || !scorePill || !vVector || !flyPill) return;

  const colRect = col.getBoundingClientRect();
  const scoreRect = scorePill.getBoundingClientRect();
  const vRect = vVector.getBoundingClientRect();
  const startX = scoreRect.left - colRect.left + scoreRect.width * 0.5;
  const startY = scoreRect.top - colRect.top + scoreRect.height * 0.5;
  const endX = vRect.left - colRect.left + vRect.width * 0.5;
  const endY = vRect.top - colRect.top + vRect.height * 0.5;
  const duration = Math.max(ATTN_STEP4_PAIR_ANIM_MS, 780);
  const fadeDuration = 180;
  const fadeStart = Math.max(Math.floor(duration * 0.82), duration - fadeDuration);

  flyPill.textContent = formatWeightValue(ATTN_STEP4_WEIGHT_BY_TOKEN[token] || 0);
  flyPill.style.transition = 'none';
  flyPill.style.left = startX.toFixed(2) + 'px';
  flyPill.style.top = startY.toFixed(2) + 'px';
  flyPill.style.opacity = '0';
  void flyPill.getBoundingClientRect();

  flyPill.style.transition = 'left ' + duration + 'ms cubic-bezier(0.2, 0.75, 0.3, 1), top ' + duration + 'ms cubic-bezier(0.2, 0.75, 0.3, 1)';
  flyPill.style.opacity = '0.94';
  requestAnimationFrame(() => {
    flyPill.style.left = endX.toFixed(2) + 'px';
    flyPill.style.top = endY.toFixed(2) + 'px';
  });

  const fadeTimer = setTimeout(() => {
    flyPill.style.transition = 'opacity ' + fadeDuration + 'ms ease';
    flyPill.style.opacity = '0';
  }, fadeStart);
  state.attentionStep4.pairTimers.push(fadeTimer);

  const cleanupTimer = setTimeout(() => {
    resetAttentionStep4FlyPill(token);
  }, duration + fadeDuration + 60);
  state.attentionStep4.pairTimers.push(cleanupTimer);
}

function clearAttentionStep4PairTimers() {
  state.attentionStep4.pairTimers.forEach((timerId) => clearTimeout(timerId));
  state.attentionStep4.pairTimers = [];
  state.attentionStep4.pairRafIds.forEach((rafId) => cancelAnimationFrame(rafId));
  state.attentionStep4.pairRafIds = [];
}

function resetAttentionStep4PairVisuals() {
  ATTN_STEP4_TOKENS.forEach((token) => {
    const col = document.getElementById('attn22-col-' + token);
    if (!col) return;
    col.classList.remove('is-pairing');
    col.classList.remove('is-flying');
    col.classList.remove('is-paired');
    resetAttentionStep4FlyPill(token);
  });
  state.attentionStep4.firstPairDone = false;
  state.attentionStep4.pairingDone = false;
  state.attentionStep4.pairVisibleCount = 0;
}

function settleAttentionStep4PairState() {
  clearAttentionStep4PairTimers();
  setAttentionStep4AllValueVectorsWeighted();
  ATTN_STEP4_TOKENS.forEach((token) => {
    const col = document.getElementById('attn22-col-' + token);
    if (!col) return;
    col.classList.remove('is-pairing');
    col.classList.remove('is-flying');
    col.classList.add('is-paired');
    resetAttentionStep4FlyPill(token);
  });
  state.attentionStep4.firstPairDone = true;
  state.attentionStep4.pairVisibleCount = ATTN_STEP4_TOKENS.length;
  state.attentionStep4.pairingDone = true;
}

function animateAttentionStep4PairToken(token, delayMs, isLastToken = false, onComplete = null) {
  const timerId = setTimeout(() => {
    if (state.attentionStep4.step < 2) return;
    const col = document.getElementById('attn22-col-' + token);
    if (!col) return;
    col.classList.add('is-pairing');
    col.classList.add('is-flying');
    col.classList.remove('is-paired');
    setAttentionStep4ValueLabelForToken(token, 'weighted');
    animateAttentionStep4FlyPill(token);

    const baseValues = ATTN_QKV_VALUE_VECTORS[token] || [];
    const weightedValues = ATTN_STEP4_WEIGHTED_VALUE_VECTORS[token] || baseValues;
    const startTime = performance.now();

    const animate = (now) => {
      if (state.attentionStep4.step < 2) return;
      const progress = Math.max(0, Math.min(1, (now - startTime) / ATTN_STEP4_PAIR_ANIM_MS));
      const eased = progress < 0.5
        ? (2 * progress * progress)
        : (1 - Math.pow(-2 * progress + 2, 2) / 2);
      const values = baseValues.map((value, idx) => {
        const target = typeof weightedValues[idx] === 'number' ? weightedValues[idx] : 0;
        return value + (target - value) * eased;
      });
      setAttentionStep4ValueVectorForToken(token, values, formatWeightValue);

      if (progress < 1) {
        const rafId = requestAnimationFrame(animate);
        state.attentionStep4.pairRafIds.push(rafId);
        return;
      }

      col.classList.remove('is-pairing');
      col.classList.remove('is-flying');
      col.classList.add('is-paired');
      state.attentionStep4.pairVisibleCount = Math.min(
        ATTN_STEP4_TOKENS.length,
        state.attentionStep4.pairVisibleCount + 1
      );
      if (typeof onComplete === 'function') onComplete();
      if (isLastToken) {
        state.attentionStep4.pairingDone = true;
      }
    };

    const rafId = requestAnimationFrame(animate);
    state.attentionStep4.pairRafIds.push(rafId);
  }, delayMs);

  state.attentionStep4.pairTimers.push(timerId);
}

function settleAttentionStep4PairToken(token) {
  const col = document.getElementById('attn22-col-' + token);
  setAttentionStep4ValueVectorForToken(token, ATTN_STEP4_WEIGHTED_VALUE_VECTORS[token] || [], formatWeightValue);
  setAttentionStep4ValueLabelForToken(token, 'weighted');
  if (!col) return;
  col.classList.remove('is-pairing');
  col.classList.remove('is-flying');
  col.classList.add('is-paired');
  resetAttentionStep4FlyPill(token);
}

function runAttentionStep4FirstPair() {
  clearAttentionStep4PairTimers();
  resetAttentionStep4PairVisuals();
  setAttentionStep4AllValueVectorsBase();
  if (!ATTN_STEP4_TOKENS.length) {
    state.attentionStep4.firstPairDone = true;
    state.attentionStep4.pairingDone = true;
    return;
  }

  const firstToken = ATTN_STEP4_TOKENS[0];
  const isOnlyToken = ATTN_STEP4_TOKENS.length === 1;
  animateAttentionStep4PairToken(firstToken, 0, isOnlyToken, () => {
    state.attentionStep4.firstPairDone = true;
  });
}

function runAttentionStep4RemainingPairs() {
  if (!ATTN_STEP4_TOKENS.length) {
    state.attentionStep4.firstPairDone = true;
    state.attentionStep4.pairingDone = true;
    return;
  }
  clearAttentionStep4PairTimers();
  const firstToken = ATTN_STEP4_TOKENS[0];
  settleAttentionStep4PairToken(firstToken);
  state.attentionStep4.firstPairDone = true;

  const restTokens = ATTN_STEP4_TOKENS.slice(1);
  if (!restTokens.length) {
    state.attentionStep4.pairingDone = true;
    return;
  }

  state.attentionStep4.pairingDone = false;
  const restStartDelay = ATTN_STEP4_PAIR_FIRST_HOLD_MS;
  restTokens.forEach((token, idx) => {
    animateAttentionStep4PairToken(
      token,
      restStartDelay + idx * ATTN_STEP4_PAIR_STAGGER_MS,
      idx === restTokens.length - 1
    );
  });
}

function clearAttentionStep4MergeTimers() {
  state.attentionStep4.mergeTimers.forEach((timerId) => clearTimeout(timerId));
  state.attentionStep4.mergeTimers = [];
  state.attentionStep4.mergeRafIds.forEach((rafId) => cancelAnimationFrame(rafId));
  state.attentionStep4.mergeRafIds = [];
}

function clearAttentionStep4MergeGhosts() {
  const layer = document.getElementById('attn22-merge-layer');
  if (!layer) return;
  layer.querySelectorAll('.attn22-merge-ghost').forEach((ghost) => ghost.remove());
}

function resetAttentionStep4MergeVisuals() {
  const slide = document.getElementById('slide-22');
  clearAttentionStep4MergeTimers();
  clearAttentionStep4MergeGhosts();
  ATTN_STEP4_TOKENS.forEach((token) => {
    const col = document.getElementById('attn22-col-' + token);
    if (!col) return;
    col.classList.remove('is-merged-out');
    col.classList.remove('is-merge-target');
  });
  if (slide) slide.classList.remove('attn22-show-merged');
  state.attentionStep4.mergeDone = false;
  state.attentionStep4.mergeVisibleCount = 0;
}

function createAttentionStep4MergeGhost(token) {
  const stage = document.getElementById('attn22-stage');
  const layer = document.getElementById('attn22-merge-layer');
  const sourceVector = document.getElementById('attn22-vector-v-' + token);
  if (!stage || !layer || !sourceVector) return null;

  const stageRect = stage.getBoundingClientRect();
  const sourceRect = sourceVector.getBoundingClientRect();
  if (stageRect.width < 1 || stageRect.height < 1 || sourceRect.width < 1 || sourceRect.height < 1) return null;

  const ghost = sourceVector.cloneNode(true);
  const stripIds = (node) => {
    if (!node || node.nodeType !== 1) return;
    node.removeAttribute('id');
    Array.from(node.children).forEach(stripIds);
  };
  stripIds(ghost);
  ghost.classList.add('attn22-merge-ghost');
  ghost.style.left = (sourceRect.left - stageRect.left).toFixed(2) + 'px';
  ghost.style.top = (sourceRect.top - stageRect.top).toFixed(2) + 'px';
  ghost.style.width = sourceRect.width.toFixed(2) + 'px';
  ghost.style.height = sourceRect.height.toFixed(2) + 'px';
  ghost.style.opacity = '0.94';
  ghost.style.transform = 'translate3d(0, 0, 0)';
  layer.appendChild(ghost);
  return ghost;
}

function settleAttentionStep4MergedState() {
  const slide = document.getElementById('slide-22');
  clearAttentionStep4MergeTimers();
  clearAttentionStep4MergeGhosts();

  ATTN_STEP4_TOKENS.forEach((token) => {
    const col = document.getElementById('attn22-col-' + token);
    if (!col) return;
    col.classList.remove('is-merge-target');
    col.classList.toggle('is-merged-out', token !== ATTN_STEP4_MERGE_TARGET);
  });

  const targetCol = document.getElementById('attn22-col-' + ATTN_STEP4_MERGE_TARGET);
  if (targetCol) targetCol.classList.add('is-merge-target');
  setAttentionStep4SatAggregateValues(ATTN_STEP4_AGG_VECTOR);
  setAttentionStep4ValueLabelForToken(ATTN_STEP4_MERGE_TARGET, 'aggregate');

  if (slide) slide.classList.add('attn22-show-merged');
  state.attentionStep4.mergeDone = true;
  state.attentionStep4.mergeVisibleCount = ATTN_STEP4_MERGE_ORDER.length;
}

function animateAttentionStep4MergeToken(token, delayMs, runningFrom, runningTo, isLastToken = false, onComplete = null) {
  const timerId = setTimeout(() => {
    if (state.attentionStep4.step < 4) return;

    const slide = document.getElementById('slide-22');
    const stage = document.getElementById('attn22-stage');
    const sourceVector = document.getElementById('attn22-vector-v-' + token);
    const sourceCol = document.getElementById('attn22-col-' + token);
    const targetVector = document.getElementById('attn22-vector-v-' + ATTN_STEP4_MERGE_TARGET);
    const targetCol = document.getElementById('attn22-col-' + ATTN_STEP4_MERGE_TARGET);
    if (!stage || !sourceVector || !targetVector || !targetCol) return;

    targetCol.classList.add('is-merge-target');
    setAttentionStep4ValueLabelForToken(ATTN_STEP4_MERGE_TARGET, 'aggregate');
    setAttentionStep4SatAggregateValues(runningFrom);

    let ghost = null;
    if (token !== ATTN_STEP4_MERGE_TARGET) {
      ghost = createAttentionStep4MergeGhost(token);
      if (ghost) {
        const stageRect = stage.getBoundingClientRect();
        const sourceRect = sourceVector.getBoundingClientRect();
        const targetRect = targetVector.getBoundingClientRect();
        const fromX = sourceRect.left - stageRect.left + sourceRect.width * 0.5;
        const fromY = sourceRect.top - stageRect.top + sourceRect.height * 0.5;
        const toX = targetRect.left - stageRect.left + targetRect.width * 0.5;
        const toY = targetRect.top - stageRect.top + targetRect.height * 0.5;
        const dx = toX - fromX;
        const dy = toY - fromY;
        requestAnimationFrame(() => {
          ghost.style.transform = 'translate3d(' + dx.toFixed(2) + 'px, ' + dy.toFixed(2) + 'px, 0) scale(0.9)';
        });
      }
    }

    const startTime = performance.now();
    const animate = (now) => {
      if (state.attentionStep4.step < 4) return;
      const progress = Math.max(0, Math.min(1, (now - startTime) / ATTN_STEP4_VEC_MERGE_MS));
      const eased = progress < 0.5
        ? (2 * progress * progress)
        : (1 - Math.pow(-2 * progress + 2, 2) / 2);
      setAttentionStep4SatAggregateValues(interpolateVectors(runningFrom, runningTo, eased));

      if (progress < 1) {
        const rafId = requestAnimationFrame(animate);
        state.attentionStep4.mergeRafIds.push(rafId);
        return;
      }

      setAttentionStep4SatAggregateValues(runningTo);
      state.attentionStep4.mergeVisibleCount = Math.min(
        ATTN_STEP4_MERGE_ORDER.length,
        state.attentionStep4.mergeVisibleCount + 1
      );
      if (sourceCol && token !== ATTN_STEP4_MERGE_TARGET) {
        sourceCol.classList.add('is-merged-out');
      }
      if (ghost) {
        ghost.style.opacity = '0';
        const cleanupTimer = setTimeout(() => {
          if (ghost && ghost.parentNode) ghost.parentNode.removeChild(ghost);
          ghost = null;
        }, ATTN_STEP4_VEC_MERGE_FADE_MS + 30);
        state.attentionStep4.mergeTimers.push(cleanupTimer);
      }

      if (isLastToken) {
        setAttentionStep4SatAggregateValues(ATTN_STEP4_AGG_VECTOR);
        if (slide) slide.classList.add('attn22-show-merged');
        state.attentionStep4.mergeDone = true;
      }
      if (typeof onComplete === 'function') onComplete();
    };

    const rafId = requestAnimationFrame(animate);
    state.attentionStep4.mergeRafIds.push(rafId);
  }, delayMs);

  state.attentionStep4.mergeTimers.push(timerId);
}

function runAttentionStep4VectorMergeSequence(onComplete = null) {
  const slide = document.getElementById('slide-22');
  resetAttentionStep4MergeVisuals();
  if (slide) slide.classList.remove('attn22-show-merged');

  const mergeOrder = ATTN_STEP4_MERGE_ORDER.filter((token) => ATTN_STEP4_TOKENS.includes(token));
  if (!mergeOrder.length) {
    setAttentionStep4SatAggregateValues(ATTN_STEP4_AGG_VECTOR);
    setAttentionStep4ValueLabelForToken(ATTN_STEP4_MERGE_TARGET, 'aggregate');
    state.attentionStep4.mergeDone = true;
    if (slide) slide.classList.add('attn22-show-merged');
    if (typeof onComplete === 'function') onComplete();
    return;
  }

  const targetCol = document.getElementById('attn22-col-' + ATTN_STEP4_MERGE_TARGET);
  if (targetCol) targetCol.classList.add('is-merge-target');
  setAttentionStep4ValueLabelForToken(ATTN_STEP4_MERGE_TARGET, 'aggregate');

  let running = ATTN_STEP4_AGG_VECTOR.map(() => 0);
  setAttentionStep4SatAggregateValues(running);
  state.attentionStep4.mergeDone = false;
  state.attentionStep4.mergeVisibleCount = 0;

  mergeOrder.forEach((token, idx) => {
    const weighted = ATTN_STEP4_WEIGHTED_VALUE_VECTORS[token] || [];
    const fromValues = running.slice();
    const toValues = running.map((value, compIdx) => value + (Number(weighted[compIdx]) || 0));
    const isLast = idx === mergeOrder.length - 1;

    animateAttentionStep4MergeToken(
      token,
      idx * ATTN_STEP4_VEC_MERGE_STAGGER_MS,
      fromValues,
      toValues,
      isLast,
      () => {
        if (!isLast) return;
        if (state.attentionStep4.step < 4) return;
        setAttentionStep4SatAggregateValues(ATTN_STEP4_AGG_VECTOR);
        setAttentionStep4ValueLabelForToken(ATTN_STEP4_MERGE_TARGET, 'aggregate');
        state.attentionStep4.mergeDone = true;
        state.attentionStep4.mergeVisibleCount = mergeOrder.length;
        if (slide) slide.classList.add('attn22-show-merged');
        if (typeof onComplete === 'function') onComplete();
      }
    );

    running = toValues;
  });
}

function clearAttentionStep4ResidualTimers() {
  state.attentionStep4.residualTimers.forEach((timerId) => clearTimeout(timerId));
  state.attentionStep4.residualTimers = [];
  state.attentionStep4.residualRafIds.forEach((rafId) => cancelAnimationFrame(rafId));
  state.attentionStep4.residualRafIds = [];
}

function clearAttentionStep4ResidualGhosts() {
  const layer = document.getElementById('attn22-merge-layer');
  if (!layer) return;
  layer.querySelectorAll('.attn22-residual-ghost').forEach((ghost) => ghost.remove());
}

function resetAttentionStep4ResidualVisuals(options = {}) {
  const settings = Object.assign({ restoreX: true }, options);
  clearAttentionStep4ResidualTimers();
  clearAttentionStep4ResidualGhosts();
  ATTN_STEP4_TOKENS.forEach((token) => {
    const col = document.getElementById('attn22-col-' + token);
    if (!col) return;
    col.classList.remove('is-residual-target');
    col.classList.remove('is-residual-consumed');
  });
  if (settings.restoreX) {
    setAttentionStep4SatXValues(ATTN_STEP4_RESIDUAL_INPUT_VECTOR, formatVectorValue);
    setAttentionStep4SatXLabel('base');
  }
  state.attentionStep4.residualDone = false;
}

function createAttentionStep4ResidualGhost() {
  const stage = document.getElementById('attn22-stage');
  const layer = document.getElementById('attn22-merge-layer');
  const sourceVector = document.getElementById('attn22-vector-v-' + ATTN_STEP4_MERGE_TARGET);
  if (!stage || !layer || !sourceVector) return null;

  const stageRect = stage.getBoundingClientRect();
  const sourceRect = sourceVector.getBoundingClientRect();
  if (stageRect.width < 1 || stageRect.height < 1 || sourceRect.width < 1 || sourceRect.height < 1) return null;

  const ghost = sourceVector.cloneNode(true);
  const stripIds = (node) => {
    if (!node || node.nodeType !== 1) return;
    node.removeAttribute('id');
    Array.from(node.children).forEach(stripIds);
  };
  stripIds(ghost);
  ghost.classList.add('attn22-residual-ghost');
  ghost.style.left = (sourceRect.left - stageRect.left).toFixed(2) + 'px';
  ghost.style.top = (sourceRect.top - stageRect.top).toFixed(2) + 'px';
  ghost.style.width = sourceRect.width.toFixed(2) + 'px';
  ghost.style.height = sourceRect.height.toFixed(2) + 'px';
  layer.appendChild(ghost);
  return ghost;
}

function settleAttentionStep4ResidualState() {
  clearAttentionStep4ResidualTimers();
  clearAttentionStep4ResidualGhosts();
  const satCol = document.getElementById('attn22-col-' + ATTN_STEP4_FOCUS);
  if (satCol) {
    satCol.classList.remove('is-residual-target');
    satCol.classList.add('is-residual-consumed');
  }
  setAttentionStep4SatXValues(ATTN_STEP4_RESIDUAL_OUTPUT_VECTOR, formatWeightValue);
  setAttentionStep4SatXLabel('residual');
  state.attentionStep4.residualDone = true;
}

function runAttentionStep4ResidualSequence() {
  const stage = document.getElementById('attn22-stage');
  const satCol = document.getElementById('attn22-col-' + ATTN_STEP4_FOCUS);
  const satVVector = document.getElementById('attn22-vector-v-' + ATTN_STEP4_FOCUS);
  const satXVector = document.getElementById('attn22-vector-x-' + ATTN_STEP4_FOCUS);
  if (!stage || !satCol || !satVVector || !satXVector) {
    settleAttentionStep4ResidualState();
    return;
  }

  resetAttentionStep4ResidualVisuals({ restoreX: false });
  satCol.classList.add('is-residual-target');
  satCol.classList.remove('is-residual-consumed');
  setAttentionStep4SatXLabel('base');
  setAttentionStep4SatXValues(ATTN_STEP4_RESIDUAL_INPUT_VECTOR, formatWeightValue);

  const ghost = createAttentionStep4ResidualGhost();
  if (ghost) {
    const stageRect = stage.getBoundingClientRect();
    const vRect = satVVector.getBoundingClientRect();
    const xRect = satXVector.getBoundingClientRect();
    const fromX = vRect.left - stageRect.left + vRect.width * 0.5;
    const fromY = vRect.top - stageRect.top + vRect.height * 0.5;
    const toX = xRect.left - stageRect.left + xRect.width * 0.5;
    const toY = xRect.top - stageRect.top + xRect.height * 0.5;
    const dx = toX - fromX;
    const dy = toY - fromY;
    requestAnimationFrame(() => {
      ghost.style.transform = 'translate3d(' + dx.toFixed(2) + 'px, ' + dy.toFixed(2) + 'px, 0) scale(0.9)';
    });
  }

  const startValues = ATTN_STEP4_RESIDUAL_INPUT_VECTOR.slice();
  const targetValues = ATTN_STEP4_RESIDUAL_OUTPUT_VECTOR.slice();
  const startTime = performance.now();
  const animate = (now) => {
    if (state.attentionStep4.step < 5) return;
    const progress = Math.max(0, Math.min(1, (now - startTime) / ATTN_STEP4_RESIDUAL_ANIM_MS));
    const eased = progress < 0.5
      ? (2 * progress * progress)
      : (1 - Math.pow(-2 * progress + 2, 2) / 2);
    setAttentionStep4SatXValues(interpolateVectors(startValues, targetValues, eased), formatWeightValue);

    if (progress < 1) {
      const rafId = requestAnimationFrame(animate);
      state.attentionStep4.residualRafIds.push(rafId);
      return;
    }

    setAttentionStep4SatXValues(targetValues, formatWeightValue);
    setAttentionStep4SatXLabel('residual');
    satCol.classList.remove('is-residual-target');
    satCol.classList.add('is-residual-consumed');
    if (ghost) {
      ghost.style.opacity = '0';
      const cleanupTimer = setTimeout(() => {
        if (ghost.parentNode) ghost.parentNode.removeChild(ghost);
      }, ATTN_STEP4_RESIDUAL_FADE_MS + 30);
      state.attentionStep4.residualTimers.push(cleanupTimer);
    }
    state.attentionStep4.residualDone = true;
  };
  const rafId = requestAnimationFrame(animate);
  state.attentionStep4.residualRafIds.push(rafId);
}

function clearAttentionStep4AggTimers() {
  state.attentionStep4.aggTimers.forEach((timerId) => clearTimeout(timerId));
  state.attentionStep4.aggTimers = [];
}

function renderAttentionStep4AggregationBase() {
  const wrap = document.getElementById('attn22-agg-wrap');
  const formula = document.getElementById('attn22-agg-formula');
  const outputVector = document.getElementById('attn22-agg-output-vector');
  if (!wrap || !formula || !outputVector) return;
  wrap.style.setProperty('--attn22-agg-collapse-ms', ATTN_STEP4_AGG_COLLAPSE_MS + 'ms');

  if (!formula.dataset.rendered) {
    formula.innerHTML = '';
    ATTN_STEP4_TOKENS.forEach((token, idx) => {
      const weight = ATTN_STEP4_WEIGHT_BY_TOKEN[token] || 0;
      const prefix = idx === 0 ? '' : '+ ';
      const term = createEl('span', {
        className: 'attn22-agg-term',
        id: 'attn22-agg-term-' + token
      });
      term.innerHTML = inlineMath(prefix + formatWeightValue(weight) + ' \\cdot ' + formatMathSubscript('v', token));
      formula.appendChild(term);
    });
    formula.dataset.rendered = '1';
    typesetMath(formula);
  }
}

function renderAttentionStep4PanelMode(mode = 'aggregation') {
  const wrap = document.getElementById('attn22-agg-wrap');
  const outputLabel = document.getElementById('attn22-agg-output-label');
  const outputVector = document.getElementById('attn22-agg-output-vector');
  if (!wrap || !outputLabel || !outputVector) return;

  const useResidual = mode === 'residual';
  const values = useResidual ? ATTN_STEP4_RESIDUAL_OUTPUT_VECTOR : ATTN_STEP4_AGG_VECTOR;
  wrap.classList.toggle('attn22-agg-mode-residual', useResidual);
  setMathHTML(outputLabel, useResidual
    ? inlineMath(formatMathSubscript('y', ATTN_STEP4_FOCUS) + ' \\approx')
    : inlineMath(formatMathSubscript('o', ATTN_STEP4_FOCUS) + ' \\approx'));
  outputVector.innerHTML = '';
  values.forEach((value) => {
    outputVector.appendChild(createEl('span', {
      className: 'attn22-agg-out-val',
      text: formatWeightValue(value)
    }));
  });
}

function syncAttentionStep4AggregationVisuals(visibleCount = state.attentionStep4.aggTermsVisibleCount, collapsed = state.attentionStep4.aggCollapsed) {
  const wrap = document.getElementById('attn22-agg-wrap');
  if (!wrap) return;
  const clamped = Math.max(0, Math.min(ATTN_STEP4_TOKENS.length, visibleCount));
  ATTN_STEP4_TOKENS.forEach((token, idx) => {
    const term = document.getElementById('attn22-agg-term-' + token);
    if (!term) return;
    term.classList.toggle('is-visible', idx < clamped);
  });
  wrap.classList.toggle('is-collapsed', !!collapsed);
}

function runAttentionStep4AggregationSequence() {
  renderAttentionStep4AggregationBase();
  renderAttentionStep4PanelMode('aggregation');
  clearAttentionStep4AggTimers();
  resetAttentionStep4MergeVisuals();
  state.attentionStep4.aggDone = false;
  state.attentionStep4.aggCollapsed = false;
  state.attentionStep4.aggTermsVisibleCount = 0;
  syncAttentionStep4AggregationVisuals(0, false);

  if (!ATTN_STEP4_TOKENS.length) {
    state.attentionStep4.aggTermsVisibleCount = 0;
    state.attentionStep4.aggCollapsed = true;
    settleAttentionStep4MergedState();
    state.attentionStep4.aggDone = true;
    syncAttentionStep4AggregationVisuals(0, true);
    return;
  }

  ATTN_STEP4_TOKENS.forEach((_, idx) => {
    const timerId = setTimeout(() => {
      if (state.attentionStep4.step < 4) return;
      state.attentionStep4.aggTermsVisibleCount = idx + 1;
      syncAttentionStep4AggregationVisuals(state.attentionStep4.aggTermsVisibleCount, false);
    }, idx * ATTN_STEP4_AGG_TERM_STAGGER_MS);
    state.attentionStep4.aggTimers.push(timerId);
  });

  const collapseDelay = (ATTN_STEP4_TOKENS.length - 1) * ATTN_STEP4_AGG_TERM_STAGGER_MS
    + ATTN_STEP4_AGG_COLLAPSE_DELAY_MS;
  const collapseTimer = setTimeout(() => {
    if (state.attentionStep4.step < 4) return;
    state.attentionStep4.aggTermsVisibleCount = ATTN_STEP4_TOKENS.length;
    state.attentionStep4.aggCollapsed = true;
    syncAttentionStep4AggregationVisuals(state.attentionStep4.aggTermsVisibleCount, true);
    runAttentionStep4VectorMergeSequence(() => {
      if (state.attentionStep4.step < 4) return;
      state.attentionStep4.aggDone = true;
    });
  }, collapseDelay);
  state.attentionStep4.aggTimers.push(collapseTimer);
}

function settleAttentionStep4AggregationState() {
  renderAttentionStep4AggregationBase();
  renderAttentionStep4PanelMode('aggregation');
  clearAttentionStep4AggTimers();
  state.attentionStep4.aggTermsVisibleCount = ATTN_STEP4_TOKENS.length;
  state.attentionStep4.aggCollapsed = true;
  settleAttentionStep4MergedState();
  state.attentionStep4.aggDone = true;
  syncAttentionStep4AggregationVisuals(state.attentionStep4.aggTermsVisibleCount, true);
}

function clearAttentionStep4CompareTimers() {
  state.attentionStep4.compareTimers.forEach((timerId) => clearTimeout(timerId));
  state.attentionStep4.compareTimers = [];
}

function getAttentionStep4CompareTriplet(token) {
  return [
    {
      path: document.getElementById('attn22-op-q-path-' + token),
      head: document.getElementById('attn22-op-q-head-' + token)
    },
    {
      path: document.getElementById('attn22-op-k-path-' + token),
      head: document.getElementById('attn22-op-k-head-' + token)
    },
    {
      path: document.getElementById('attn22-op-out-path-' + token),
      head: document.getElementById('attn22-op-out-head-' + token)
    }
  ];
}

function hideAttentionStep4CompareArrow(token) {
  getAttentionStep4CompareTriplet(token).forEach((elements) => {
    hideArrowElements({ path: elements.path, head: elements.head });
  });
}

function showAttentionStep4CompareArrow(token) {
  getAttentionStep4CompareTriplet(token).forEach((elements) => {
    showArrowElements({ path: elements.path, head: elements.head });
  });
}

function syncAttentionStep4CompareVisuals(visibleCount = state.attentionStep4.compareVisibleCount) {
  const clamped = Math.max(0, Math.min(ATTN_STEP4_COMPARE_TOKENS.length, visibleCount));
  ATTN_STEP4_COMPARE_TOKENS.forEach((token, idx) => {
    if (idx < clamped) showAttentionStep4CompareArrow(token);
    else hideAttentionStep4CompareArrow(token);
  });
}

function animateAttentionStep4CompareArrow(token) {
  hideAttentionStep4CompareArrow(token);
  const triplet = getAttentionStep4CompareTriplet(token).filter((elements) => elements.path && elements.head);
  if (!triplet.length) return;

  triplet.forEach((elements) => {
    const path = elements.path;
    path.style.transition = 'none';
    void path.getBoundingClientRect();
    path.style.transition = 'stroke-dashoffset ' + ATTN_STEP4_COMPARE_DRAW_MS + 'ms cubic-bezier(0.34, 0.08, 0.22, 1)';
    path.style.strokeDashoffset = '0';
  });

  const headTimer = setTimeout(() => {
    triplet.forEach((elements) => {
      const head = elements.head;
      head.style.transition = 'opacity ' + ATTN_STEP4_COMPARE_HEAD_FADE_MS + 'ms ease';
      head.style.opacity = '1';
    });
  }, Math.max(ATTN_STEP4_COMPARE_DRAW_MS - 70, 120));
  state.attentionStep4.compareTimers.push(headTimer);
}

function runAttentionStep4CompareSequence() {
  const slide = document.getElementById('slide-22');
  if (!slide) return;
  clearAttentionStep4CompareTimers();
  state.attentionStep4.compareDone = false;
  state.attentionStep4.compareVisibleCount = 0;
  slide.classList.remove('attn19-compare-done');
  syncAttentionStep4CompareVisuals(0);

  ATTN_STEP4_COMPARE_TOKENS.forEach((token, idx) => {
    const timer = setTimeout(() => {
      if (state.attentionStep4.step < 4) return;
      state.attentionStep4.compareVisibleCount = idx + 1;
      animateAttentionStep4CompareArrow(token);

      if (idx === ATTN_STEP4_COMPARE_TOKENS.length - 1) {
        const doneTimer = setTimeout(() => {
          if (state.attentionStep4.step < 4) return;
          state.attentionStep4.compareDone = true;
          state.attentionStep4.compareVisibleCount = ATTN_STEP4_COMPARE_TOKENS.length;
          slide.classList.add('attn19-compare-done');
          syncAttentionStep4CompareVisuals(state.attentionStep4.compareVisibleCount);
        }, ATTN_STEP4_COMPARE_DRAW_MS + ATTN_STEP4_SCORE_REVEAL_DELAY_MS);
        state.attentionStep4.compareTimers.push(doneTimer);
      }
    }, idx * ATTN_STEP4_COMPARE_STAGGER_MS);
    state.attentionStep4.compareTimers.push(timer);
  });
}

function updateAttentionStep4Overlay() {
  const stage = document.getElementById('attn22-stage');
  const overlay = document.getElementById('attn22-overlay');
  const qCallout = document.getElementById('attn22-q-callout');
  const qVector = document.getElementById('attn22-q-sat');
  const satDotNode = document.getElementById('attn22-score-op-wrap-' + ATTN_STEP4_FOCUS);
  const satScorePill = document.getElementById('attn22-score-pill-' + ATTN_STEP4_FOCUS);
  const vSatVector = document.getElementById('attn22-vector-v-' + ATTN_STEP4_FOCUS);
  const satChip = document.getElementById('attn22-chip-' + ATTN_STEP4_FOCUS);
  const satXVector = document.getElementById('attn22-vector-x-' + ATTN_STEP4_FOCUS);
  const satKVector = document.getElementById('attn22-vector-k-' + ATTN_STEP4_FOCUS);
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

  const rowLabels = document.getElementById('attn22-row-labels');
  const rowQLabel = document.getElementById('attn22-row-q-label');

  const satChipRect = satChip.getBoundingClientRect();
  const satScoreWrapRect = satDotNode.getBoundingClientRect();
  const satCenterX = satChipRect.left - stageRect.left + satChipRect.width * 0.5;
  const xSatBottom = anchor(satXVector, 0.5, 1);
  const dotSatTop = satScoreWrapRect.top - stageRect.top;
  const qCalloutRect = qCallout.getBoundingClientRect();
  const gapFromX = Math.max(stageH * 0.03, 10);
  const gapToDotNode = Math.max(stageH * 0.018, 7);
  const qDownNudge = Math.max(stageH * 0.012, 4);
  const qTopDesired = xSatBottom.y + gapFromX + qDownNudge;
  const qTopMax = dotSatTop - qCalloutRect.height - gapToDotNode;
  const qTop = Math.max(stageH * 0.02, Math.min(qTopDesired, qTopMax));
  // Align callout left edge with sat chip left edge (no arrow needed)
  const satChipLeft = satChipRect.left - stageRect.left;
  qCallout.style.left = satChipLeft.toFixed(2) + 'px';
  qCallout.style.top = qTop.toFixed(2) + 'px';
  qCallout.style.removeProperty('--attn19-q-nudge');

  // Q label follows the floating callout (X/S/K/V labels are CSS grid items and auto-align)
  if (rowQLabel && rowLabels) {
    const rowLabelsRect = rowLabels.getBoundingClientRect();
    const qLabelCenter = (stageRect.top + qTop + qCalloutRect.height * 0.5) - rowLabelsRect.top;
    rowQLabel.style.top = qLabelCenter.toFixed(2) + 'px';
  }

  const qBottom = anchor(qVector, 0.5, 1);
  const compareSource = { x: satCenterX, y: qBottom.y };
  const headLenMain = Math.max(stageH * 0.02, 4.8);
  const headLenMinor = Math.max(stageH * 0.017, 4.2);

  const spineRefWrap = document.getElementById('attn22-score-op-wrap-' + ATTN_STEP4_COMPARE_TOKENS[0]);
  const spineRefTop = spineRefWrap ? anchor(spineRefWrap, 0.5, 0) : null;
  const spineY = spineRefTop
    ? compareSource.y + (spineRefTop.y - compareSource.y) * 0.5
    : compareSource.y + Math.max(stageH * 0.06, 12);

  ATTN_STEP4_COMPARE_TOKENS.forEach((token) => {
    const dotNode = document.getElementById('attn22-score-op-wrap-' + token);
    const scorePill = document.getElementById('attn22-score-pill-' + token);
    const kVector = document.getElementById('attn22-vector-k-' + token);
    const qPath = document.getElementById('attn22-op-q-path-' + token);
    const qHead = document.getElementById('attn22-op-q-head-' + token);
    const kPath = document.getElementById('attn22-op-k-path-' + token);
    const kHead = document.getElementById('attn22-op-k-head-' + token);
    const outPath = document.getElementById('attn22-op-out-path-' + token);
    const outHead = document.getElementById('attn22-op-out-head-' + token);
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
    const qPathD = 'M ' + compareSource.x.toFixed(2) + ' ' + compareSource.y.toFixed(2)
      + ' L ' + compareSource.x.toFixed(2) + ' ' + spineY.toFixed(2)
      + ' L ' + dotTop.x.toFixed(2) + ' ' + spineY.toFixed(2)
      + ' L ' + dotTop.x.toFixed(2) + ' ' + qTarget.y.toFixed(2);
    qPath.setAttribute('d', qPathD);
    setArrowHeadPoints(qHead, { x: dotTop.x, y: spineY }, qTarget, headLenMain);

    const kPad = Math.max(stageH * 0.0035, 1.2);
    const kSource = { x: kTop.x, y: kTop.y - kPad };
    const kTarget = { x: dotBottom.x, y: dotBottom.y };
    const kMidY = kSource.y + (kTarget.y - kSource.y) * 0.5;
    // 3-segment orthogonal: up → horizontal → up
    const kPathD = 'M ' + kSource.x.toFixed(2) + ' ' + kSource.y.toFixed(2)
      + ' L ' + kSource.x.toFixed(2) + ' ' + kMidY.toFixed(2)
      + ' L ' + kTarget.x.toFixed(2) + ' ' + kMidY.toFixed(2)
      + ' L ' + kTarget.x.toFixed(2) + ' ' + kTarget.y.toFixed(2);
    kPath.setAttribute('d', kPathD);
    setArrowHeadPoints(kHead, { x: kTarget.x, y: kMidY }, kTarget, headLenMain);

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

  if (state.attentionStep4.step < 4) {
    syncAttentionStep4CompareVisuals(0);
  } else if (state.attentionStep4.compareDone) {
    syncAttentionStep4CompareVisuals(ATTN_STEP4_COMPARE_TOKENS.length);
  }
}

function renderAttentionStep4ScoreMode(step) {
  const scoreLabel = document.getElementById('attn22-row-s-label');
  const scoreCaption = document.getElementById('attn22-score-caption');
  const useWeights = step >= 1;

  if (scoreLabel) {
    setMathHTML(scoreLabel, useWeights ? 'Attention Weights \\(\\mathbf{a}\\)' : 'Scores \\(\\mathbf{s}\\)');
  }
  if (scoreCaption) {
    if (useWeights) {
      setMathHTML(scoreCaption, '\\(a_j = \\frac{\\exp(z_j)}{\\sum_{\\ell} \\exp(z_{\\ell})},\\; z_j = \\frac{s_j}{\\sqrt{d_k}}\\)');
    } else {
      setMathHTML(scoreCaption, '\\(s_j = q_{\\mathrm{sat}}^{\\mathsf{T}} k_j\\)');
    }
  }

  ATTN_STEP4_TOKENS.forEach((token) => {
    const scorePill = document.getElementById('attn22-score-pill-' + token);
    if (!scorePill) return;
    const value = useWeights ? ATTN_STEP4_WEIGHT_BY_TOKEN[token] : ATTN_STEP4_SCORE_BY_TOKEN[token];
    scorePill.textContent = useWeights ? formatWeightValue(value) : formatScoreValue(value);
  });
}

function setAttentionStep4Step(step) {
  const slide = document.getElementById('slide-22');
  const takeaway = document.getElementById('attn22-takeaway');
  if (!slide || !takeaway) return;

  const clamped = Math.max(0, Math.min(ATTN_STEP4_MAX_STEP, step));
  state.attentionStep4.step = clamped;

  slide.classList.remove('attn19-show-k');
  slide.classList.add('attn19-show-v');
  slide.classList.remove('attn19-show-q');
  slide.classList.add('attn19-show-scores');
  slide.classList.remove('attn19-show-compare');
  slide.classList.toggle('attn22-show-weights', clamped >= 1);
  slide.classList.toggle('attn22-show-pairing', clamped >= 2);
  slide.classList.toggle('attn22-show-aggregate', clamped >= 4);
  slide.classList.toggle('attn22-show-residual', clamped >= 5);
  if (clamped < 4) slide.classList.remove('attn22-show-merged');
  setMathHTML(takeaway, ATTN_STEP4_TAKEAWAYS[clamped] || ATTN_STEP4_TAKEAWAYS[0]);
  renderAttentionStep4ScoreMode(clamped);
  renderAttentionStep4PanelMode(clamped >= 5 ? 'residual' : 'aggregation');

  if (clamped < 2) {
    clearAttentionStep4PairTimers();
    resetAttentionStep4PairVisuals();
    setAttentionStep4AllValueVectorsBase();
  } else if (clamped === 2) {
    if (!state.attentionStep4.firstPairDone) {
      runAttentionStep4FirstPair();
    } else {
      settleAttentionStep4PairToken(ATTN_STEP4_TOKENS[0]);
    }
  } else if (clamped === 3) {
    if (!state.attentionStep4.pairingDone) {
      runAttentionStep4RemainingPairs();
    } else {
      settleAttentionStep4PairState();
    }
  } else {
    settleAttentionStep4PairState();
  }

  if (clamped < 4) {
    clearAttentionStep4AggTimers();
    state.attentionStep4.aggDone = false;
    state.attentionStep4.aggTermsVisibleCount = 0;
    state.attentionStep4.aggCollapsed = false;
    syncAttentionStep4AggregationVisuals(0, false);
    resetAttentionStep4MergeVisuals();
  } else if (clamped === 4) {
    if (!state.attentionStep4.aggDone) {
      runAttentionStep4AggregationSequence();
    } else {
      settleAttentionStep4AggregationState();
    }
  } else {
    settleAttentionStep4AggregationState();
  }

  if (clamped < 5) {
    renderAttentionStep4PanelMode('aggregation');
    resetAttentionStep4ResidualVisuals({ restoreX: true });
  } else {
    renderAttentionStep4PanelMode('residual');
    if (!state.attentionStep4.aggDone) {
      settleAttentionStep4AggregationState();
    }
    if (!state.attentionStep4.residualDone) {
      runAttentionStep4ResidualSequence();
    } else {
      settleAttentionStep4ResidualState();
    }
  }

  clearAttentionStep4CompareTimers();
  state.attentionStep4.compareDone = false;
  state.attentionStep4.compareVisibleCount = 0;
  slide.classList.remove('attn19-compare-done');

  updateAttentionStep4Overlay();
  syncAttentionStep4CompareVisuals(0);

  requestAnimationFrame(updateAttentionStep4Overlay);
  if (state.attentionStep4.overlayTimer) clearTimeout(state.attentionStep4.overlayTimer);
  state.attentionStep4.overlayTimer = setTimeout(() => {
    updateAttentionStep4Overlay();
    state.attentionStep4.overlayTimer = null;
  }, 240);
}

function initAttentionStep4Slide() {
  const slide = document.getElementById('slide-22');
  const cols = document.getElementById('attn22-cols');
  const qVector = document.getElementById('attn22-q-sat');
  if (!slide || !cols) return;

  if (!state.attentionStep4.initialized) {
    cols.innerHTML = '';
    ATTN_STEP4_TOKENS.forEach((token) => {
      cols.appendChild(createAttentionStep4Column(token));
    });

    if (!state.attentionStep4.resizeBound) {
      addTrackedListener(window, 'resize', () => {
        if (!state.attentionStep4.initialized) return;
        if (layoutState.syntheticResizeGuard) {
          updateAttentionStep4Overlay();
          syncAttentionStep4CompareVisuals(state.attentionStep4.compareVisibleCount);
          return;
        }
        clearAttentionStep4CompareTimers();
        if (state.attentionStep4.step >= 4) {
          state.attentionStep4.compareDone = true;
          state.attentionStep4.compareVisibleCount = ATTN_STEP4_COMPARE_TOKENS.length;
          const activeSlide = document.getElementById('slide-22');
          if (activeSlide) activeSlide.classList.add('attn19-compare-done');
        } else {
          state.attentionStep4.compareDone = false;
          state.attentionStep4.compareVisibleCount = 0;
        }
        updateAttentionStep4Overlay();
        if (state.attentionStep4.step >= 4 && state.attentionStep4.compareDone) {
          syncAttentionStep4CompareVisuals(state.attentionStep4.compareVisibleCount);
        } else {
          syncAttentionStep4CompareVisuals(0);
        }
        if (state.attentionStep4.step >= 5) {
          settleAttentionStep4ResidualState();
          renderAttentionStep4PanelMode('residual');
        } else if (state.attentionStep4.step >= 4) {
          settleAttentionStep4AggregationState();
          renderAttentionStep4PanelMode('aggregation');
        }
      });
      state.attentionStep4.resizeBound = true;
    }

    state.attentionStep4.initialized = true;
  }

  if (qVector) {
    qVector.innerHTML = '';
    populateVectorRect(qVector, ATTN_QKV_QUERY_VECTOR, 'attn19-vector-divider');
  }
  setAttentionStep4AllXVectorsBase();
  renderAttentionStep4AggregationBase();
  renderAttentionStep4PanelMode(state.attentionStep4.step >= 5 ? 'residual' : 'aggregation');
  clearAttentionStep4PairTimers();
  if (state.attentionStep4.step >= 2) {
    settleAttentionStep4PairState();
  } else {
    resetAttentionStep4PairVisuals();
    setAttentionStep4AllValueVectorsBase();
  }
  if (state.attentionStep4.step >= 4) {
    settleAttentionStep4AggregationState();
  } else {
    clearAttentionStep4AggTimers();
    state.attentionStep4.aggDone = false;
    state.attentionStep4.aggTermsVisibleCount = 0;
    state.attentionStep4.aggCollapsed = false;
    syncAttentionStep4AggregationVisuals(0, false);
    resetAttentionStep4MergeVisuals();
  }
  if (state.attentionStep4.step >= 5) {
    settleAttentionStep4ResidualState();
  } else {
    resetAttentionStep4ResidualVisuals({ restoreX: true });
  }
  renderAttentionStep4ScoreMode(state.attentionStep4.step);

  const takeaway = document.getElementById('attn22-takeaway');
  if (takeaway) setMathHTML(takeaway, ATTN_STEP4_TAKEAWAYS[state.attentionStep4.step] || ATTN_STEP4_TAKEAWAYS[0]);
  typesetMath(slide).then(() => {
    updateAttentionStep4Overlay();
  });
  updateAttentionStep4Overlay();
  if (state.attentionStep4.step >= 4 && state.attentionStep4.compareDone) {
    syncAttentionStep4CompareVisuals(state.attentionStep4.compareVisibleCount);
  } else {
    syncAttentionStep4CompareVisuals(0);
  }
}

function runAttentionStep4Step() {
  if (!state.attentionStep4.initialized) initAttentionStep4Slide();
  if (state.attentionStep4.step >= ATTN_STEP4_MAX_STEP) return false;
  setAttentionStep4Step(state.attentionStep4.step + 1);
  return true;
}

function resetAttentionStep4Slide() {
  const slide = document.getElementById('slide-22');
  if (!slide) return;
  if (state.attentionStep4.overlayTimer) {
    clearTimeout(state.attentionStep4.overlayTimer);
    state.attentionStep4.overlayTimer = null;
  }
  clearAttentionStep4CompareTimers();
  clearAttentionStep4PairTimers();
  clearAttentionStep4AggTimers();
  clearAttentionStep4MergeTimers();
  clearAttentionStep4MergeGhosts();
  clearAttentionStep4ResidualTimers();
  clearAttentionStep4ResidualGhosts();
  resetAttentionStep4PairVisuals();
  resetAttentionStep4MergeVisuals();
  resetAttentionStep4ResidualVisuals({ restoreX: true });
  state.attentionStep4.compareDone = false;
  state.attentionStep4.compareVisibleCount = 0;
  state.attentionStep4.aggDone = false;
  state.attentionStep4.aggTermsVisibleCount = 0;
  state.attentionStep4.aggCollapsed = false;
  state.attentionStep4.mergeDone = false;
  state.attentionStep4.mergeVisibleCount = 0;
  state.attentionStep4.residualDone = false;
  syncAttentionStep4AggregationVisuals(0, false);
  slide.classList.remove('attn19-compare-done');
  slide.classList.remove('attn22-show-aggregate');
  slide.classList.remove('attn22-show-residual');
  slide.classList.remove('attn22-show-merged');
  setAttentionStep4Step(0);
}

/* =====================================================
   Slide-23 — matrix view of the full sequence
   ===================================================== */
