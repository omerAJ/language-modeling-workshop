function createAttentionVectorRect(vectorId) {
  return createVectorRect({
    id: vectorId,
    baseClass: 'attn18-vector',
    dividerClass: 'attn18-vector-divider'
  });
}

function createAttentionTokenPrimitive(token) {
  const primitive = createEl('div', {
    className: 'attn18-token-primitive',
    dataset: { token }
  });

  const chipWrap = createEl('div', { className: 'attn18-chip-wrap' }, [
    createEl('div', {
      className: 'attn18-token-chip',
      id: 'attn18-chip-' + token,
      text: token
    })
  ]);
  primitive.appendChild(chipWrap);

  const vectorWrap = createEl('div', {
    className: 'attn18-vector-wrap',
    id: 'attn18-vector-wrap-' + token
  }, [
    createEl('span', {
      className: 'attn18-vector-label',
      html: formatTokenMath('x', token)
    }),
    createAttentionVectorRect('attn18-vector-' + token)
  ]);
  primitive.appendChild(vectorWrap);

  if (token === ATTN_INTRO_FOCUS) {
    const updatedWrap = createEl('div', { className: 'attn18-update-wrap' }, [
      createEl('span', {
        className: 'attn18-vector-label prime',
        html: inlineMath("x'_{\\mathrm{sat}}")
      }),
      createAttentionVectorRect('attn18-vector-sat-prime')
    ]);
    primitive.appendChild(updatedWrap);
  }

  return primitive;
}

function clearAttentionIntroFlowTimers() {
  state.attentionIntro.flowTimers.forEach((timerId) => clearTimeout(timerId));
  state.attentionIntro.flowTimers = [];
}

function hideAttentionIntroFlowArrow(path, head) {
  hideArrowElements({ path, head });
}

function showAttentionIntroFlowArrow(path, head) {
  showArrowElements({ path, head });
}

function getAttentionIntroFlowCount(step) {
  return Math.max(0, Math.min(ATTN_INTRO_FLOW_SOURCES.length, step - 2));
}

function syncAttentionIntroFlowVisuals(visibleCount = getAttentionIntroFlowCount(state.attentionIntro.step)) {
  ATTN_INTRO_FLOW_SOURCES.forEach((token, idx) => {
    const path = document.getElementById('attn18-flow-path-' + token);
    const head = document.getElementById('attn18-flow-head-' + token);
    if (!path || !head) return;
    if (idx < visibleCount) showAttentionIntroFlowArrow(path, head);
    else hideAttentionIntroFlowArrow(path, head);
  });

  const flowMicroCat = document.getElementById('attn18-flow-micro-cat');
  if (flowMicroCat) flowMicroCat.style.opacity = visibleCount >= 1 ? '1' : '0';
  const flowMicroMat = document.getElementById('attn18-flow-micro-mat');
  if (flowMicroMat) flowMicroMat.style.opacity = visibleCount >= 4 ? '1' : '0';
}

function animateAttentionIntroFlowArrow(token) {
  const path = document.getElementById('attn18-flow-path-' + token);
  const head = document.getElementById('attn18-flow-head-' + token);
  if (!path || !head) return;

  hideAttentionIntroFlowArrow(path, head);
  void path.getBoundingClientRect();
  path.style.transition = 'stroke-dashoffset ' + ATTN_INTRO_FLOW_ANIM_MS + 'ms cubic-bezier(0.34, 0.08, 0.22, 1)';
  path.style.strokeDashoffset = '0';

  const headTimer = setTimeout(() => {
    head.style.transition = 'opacity ' + ATTN_INTRO_FLOW_HEAD_FADE_MS + 'ms ease';
    head.style.opacity = '1';
    if (token === 'cat') {
      const flowMicroCat = document.getElementById('attn18-flow-micro-cat');
      if (flowMicroCat) flowMicroCat.style.opacity = '1';
    }
    if (token === 'mat') {
      const flowMicroMat = document.getElementById('attn18-flow-micro-mat');
      if (flowMicroMat) flowMicroMat.style.opacity = '1';
    }
  }, Math.max(ATTN_INTRO_FLOW_ANIM_MS - 70, 120));
  state.attentionIntro.flowTimers.push(headTimer);
}

function updateAttentionIntroOverlay() {
  const stage = document.getElementById('attn18-stage');
  const overlay = document.getElementById('attn18-overlay');
  const satChip = document.getElementById('attn18-chip-' + ATTN_INTRO_FOCUS);
  const satVector = document.getElementById('attn18-vector-sat');
  if (!stage || !overlay || !satChip || !satVector) return;

  const stageRect = stage.getBoundingClientRect();
  if (stageRect.width < 1 || stageRect.height < 1) return;
  overlay.setAttribute('viewBox', '0 0 ' + stageRect.width + ' ' + stageRect.height);

  const anchor = (el, x, y) => {
    const rect = el.getBoundingClientRect();
    return {
      x: rect.left - stageRect.left + rect.width * x,
      y: rect.top - stageRect.top + rect.height * y
    };
  };

  const satChipRect = satChip.getBoundingClientRect();
  const satTop = anchor(satChip, 0.5, 0);
  const satInlet = {
    x: satTop.x,
    y: satTop.y - Math.max(stageRect.height * 0.03, satChipRect.height * 0.2)
  };

  let flowMinX = satInlet.x;
  let flowMaxX = satInlet.x;
  let highestControlY = satInlet.y;
  const flowMicroCat = document.getElementById('attn18-flow-micro-cat');
  const flowMicroMat = document.getElementById('attn18-flow-micro-mat');

  ATTN_INTRO_FLOW_SOURCES.forEach((token) => {
    const sourceChip = document.getElementById('attn18-chip-' + token);
    const path = document.getElementById('attn18-flow-path-' + token);
    const head = document.getElementById('attn18-flow-head-' + token);
    if (!sourceChip || !path) return;

    if (token === ATTN_INTRO_FOCUS) {
      const loopStart = anchor(satChip, 0.34, 0);
      const loopEnd = anchor(satChip, 0.66, 0);
      const loopWidth = Math.max(satChipRect.width * 1.1, stageRect.width * 0.07);
      const loopLift = Math.max(stageRect.height * 0.18, satChipRect.height * 2.1);
      const cp1x = loopStart.x - loopWidth * 0.56;
      const cp2x = loopEnd.x + loopWidth * 0.56;
      const controlY = loopStart.y - loopLift;

      flowMinX = Math.min(flowMinX, loopStart.x, cp1x);
      flowMaxX = Math.max(flowMaxX, loopEnd.x, cp2x);
      highestControlY = Math.min(highestControlY, controlY);

      const d = 'M ' + loopStart.x.toFixed(2) + ' ' + loopStart.y.toFixed(2)
        + ' C ' + cp1x.toFixed(2) + ' ' + controlY.toFixed(2)
        + ' ' + cp2x.toFixed(2) + ' ' + controlY.toFixed(2)
        + ' ' + loopEnd.x.toFixed(2) + ' ' + loopEnd.y.toFixed(2);
      path.setAttribute('d', d);

      if (head) {
        const dirX = loopEnd.x - cp2x;
        const dirY = loopEnd.y - controlY;
        const dirLen = Math.hypot(dirX, dirY) || 1;
        const unitX = dirX / dirLen;
        const unitY = dirY / dirLen;
        const headLen = Math.max(stageRect.height * 0.023, 5.2);
        const wing = headLen * 0.54;
        const baseX = loopEnd.x - unitX * headLen;
        const baseY = loopEnd.y - unitY * headLen;
        const wingX = -unitY * wing;
        const wingY = unitX * wing;
        const points = loopEnd.x.toFixed(2) + ',' + loopEnd.y.toFixed(2)
          + ' ' + (baseX + wingX).toFixed(2) + ',' + (baseY + wingY).toFixed(2)
          + ' ' + (baseX - wingX).toFixed(2) + ',' + (baseY - wingY).toFixed(2);
        head.setAttribute('points', points);
      }
      return;
    }

    const source = anchor(sourceChip, 0.5, 0);
    const distanceX = Math.abs(source.x - satInlet.x);
    const topLimit = stageRect.height * 0.08;
    const baseLift = stageRect.height * (ATTN_INTRO_FLOW_HEIGHTS[token] || 0.2);
    const spreadLift = distanceX * 0.022;
    const controlY = Math.max(topLimit, Math.min(source.y, satInlet.y) - baseLift - spreadLift);
    const cp1x = source.x + (satInlet.x - source.x) * 0.3;
    const cp2x = source.x + (satInlet.x - source.x) * 0.74;

    flowMinX = Math.min(flowMinX, source.x);
    flowMaxX = Math.max(flowMaxX, source.x);
    highestControlY = Math.min(highestControlY, controlY);

    const d = 'M ' + source.x.toFixed(2) + ' ' + source.y.toFixed(2)
      + ' C ' + cp1x.toFixed(2) + ' ' + controlY.toFixed(2)
      + ' ' + cp2x.toFixed(2) + ' ' + controlY.toFixed(2)
      + ' ' + satInlet.x.toFixed(2) + ' ' + satInlet.y.toFixed(2);
    path.setAttribute('d', d);

    if (token === 'cat' && flowMicroCat) {
      const t = 0.44;
      const oneMinusT = 1 - t;
      const microX =
        (oneMinusT ** 3) * source.x +
        3 * (oneMinusT ** 2) * t * cp1x +
        3 * oneMinusT * (t ** 2) * cp2x +
        (t ** 3) * satInlet.x;
      const microY =
        (oneMinusT ** 3) * source.y +
        3 * (oneMinusT ** 2) * t * controlY +
        3 * oneMinusT * (t ** 2) * controlY +
        (t ** 3) * satInlet.y;
      flowMicroCat.setAttribute('x', microX.toFixed(2));
      flowMicroCat.setAttribute('y', (microY - Math.max(stageRect.height * 0.015, 4.5)).toFixed(2));
    }
    if (token === 'mat' && flowMicroMat) {
      const t = 0.52;
      const oneMinusT = 1 - t;
      const microX =
        (oneMinusT ** 3) * source.x +
        3 * (oneMinusT ** 2) * t * cp1x +
        3 * oneMinusT * (t ** 2) * cp2x +
        (t ** 3) * satInlet.x;
      const microY =
        (oneMinusT ** 3) * source.y +
        3 * (oneMinusT ** 2) * t * controlY +
        3 * oneMinusT * (t ** 2) * controlY +
        (t ** 3) * satInlet.y;
      flowMicroMat.setAttribute('x', microX.toFixed(2));
      flowMicroMat.setAttribute('y', (microY - Math.max(stageRect.height * 0.015, 4.5)).toFixed(2));
    }

    if (head) {
      const dirX = satInlet.x - cp2x;
      const dirY = satInlet.y - controlY;
      const dirLen = Math.hypot(dirX, dirY) || 1;
      const unitX = dirX / dirLen;
      const unitY = dirY / dirLen;
      const headLen = Math.max(stageRect.height * 0.023, 5.2);
      const wing = headLen * 0.54;
      const baseX = satInlet.x - unitX * headLen;
      const baseY = satInlet.y - unitY * headLen;
      const wingX = -unitY * wing;
      const wingY = unitX * wing;
      const points = satInlet.x.toFixed(2) + ',' + satInlet.y.toFixed(2)
        + ' ' + (baseX + wingX).toFixed(2) + ',' + (baseY + wingY).toFixed(2)
        + ' ' + (baseX - wingX).toFixed(2) + ',' + (baseY - wingY).toFixed(2);
      head.setAttribute('points', points);
    }
  });

  const flowCaption = document.getElementById('attn18-flow-caption');
  if (flowCaption) {
    const captionX = (flowMinX + flowMaxX) * 0.5;
    const captionY = Math.max(stageRect.height * 0.06, highestControlY - stageRect.height * 0.05);
    flowCaption.setAttribute('x', captionX.toFixed(2));
    flowCaption.setAttribute('y', captionY.toFixed(2));
    flowCaption.setAttribute('text-anchor', 'middle');
  }

  const satPrimeVector = document.getElementById('attn18-vector-sat-prime');
  const updateArrow = document.getElementById('attn18-update-arrow');
  const updateLabel = document.getElementById('attn18-update-text');
  if (!satPrimeVector || !updateArrow || !updateLabel) return;

  const from = anchor(satVector, 0.5, 1);
  const to = anchor(satPrimeVector, 0.5, 0);
  const satVectorRect = satVector.getBoundingClientRect();
  const updatePad = Math.max(stageRect.height * 0.007, satVectorRect.height * 0.16);
  updateArrow.setAttribute('x1', from.x.toFixed(2));
  updateArrow.setAttribute('y1', (from.y + updatePad).toFixed(2));
  updateArrow.setAttribute('x2', to.x.toFixed(2));
  updateArrow.setAttribute('y2', (to.y - updatePad).toFixed(2));
  updateLabel.setAttribute('x', (to.x + satVectorRect.width * 0.34).toFixed(2));
  updateLabel.setAttribute('y', (((from.y + to.y) * 0.5 - satVectorRect.height * 0.32) + 12).toFixed(2));
}

function setAttentionIntroStep(step) {
  const slide = document.getElementById('slide-18');
  const takeaway = document.getElementById('attn18-takeaway');
  if (!slide || !takeaway) return;

  const prevStep = state.attentionIntro.step;
  const clamped = Math.max(0, Math.min(ATTN_INTRO_MAX_STEP, step));
  const prevFlowCount = getAttentionIntroFlowCount(prevStep);
  const nextFlowCount = getAttentionIntroFlowCount(clamped);

  clearAttentionIntroFlowTimers();

  state.attentionIntro.step = clamped;
  slide.classList.toggle('attn18-show-vectors', clamped >= 1);
  slide.classList.toggle('attn18-show-focus', clamped >= 2);
  slide.classList.toggle('attn18-show-flow', clamped >= 3);
  slide.classList.toggle('attn18-show-update', clamped >= 8);
  takeaway.textContent = ATTN_INTRO_FOOTER;

  updateAttentionIntroOverlay();

  if (nextFlowCount > prevFlowCount) {
    if (nextFlowCount === prevFlowCount + 1) {
      syncAttentionIntroFlowVisuals(prevFlowCount);
      animateAttentionIntroFlowArrow(ATTN_INTRO_FLOW_SOURCES[nextFlowCount - 1]);
    } else {
      syncAttentionIntroFlowVisuals(nextFlowCount);
    }
  } else {
    syncAttentionIntroFlowVisuals(nextFlowCount);
  }

  requestAnimationFrame(updateAttentionIntroOverlay);
  if (state.attentionIntro.overlayTimer) clearTimeout(state.attentionIntro.overlayTimer);
  state.attentionIntro.overlayTimer = setTimeout(() => {
    updateAttentionIntroOverlay();
    state.attentionIntro.overlayTimer = null;
  }, 260);
}

function initAttentionIntroSlide() {
  const slide = document.getElementById('slide-18');
  const tokenBand = document.getElementById('attn18-token-band');
  if (!slide || !tokenBand) return;

  if (!state.attentionIntro.initialized) {
    tokenBand.innerHTML = '';
    ATTN_INTRO_TOKENS.forEach((token) => {
      tokenBand.appendChild(createAttentionTokenPrimitive(token));
    });

    if (!state.attentionIntro.resizeBound) {
      addTrackedListener(window, 'resize', () => {
        if (!state.attentionIntro.initialized) return;
        clearAttentionIntroFlowTimers();
        updateAttentionIntroOverlay();
        syncAttentionIntroFlowVisuals(getAttentionIntroFlowCount(state.attentionIntro.step));
      });
      state.attentionIntro.resizeBound = true;
    }

    state.attentionIntro.initialized = true;
  }

  const takeaway = document.getElementById('attn18-takeaway');
  if (takeaway) takeaway.textContent = ATTN_INTRO_FOOTER;
  typesetMath(slide).then(() => {
    updateAttentionIntroOverlay();
  });
  updateAttentionIntroOverlay();
  syncAttentionIntroFlowVisuals(getAttentionIntroFlowCount(state.attentionIntro.step));
}

function runAttentionIntroStep() {
  if (!state.attentionIntro.initialized) initAttentionIntroSlide();
  if (state.attentionIntro.step >= ATTN_INTRO_MAX_STEP) return false;
  setAttentionIntroStep(state.attentionIntro.step + 1);
  return true;
}

function resetAttentionIntroSlide() {
  const slide = document.getElementById('slide-18');
  if (!slide) return;
  if (state.attentionIntro.overlayTimer) {
    clearTimeout(state.attentionIntro.overlayTimer);
    state.attentionIntro.overlayTimer = null;
  }
  clearAttentionIntroFlowTimers();
  setAttentionIntroStep(0);
}
