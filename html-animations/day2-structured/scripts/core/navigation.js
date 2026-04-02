function refreshSlides() {
  state.nav.slides = Array.from($$('.slide'));
  state.nav.total = state.nav.slides.length;
  registerSlides();
  state.registry.order.forEach((descriptor) => {
    const root = document.getElementById(descriptor.id);
    descriptor.build(root, state);
  });
}

const STEP_STATE_CONTROLLERS = {
  'slide-3': {
    get: () => architectureOverviewState.step,
    set: (step) => setArchitectureOverviewStep(step)
  },
  'slide-18': {
    get: () => attentionIntroState.step,
    set: (step) => setAttentionIntroStep(step)
  },
  'slide-19': {
    get: () => attentionP1State.step,
    set: (step) => setAttentionP1Step(step)
  },
  'slide-20': {
    get: () => attentionQkvState.step,
    set: (step) => setAttentionQkvStep(step)
  },
  'slide-21': {
    get: () => attentionWeightsState.step,
    set: (step) => setAttentionWeightsStep(step)
  },
  'slide-22': {
    get: () => attentionStep4State.step,
    set: (step) => setAttentionStep4Step(step)
  },
  'slide-23': {
    get: () => attentionMatrixState.step,
    set: (step) => setAttentionMatrixStep(step)
  },
  'slide-24': {
    get: () => attentionMultiHeadState.step,
    set: (step) => setAttentionMultiHeadStep(step)
  },
  'slide-26': {
    get: () => positionSignalState.step,
    set: (step) => setPositionSignalStep(step)
  },
  'slide-28': {
    get: () => gptBlockState.step,
    set: (step) => setGptBlockStep(step)
  },
  'slide-32': {
    get: () => outputHeadState.step,
    set: (step) => setOutputHeadStep(step)
  },
  'slide-33': {
    get: () => generationState.step,
    set: (step) => setGenerationStep(step)
  }
};

function getCurrentSlide() {
  return state.nav.slides[state.nav.current] || null;
}

function getSlideIndexById(slideId) {
  if (!slideId) return -1;
  return state.nav.slides.findIndex((slide) => slide.id === slideId);
}

function syncSlideHash(slideEl) {
  if (!slideEl || !slideEl.id) return;
  const nextHash = '#' + slideEl.id;
  if (window.location.hash === nextHash) return;
  window.history.replaceState(null, '', window.location.pathname + window.location.search + nextHash);
}

function getInitialSlideIndex() {
  const hash = window.location.hash.replace(/^#/, '');
  if (!hash) return 0;
  const slideId = decodeURIComponent(hash);
  const matchIndex = getSlideIndexById(slideId);
  return matchIndex >= 0 ? matchIndex : 0;
}

function handleHashNavigation() {
  const nextIndex = getInitialSlideIndex();
  if (nextIndex === state.nav.current) return;
  goToSlide(nextIndex);
}

function getStepController(slideId) {
  return STEP_STATE_CONTROLLERS[slideId] || null;
}

function updateNavigationUi() {
  state.ui.btnPrev.disabled = state.nav.history.length === 0 && state.nav.current === 0;
  state.ui.btnNext.textContent = state.nav.current === state.nav.total - 1 ? 'Restart ↺' : 'Next →';
  state.ui.btnSkip.textContent = state.nav.current === state.nav.total - 1 ? 'Restart ↺' : 'Skip Slide →';
  state.ui.slideCounter.textContent = (state.nav.current + 1) + ' / ' + state.nav.total;
  state.ui.progressFill.style.width = ((state.nav.current / Math.max(state.nav.total - 1, 1)) * 100) + '%';
}

function getAutostepRevealCount(slideEl) {
  if (!slideEl) return 0;
  return slideEl.querySelectorAll('.hidden-content.autostep.revealed').length;
}

function restoreAutostepRevealCount(slideEl, count) {
  if (!slideEl) return;
  const autosteps = Array.from(slideEl.querySelectorAll('.hidden-content.autostep'));
  autosteps.forEach((el, idx) => {
    const reveal = idx < count;
    el.classList.toggle('revealed', reveal);
    el.classList.toggle('settled', reveal);
  });
}

function getProjectionLensOrder(slideEl) {
  const toolbar = slideEl ? slideEl.querySelector('#projectionToolbar16') : null;
  if (!toolbar) return [];
  return Array.from(toolbar.querySelectorAll('.lens-btn'))
    .map((btn) => btn.dataset.lens)
    .filter((lens, idx, arr) => lens && LENS_STATES[lens] && arr.indexOf(lens) === idx);
}

function restoreProjectionLens(lensKey) {
  const nextLens = LENS_STATES[lensKey] ? lensKey : DEFAULT_PROJECTION_LENS;

  if (!projectionState.initialized) {
    initProjectionSlide();
  }

  if (projectionState.rafId) {
    cancelAnimationFrame(projectionState.rafId);
    projectionState.rafId = null;
  }
  if (projectionState.readoutTimer) {
    clearTimeout(projectionState.readoutTimer);
    projectionState.readoutTimer = null;
  }

  projectionState.activeLens = nextLens;
  projectionState.currentPositions = cloneLensState(nextLens);
  projectionState.animFrom = null;
  projectionState.animTo = null;
  projectionState.animStart = 0;

  syncProjectionButtons(nextLens);
  updateProjectionReadout(nextLens);
  resizeProjectionCanvas();
  drawProjection();
}

function createHistorySnapshot() {
  const activeSlide = getCurrentSlide();
  if (!activeSlide) return null;

  const controller = getStepController(activeSlide.id);
  const snapshot = {
    index: state.nav.current,
    slideId: activeSlide.id,
    autostepCount: getAutostepRevealCount(activeSlide)
  };

  if (activeSlide.id === 'slide-16') {
    snapshot.projectionLens = projectionState.activeLens;
  }
  if (controller) {
    snapshot.step = controller.get();
  }

  return snapshot;
}

function snapshotsEqual(a, b) {
  if (!a || !b) return false;
  return (
    a.index === b.index &&
    a.slideId === b.slideId &&
    a.autostepCount === b.autostepCount &&
    a.projectionLens === b.projectionLens &&
    a.step === b.step
  );
}

function commitHistorySnapshot(before) {
  const after = createHistorySnapshot();
  if (!before || !after || snapshotsEqual(before, after)) return false;
  state.nav.history.push(before);
  updateNavigationUi();
  return true;
}

function runMutationWithHistory(mutator) {
  const before = createHistorySnapshot();
  mutator();
  return commitHistorySnapshot(before);
}

function restoreHistorySnapshot(snapshot) {
  if (!snapshot) return false;
  const targetSlide = state.nav.slides[snapshot.index];
  if (!targetSlide || targetSlide.id !== snapshot.slideId) return false;

  goToSlide(snapshot.index);

  const activeSlide = getCurrentSlide();
  if (!activeSlide || activeSlide.id !== snapshot.slideId) return false;

  if (snapshot.slideId === 'slide-16') {
    restoreProjectionLens(snapshot.projectionLens);
  }

  const controller = getStepController(snapshot.slideId);
  if (controller) {
    controller.set(snapshot.step || 0);
  }

  restoreAutostepRevealCount(activeSlide, snapshot.autostepCount || 0);
  typesetMath(activeSlide);
  scheduleDeckRefresh({ reason: 'history-restore' });
  return true;
}

function resetSlideInteractions(slideEl) {
  if (!slideEl) return;
  const descriptor = getSlideDescriptorById(slideEl.id);
  descriptor.reset(state, slideEl);
  captureSnapshot('reset:' + slideEl.id);
}

function resetHiddenContent(slideEl) {
  if (!slideEl) return;
  slideEl.querySelectorAll('.hidden-content').forEach((el) => {
    el.classList.remove('revealed', 'settled');
  });
  resetSlideInteractions(slideEl);
}

function goToSlide(index) {
  if (state.nav.total === 0) return;
  if (index < 0 || index >= state.nav.total) return;

  state.nav.slides.forEach((slide, i) => {
    slide.classList.remove('active');
    if (i !== index) resetHiddenContent(slide);
  });

  const activeSlide = state.nav.slides[index];
  activeSlide.classList.add('active');
  state.nav.current = index;
  syncSlideHash(activeSlide);

  updateNavigationUi();

  typesetMath(activeSlide);
  const descriptor = getSlideDescriptorById(activeSlide.id);
  descriptor.init(state, activeSlide);
  scheduleDeckRefresh({ reason: 'goToSlide' });
  captureSnapshot('goToSlide:' + activeSlide.id);
}

function goTo(index) {
  goToSlide(index);
}

function nextSlide() {
  if (state.nav.total === 0) return;
  runMutationWithHistory(() => {
    if (state.nav.current === state.nav.total - 1) {
      goToSlide(0);
      return;
    }
    goToSlide(state.nav.current + 1);
  });
}

function prevSlide() {
  goToSlide(state.nav.current - 1);
}

function prevWithInteractions() {
  if (state.nav.history.length === 0) {
    if (state.nav.current > 0) {
      prevSlide();
      updateNavigationUi();
      return true;
    }
    updateNavigationUi();
    return false;
  }

  const snapshot = state.nav.history.pop();
  if (!restoreHistorySnapshot(snapshot)) {
    updateNavigationUi();
    return false;
  }

  updateNavigationUi();
  return true;
}

function nextStep() {
  if (state.nav.total === 0) return false;
  const active = getCurrentSlide();
  if (!active) return false;
  const descriptor = getSlideDescriptorById(active.id);
  const before = createHistorySnapshot();
  if (descriptor.step(active, state)) {
    commitHistorySnapshot(before);
    scheduleDeckRefresh({ reason: 'step' });
    captureSnapshot('nextStep:' + active.id);
    return true;
  }
  const beforeAutoStep = createHistorySnapshot();
  if (runAutoStep(active)) {
    commitHistorySnapshot(beforeAutoStep);
    scheduleDeckRefresh({ reason: 'autostep' });
    captureSnapshot('nextStep:auto:' + active.id);
    return true;
  }
  return false;
}

function prevStep() {
  return prevWithInteractions();
}

function runAutoStep(slideEl) {
  const next = slideEl.querySelector('.hidden-content.autostep:not(.revealed)');
  if (!next) return false;
  next.classList.add('revealed');
  return true;
}

function nextWithInteractions() {
  if (nextStep()) return;
  nextSlide();
}

function runProjectionLensStep(slideEl) {
  if (!slideEl || slideEl.id !== 'slide-16' || !projectionState.initialized) return false;

  const toolbar = document.getElementById('projectionToolbar16');
  if (!toolbar) return false;
  const lensOrder = Array.from(toolbar.querySelectorAll('.lens-btn'))
    .map((btn) => btn.dataset.lens)
    .filter((lens, idx, arr) => lens && LENS_STATES[lens] && arr.indexOf(lens) === idx);
  if (lensOrder.length === 0) return false;

  const currentIdx = lensOrder.indexOf(projectionState.activeLens);
  if (currentIdx === -1) {
    setProjectionLens(lensOrder[0]);
    return true;
  }
  if (currentIdx >= lensOrder.length - 1) return false;

  setProjectionLens(lensOrder[currentIdx + 1]);
  return true;
}
