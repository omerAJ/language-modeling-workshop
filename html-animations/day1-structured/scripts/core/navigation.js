function refreshSlides() {
  state.nav.slides = state.nav.order
    .map((id) => document.getElementById('slide-' + id))
    .filter(Boolean);
  state.nav.total = state.nav.slides.length;
  registerSlides();
  state.registry.order.forEach((descriptor) => {
    const root = document.getElementById(descriptor.id);
    descriptor.build(root, state);
  });
}

function getCurrentSlide() {
  return state.nav.slides[state.nav.current] || null;
}

function updateNavigationUi() {
  state.ui.btnPrev.disabled = state.nav.history.length === 0;
  state.ui.btnNext.textContent = state.nav.current === state.nav.total - 1 ? 'Restart ↺' : 'Next →';
  state.ui.btnSkip.textContent = state.nav.current === state.nav.total - 1 ? 'Restart ↺' : 'Skip Slide →';
  state.ui.slideCounter.textContent = (state.nav.current + 1) + ' / ' + state.nav.total;
  state.ui.progressFill.style.width = ((state.nav.current / Math.max(state.nav.total - 1, 1)) * 100) + '%';
}

function cloneHistoryData() {
  return {
    inference: {
      step: inferenceState.step,
      pendingBlank: inferenceState.pendingBlank
    },
    training: {
      phase: trainingState.phase
    },
    completion: {
      fixesRevealed: Object.assign({}, completionState.fixesRevealed),
      pendingBridge: completionState.pendingBridge
    }
  };
}

function applyHistoryData(data) {
  const snapshotData = data || {};
  const inferenceData = snapshotData.inference || {};
  const trainingData = snapshotData.training || {};
  const completionData = snapshotData.completion || {};

  inferenceState.step = typeof inferenceData.step === 'number' ? inferenceData.step : 0;
  inferenceState.pendingBlank = !!inferenceData.pendingBlank;
  inferenceState.blankTimer = null;

  trainingState.phase = typeof trainingData.phase === 'number' ? trainingData.phase : 0;

  completionState.fixesRevealed = Object.assign(
    { chat: false, math: false, code: false },
    completionData.fixesRevealed || {}
  );
  completionState.pendingBridge = !!completionData.pendingBridge;
  completionState.bridgeTimer = null;
}

function createHistorySnapshot() {
  const activeSlide = getCurrentSlide();
  if (!activeSlide) return null;
  return {
    index: state.nav.current,
    slideId: activeSlide.id,
    html: activeSlide.innerHTML,
    data: cloneHistoryData()
  };
}

function snapshotsEqual(a, b) {
  if (!a || !b) return false;
  return (
    a.index === b.index &&
    a.slideId === b.slideId &&
    a.html === b.html &&
    a.data.inference.step === b.data.inference.step &&
    a.data.inference.pendingBlank === b.data.inference.pendingBlank &&
    a.data.training.phase === b.data.training.phase &&
    !!a.data.completion.fixesRevealed.chat === !!b.data.completion.fixesRevealed.chat &&
    !!a.data.completion.fixesRevealed.math === !!b.data.completion.fixesRevealed.math &&
    !!a.data.completion.fixesRevealed.code === !!b.data.completion.fixesRevealed.code &&
    a.data.completion.pendingBridge === b.data.completion.pendingBridge
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

function restorePendingInteractiveState(slideEl) {
  if (!slideEl) return;
  if (slideEl.id === 'slide-5') resumeInferenceBlank();
  if (slideEl.id === 'slide-32') resumeCompletionBridgeReveal();
}

function resetSlideInteractions(slideEl) {
  if (!slideEl) return;
  slideEl.querySelectorAll('.hidden-content').forEach(function(el) {
    el.classList.remove('revealed', 'settled');
  });
  const descriptor = getSlideDescriptorById(slideEl.id);
  descriptor.reset(state, slideEl);
}

function goToSlide(index) {
  if (state.nav.total === 0) return;
  if (index < 0 || index >= state.nav.total) return;

  clearTrackedTimeouts();
  const nextSlideEl = state.nav.slides[index];
  state.nav.slides.forEach((slide) => {
    if (slide !== nextSlideEl) {
      resetSlideInteractions(slide);
    }
    slide.classList.remove('active');
  });

  nextSlideEl.classList.add('active');
  state.nav.current = index;
  updateNavigationUi();

  const descriptor = getSlideDescriptorById(nextSlideEl.id);
  descriptor.init(state, nextSlideEl);
  typesetMath(nextSlideEl);
  scheduleActiveSlideFit({ reason: 'goToSlide' });
}

function goTo(index) {
  goToSlide(index);
}

function nextSlide() {
  if (state.nav.total === 0) return;
  runMutationWithHistory(function() {
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
    updateNavigationUi();
    return false;
  }

  const snapshot = state.nav.history.pop();
  if (!snapshot) {
    updateNavigationUi();
    return false;
  }

  clearTrackedTimeouts();
  const nextSlideEl = state.nav.slides[snapshot.index];
  if (!nextSlideEl || nextSlideEl.id !== snapshot.slideId) {
    updateNavigationUi();
    return false;
  }

  state.nav.slides.forEach((slide, index) => {
    if (index !== snapshot.index) {
      resetSlideInteractions(slide);
    }
    slide.classList.remove('active');
  });

  applyHistoryData(snapshot.data);
  nextSlideEl.innerHTML = snapshot.html;
  nextSlideEl.classList.add('active');
  state.nav.current = snapshot.index;
  updateNavigationUi();
  restorePendingInteractiveState(nextSlideEl);
  typesetMath(nextSlideEl);
  scheduleActiveSlideFit({ reason: 'history-restore' });
  return true;
}

function runAutoStep(slideEl) {
  const next = slideEl.querySelector('.hidden-content.autostep:not(.revealed)');
  if (!next) return false;
  next.classList.add('revealed');
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
    scheduleActiveSlideFit({ reason: 'step' });
    return true;
  }
  const beforeAutoStep = createHistorySnapshot();
  if (runAutoStep(active)) {
    commitHistorySnapshot(beforeAutoStep);
    scheduleActiveSlideFit({ reason: 'autostep' });
    return true;
  }
  return false;
}

function nextWithInteractions() {
  if (nextStep()) return;
  nextSlide();
}
