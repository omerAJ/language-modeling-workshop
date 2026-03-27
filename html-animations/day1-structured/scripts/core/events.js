function cacheUiReferences() {
  state.ui.btnPrev = $('#btnPrev');
  state.ui.btnNext = $('#btnNext');
  state.ui.btnSkip = $('#btnSkip');
  state.ui.slideCounter = $('#slideCounter');
  state.ui.progressFill = $('#progressFill');
}

function handleDelegatedClick(event) {
  const noteToggle = event.target.closest('.notes-toggle');
  if (noteToggle) {
    const notesBody = noteToggle.nextElementSibling;
    if (notesBody) notesBody.classList.toggle('open');
    return;
  }

  const ingredientBtn = event.target.closest('[data-ingredient]');
  if (ingredientBtn) {
    if (runMutationWithHistory(function() {
      revealIngredient(Number(ingredientBtn.dataset.ingredient));
    })) {
      scheduleActiveSlideFit({ reason: 'click-reveal' });
    }
    return;
  }

  const trainBtn = event.target.closest('[data-train-phase]');
  if (trainBtn) {
    if (runMutationWithHistory(function() {
      trainStep(Number(trainBtn.dataset.trainPhase));
    })) {
      scheduleActiveSlideFit({ reason: 'click-reveal' });
    }
    return;
  }

  const suspectBtn = event.target.closest('[data-suspect]');
  if (suspectBtn) {
    if (runMutationWithHistory(function() {
      chooseDramaSuspect(suspectBtn.dataset.suspect, suspectBtn);
    })) {
      scheduleActiveSlideFit({ reason: 'click-reveal' });
    }
    return;
  }

  const fixBtn = event.target.closest('[data-fix]');
  if (fixBtn) {
    if (runMutationWithHistory(function() {
      revealFix(fixBtn.dataset.fix);
    })) {
      scheduleActiveSlideFit({ reason: 'click-reveal' });
    }
    return;
  }

  const liveDemoPreset = event.target.closest('[data-demo-preset]');
  if (liveDemoPreset) {
    triggerLiveDemoPreset(liveDemoPreset.dataset.demoPreset);
    return;
  }

  const liveDemoClear = event.target.closest('[data-demo-clear]');
  if (liveDemoClear) {
    clearLiveFailureDemo();
    return;
  }

  const actionBtn = event.target.closest('[data-action]');
  if (actionBtn) {
    if (runMutationWithHistory(function() {
      const action = actionBtn.dataset.action;
      if (action === 'reveal-reasoning-pressure') revealReasoningPressure();
      if (action === 'reveal-drama-answer') revealDramaAnswer();
      if (action === 'reveal-dumb-response') revealDumbResponse();
    })) {
      scheduleActiveSlideFit({ reason: 'click-reveal' });
    }
    return;
  }

  const inferenceBlank = event.target.closest('#infCurrent');
  if (inferenceBlank) {
    if (runMutationWithHistory(function() {
      animateInference();
    })) {
      scheduleActiveSlideFit({ reason: 'click-reveal' });
    }
    return;
  }

  const ntpBlank = event.target.closest('.ntp-blank');
  if (ntpBlank) {
    if (runMutationWithHistory(function() {
      revealNTP(ntpBlank);
    })) {
      scheduleActiveSlideFit({ reason: 'click-reveal' });
    }
    return;
  }

  const ntpConceptBtn = event.target.closest('.ntp-concept-btn');
  if (ntpConceptBtn) {
    if (runMutationWithHistory(function() {
      revealNTPConcept(ntpConceptBtn);
    })) {
      scheduleActiveSlideFit({ reason: 'click-reveal' });
    }
  }
}

function handleSubmit(event) {
  const liveDemoForm = event.target.closest('#liveDemoForm');
  if (!liveDemoForm) return;
  event.preventDefault();
  submitLiveDemoPrompt();
}

function handleKeydown(e) {
  const tag = e.target && e.target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (e.target && e.target.isContentEditable)) return;

  if (e.key === ' ') {
    e.preventDefault();
    nextWithInteractions();
    return;
  }

  if (e.key === 'ArrowRight') {
    e.preventDefault();
    nextWithInteractions();
  }
  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    prevWithInteractions();
  }
  if (e.key === 'n' || e.key === 'N') {
    document.body.classList.toggle('show-notes');
    scheduleActiveSlideFit({ reason: 'toggle-notes' });
  }
}

function handleTransitionEnd(e) {
  var el = e.target;
  if (!el.classList.contains('hidden-content') || e.propertyName !== 'max-height') return;
  if (el.classList.contains('revealed')) {
    el.classList.add('settled');
    scheduleActiveSlideFit({ reason: 'reveal-settled', dispatchResize: false });
  } else {
    el.classList.remove('settled');
  }
}

function bindGlobalEvents() {
  state.ui.btnNext.addEventListener('click', nextWithInteractions);
  state.ui.btnSkip.addEventListener('click', nextSlide);
  state.ui.btnPrev.addEventListener('click', prevWithInteractions);
  document.addEventListener('click', handleDelegatedClick);
  document.addEventListener('submit', handleSubmit);
  document.addEventListener('keydown', handleKeydown);
  document.addEventListener('transitionend', handleTransitionEnd);
}
