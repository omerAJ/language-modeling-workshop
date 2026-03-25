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
    revealIngredient(Number(ingredientBtn.dataset.ingredient));
    return;
  }

  const trainBtn = event.target.closest('[data-train-phase]');
  if (trainBtn) {
    trainStep(Number(trainBtn.dataset.trainPhase));
    return;
  }

  const suspectBtn = event.target.closest('[data-suspect]');
  if (suspectBtn) {
    chooseDramaSuspect(suspectBtn.dataset.suspect, suspectBtn);
    return;
  }

  const fixBtn = event.target.closest('[data-fix]');
  if (fixBtn) {
    revealFix(fixBtn.dataset.fix);
    return;
  }

  const actionBtn = event.target.closest('[data-action]');
  if (actionBtn) {
    const action = actionBtn.dataset.action;
    if (action === 'reveal-reasoning-pressure') revealReasoningPressure();
    if (action === 'reveal-drama-answer') revealDramaAnswer();
    if (action === 'reveal-all-ntp') revealAllNTP();
    if (action === 'reveal-dumb-response') revealDumbResponse();
    return;
  }

  const inferenceBlank = event.target.closest('#infCurrent');
  if (inferenceBlank) {
    animateInference();
    return;
  }

  const ntpBlank = event.target.closest('.ntp-blank');
  if (ntpBlank) {
    revealNTP(ntpBlank);
    return;
  }

  const ntpConceptBtn = event.target.closest('.ntp-concept-btn');
  if (ntpConceptBtn) {
    revealNTPConcept(ntpConceptBtn);
  }
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
    prevSlide();
  }
}

function handleTransitionEnd(e) {
  var el = e.target;
  if (!el.classList.contains('hidden-content') || e.propertyName !== 'max-height') return;
  if (el.classList.contains('revealed')) {
    el.classList.add('settled');
  } else {
    el.classList.remove('settled');
  }
}

function bindGlobalEvents() {
  state.ui.btnNext.addEventListener('click', nextWithInteractions);
  state.ui.btnSkip.addEventListener('click', nextSlide);
  state.ui.btnPrev.addEventListener('click', prevSlide);
  document.addEventListener('click', handleDelegatedClick);
  document.addEventListener('keydown', handleKeydown);
  document.addEventListener('transitionend', handleTransitionEnd);
}
