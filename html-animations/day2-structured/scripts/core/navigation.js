function refreshSlides() {
  state.nav.slides = Array.from($$('.slide'));
  state.nav.total = state.nav.slides.length;
  registerSlides();
  state.registry.order.forEach((descriptor) => {
    const root = document.getElementById(descriptor.id);
    descriptor.build(root, state);
  });
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

  state.ui.btnPrev.disabled = state.nav.current === 0;
  state.ui.btnNext.textContent = state.nav.current === state.nav.total - 1 ? 'Restart ↺' : 'Next →';
  state.ui.btnSkip.textContent = state.nav.current === state.nav.total - 1 ? 'Restart ↺' : 'Skip Slide →';
  state.ui.slideCounter.textContent = (state.nav.current + 1) + ' / ' + state.nav.total;
  state.ui.progressFill.style.width = ((state.nav.current / Math.max(state.nav.total - 1, 1)) * 100) + '%';

  typesetMath(activeSlide);
  const descriptor = getSlideDescriptorById(activeSlide.id);
  descriptor.init(state, activeSlide);
  captureSnapshot('goToSlide:' + activeSlide.id);
}

function goTo(index) {
  goToSlide(index);
}

function nextSlide() {
  if (state.nav.total === 0) return;
  if (state.nav.current === state.nav.total - 1) {
    goToSlide(0);
    return;
  }
  goToSlide(state.nav.current + 1);
}

function prevSlide() {
  goToSlide(state.nav.current - 1);
}

function nextStep() {
  if (state.nav.total === 0) return false;
  const active = state.nav.slides[state.nav.current];
  if (!active) return false;
  const descriptor = getSlideDescriptorById(active.id);
  if (descriptor.step(active, state)) {
    captureSnapshot('nextStep:' + active.id);
    return true;
  }
  if (runAutoStep(active)) {
    captureSnapshot('nextStep:auto:' + active.id);
    return true;
  }
  return false;
}

function prevStep() {
  prevSlide();
  return true;
}

function runAutoStep(slideEl) {
  const next = slideEl.querySelector('.hidden-content.autostep:not(.revealed)');
  if (!next) return false;
  next.classList.add('revealed');
  return true;
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
