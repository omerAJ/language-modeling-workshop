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

function resetSlideInteractions(slideEl) {
  if (!slideEl) return;
  const descriptor = getSlideDescriptorById(slideEl.id);
  descriptor.reset(state, slideEl);
}

function goToSlide(index) {
  if (state.nav.total === 0) return;
  if (index < 0 || index >= state.nav.total) return;

  const nextSlideEl = state.nav.slides[index];
  state.nav.slides.forEach((slide) => {
    if (slide !== nextSlideEl && slide.classList.contains('active')) {
      resetSlideInteractions(slide);
    }
    slide.classList.remove('active');
  });

  nextSlideEl.classList.add('active');
  state.nav.current = index;

  state.ui.btnPrev.disabled = state.nav.current === 0;
  state.ui.btnNext.textContent = state.nav.current === state.nav.total - 1 ? 'Restart ↺' : 'Next →';
  state.ui.btnSkip.textContent = state.nav.current === state.nav.total - 1 ? 'Restart ↺' : 'Skip Slide →';
  state.ui.slideCounter.textContent = (state.nav.current + 1) + ' / ' + state.nav.total;
  state.ui.progressFill.style.width = ((state.nav.current / Math.max(state.nav.total - 1, 1)) * 100) + '%';

  const descriptor = getSlideDescriptorById(nextSlideEl.id);
  descriptor.init(state, nextSlideEl);
  typesetMath(nextSlideEl);
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
  if (descriptor.step(active, state)) return true;
  if (runAutoStep(active)) return true;
  return false;
}

function nextWithInteractions() {
  if (nextStep()) return;
  nextSlide();
}
