cacheUiReferences();
refreshSlides();
initializeDeckScaleSystem();

addTrackedListener(state.ui.btnNext, 'click', nextWithInteractions);
addTrackedListener(state.ui.btnSkip, 'click', nextSlide);
addTrackedListener(state.ui.btnPrev, 'click', prevWithInteractions);
addTrackedListener(window, 'hashchange', handleHashNavigation);

addTrackedListener(document, 'keydown', (e) => {
  const tag = e.target && e.target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (e.target && e.target.isContentEditable)) return;

  if (e.key === ' ') {
    e.preventDefault();
    nextWithInteractions();
  }
  if (e.key === 'ArrowRight') {
    e.preventDefault();
    nextWithInteractions();
  }
  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    prevWithInteractions();
  }
});

addTrackedListener(document, 'transitionend', (e) => {
  const el = e.target;
  if (!el.classList.contains('hidden-content') || e.propertyName !== 'max-height') return;
  if (el.classList.contains('revealed')) el.classList.add('settled');
  else el.classList.remove('settled');
});

runDevStartupChecks();
goToSlide(getInitialSlideIndex());
captureSnapshot('init');
