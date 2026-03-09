function devLog() {
  if (!DEV) return;
  console.log.apply(console, ['[DEV]'].concat(Array.from(arguments)));
}

function devAssert(condition, message, context) {
  if (!DEV || condition) return;
  if (typeof context === 'undefined') {
    console.warn('[DEV][ASSERT]', message);
    return;
  }
  console.warn('[DEV][ASSERT]', message, context);
}

function captureSnapshot(label) {
  if (!DEV) return;
  const activeSlide = state.nav.slides[state.nav.current];
  const attn18 = document.getElementById('slide-18');
  const attn19 = document.getElementById('slide-19');
  const snapshot = {
    label,
    currentIndex: state.nav.current,
    currentSlideId: activeSlide ? activeSlide.id : null,
    revealedCountOnActive: activeSlide ? activeSlide.querySelectorAll('.hidden-content.revealed').length : 0,
    projectionLens: projectionState.activeLens,
    attentionIntroStep: attentionIntroState.step,
    attentionIntroClasses: attn18
      ? ['attn18-show-vectors', 'attn18-show-focus', 'attn18-show-flow', 'attn18-show-update'].filter((c) => attn18.classList.contains(c))
      : [],
    attentionQkvStep: attentionQkvState.step,
    attentionQkvCompareDone: attentionQkvState.compareDone,
    attentionQkvClasses: attn19
      ? ['attn19-show-k', 'attn19-show-v', 'attn19-show-q', 'attn19-show-compare', 'attn19-compare-done'].filter((c) => attn19.classList.contains(c))
      : []
  };
  state.dev.snapshots.push(snapshot);
  devLog('snapshot', snapshot);
}

function runDevStartupChecks() {
  if (!DEV) return;
  const expectedOrder = [
    'slide-0', 'slide-1', 'slide-2', 'slide-3', 'slide-4',
    'slide-5', 'slide-6', 'slide-7', 'slide-8', 'slide-9',
    'slide-10', 'slide-11', 'slide-12', 'slide-13', 'slide-14',
    'slide-16', 'slide-17', 'slide-18', 'slide-19'
  ];
  const foundOrder = state.nav.slides.map((el) => el.id);
  devAssert(state.nav.total === 19, 'Slide count mismatch', { expected: 19, actual: state.nav.total });
  devAssert(
    expectedOrder.join('|') === foundOrder.join('|'),
    'Slide order mismatch',
    { expected: expectedOrder, actual: foundOrder }
  );
  devAssert(document.querySelectorAll('.hidden-content').length === 11, 'hidden-content count mismatch', {
    expected: 11,
    actual: document.querySelectorAll('.hidden-content').length
  });
  devAssert(document.querySelectorAll('.hidden-content.autostep').length === 11, 'autostep count mismatch', {
    expected: 11,
    actual: document.querySelectorAll('.hidden-content.autostep').length
  });
  devAssert(Boolean(state.ui.btnPrev), 'Missing #btnPrev');
  devAssert(Boolean(state.ui.btnNext), 'Missing #btnNext');
  devAssert(Boolean(state.ui.btnSkip), 'Missing #btnSkip');
  devAssert(Boolean(state.ui.progressFill), 'Missing #progressFill');
  devAssert(Boolean(state.ui.slideCounter), 'Missing #slideCounter');
  devLog('listener counts', state.dev.listenerCounts);
}
